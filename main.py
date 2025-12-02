from gritlm import GritLM
from scipy.spatial.distance import cosine
import json
from jsonargparse import CLI
import torch.nn.functional as F
import torch
from tqdm import tqdm
import numpy as np
from transformers import set_seed, AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType,PeftModel
import os
from utils import search_number,extract_movie_name, recall_score, add_roles, is_float, load_sparse_matrix, EASE, recommend_top_k,get_last_turn_line_numbers,hebin3_recall,min_max_normalize_rows,hebin4_recall

import time
import re
import pickle
from scipy.sparse import csr_matrix
from scipy.linalg import cho_factor, cho_solve

import logging
# 配置日志记录
logging.basicConfig(
    filename="CoSiGP_RLM/log/recall.log",  # 日志文件名
    level=logging.INFO,        # 日志级别
    format="%(asctime)s - %(levelname)s - %(message)s"  # 日志格式
)

def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False
        
#merge the model weights
def apply_lora(base_model_path, target_model_path):


    model = GritLM(base_model_path, low_cpu_mem_usage=True, torch_dtype="auto")
    
    if os.path.exists(os.path.join(target_model_path, 'non_lora_trainables.bin')):
        non_lora_trainables = torch.load(os.path.join(target_model_path, 'non_lora_trainables.bin'), map_location='cpu')
        print(non_lora_trainables)
        model.load_state_dict(non_lora_trainables, strict=False)

    #peft model
    print(f"Loading LoRA weights from {target_model_path}")
    lora_model = PeftModel.from_pretrained(model.model, target_model_path)
    print(f"Merging weights")
    model.model = lora_model.merge_and_unload()
    return model

def gritlm_instruction(instruction):
    return "<|user|>\n" + instruction + "\n<|embed|>\n" if instruction else "<|embed|>\n"


def get_instruction(data, task_type, gen_instr):

    output = []
    for example in data:
        context = example["context"]
        
        if task_type == "Ranking":
        
            num = 10
            cand_dict = example["cand_list"]
            top_k_items = {k: cand_dict[k] for k in list(cand_dict)[:10]}
            cand_items = ""
            for key, value in top_k_items.items():
                cand_items += f"[{str(key)}] {str(value)}\n"        
            
            rag_kg_conv = ' '.join(example["re_kg"]["context"][-4:])
            rag_kg_target = example["re_kg"]["target"]
            if len(rag_kg_target) > 0:
                target = ', '.join(rag_kg_target)
            else:
                target = rag_kg_target[0]
            retrieved_kg = f"Users with intentions similar to the current user were recommended {target} by the system. The refered content is: {rag_kg_conv[-512:]}"


            pre_prompt = gen_instr.format(cand_items,context[-512:],retrieved_kg,num)
            #pre_prompt = gen_instr.format(cand_items,context[-512:],num)
            #pre_prompt = gen_instr.format(context[-512:],retrieved_kg,cand_items)

        if task_type == "Dialoge_Manage":
            pre_prompt = gen_instr.format(context[-516:])

        if task_type == "Response_Gen":
            recommend_item = " ".join(example["rec"])
            pre_prompt = gen_instr.format(context[-516:],recommend_item)

        #print("pre_prompt:",pre_prompt)
        messages = [{ 
                        "role":"user",
                        "content":pre_prompt}]

        output.append(messages)

    return output


def extract_prev_entity_by_lineno(jsonl_path, item2idx_path):
    """
    读取jsonl文件，返回{行号: {"matched": [...], "unmatched": [...]}}的字典（行号从0开始）
    prev_entity中的每个item都用item2idx.json中对应的id替换，未匹配到的原样保留
    """
    # 加载item2idx映射
    with open(item2idx_path, 'r', encoding='utf-8') as f:
        item2idx = json.load(f)

    result = {}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if line.strip():
                obj = json.loads(line)
                # Score_entity = obj.get('initiatorQuestionsScore', {})
                # all_cooper = obj.get('all_cooper', [])
                prev_entity = obj.get('prev_entity', [])
                rec_id = obj.get('rec_id', [])
                # 从prev_entity中剔除掉rec_id的内容
                filtered_cooper = [item for item in prev_entity if item not in rec_id]
                matched = []
                unmatched = []
                for item in filtered_cooper:
                    key = str(item)
                    if key in item2idx:
                        matched.append(item)
                    else:
                        unmatched.append(item)
                result[idx] = {"rec_id": rec_id, "matched": matched, "unmatched": unmatched}#保存的是数值
    return result

def recommend_top_k_for_users(user_result, item2idx_path, B, k=50, vector_len=4861):
    """
    针对每个用户的 matched 列表，构造 one-hot 向量，与 B 相乘，返回 top-k 推荐下标。

    参数:
        user_result (dict): {用户编号: {"matched": [...], "unmatched": [...]}}
        item2idx (dict): {item(str): idx(int)}
        B (np.ndarray): 物品-物品权重矩阵 (shape: [n_items, n_items])
        k (int): 推荐 top-k 个物品
        vector_len (int): 向量长度（物品总数）

    返回:
        dict: {用户编号: [top-k下标列表]}
    """
    # 加载item2idx映射
    with open(item2idx_path, 'r', encoding='utf-8') as f:
        item2idx = json.load(f)
     # 构建 idx2item 映射
    idx2item = {int(v): k for k, v in item2idx.items()}

    recommendations = {}
    recommendations_score = {}
    for user_idx, info in user_result.items():
        matched = info.get("matched", [])
        rec_id = info.get("rec_id", [])


        if not matched:
            continue

        # 构造 one-hot 向量
        x = np.zeros(vector_len, dtype=np.float32)
        # x = np.ones(vector_len, dtype=np.float32)
        for item in matched:
            key = str(item)
            if key in item2idx:
                idx = item2idx[key]
                x[idx] = 1.0  # 用分数赋值
        # 计算分数
        y = x @ B  # shape: (vector_len,)
        # 把 matched 的部分在 y 中置为0，防止推荐已交互物品
        for item in matched:
            key = str(item)
            if key in item2idx:
                idx = item2idx[key]
                y[idx] = 0.0
        # 根据分数从高到低排序，获取前k个物品的下标
        top_k_idx = np.argsort(-y)[:k]
        top_k_scores = y[top_k_idx].tolist()

        recommendations[user_idx] = top_k_idx.tolist()
        recommendations_score[user_idx] = top_k_scores
         # 将下标转换为真实name
        top_k_names = [idx2item[idx] for idx in top_k_idx if idx in idx2item]
        recommendations[user_idx] = top_k_names

    return recommendations, recommendations_score

def recommend_ease_internal():
    # 1. 加载训练数据
    train_path = "CoSiGP_RLM/cf/data/redial/user_item_matrix.npz"
    X_train = load_sparse_matrix(train_path)

    # 2. 计算每列的交互总数（即每个项目的所有用户交互总数）
    item_interaction_sums = np.array(X_train.sum(axis=0)).flatten()

    # 3. 初始化并训练 EASE 模型
    ease = EASE(l2_reg=100)
    ease.fit(X_train)

    # 4. 进行推荐并保存结果
    user2idx_path = "CoSiGP_RLM/cf/data/redial/user2idx.json"
    item2idx_path = "CoSiGP_RLM/cf/data/redial/item2idx.json"
    test_jsonl_path = "CoSiGP_RLM/cf/data/redial/data/test_output.jsonl"

    B = ease.B
    pred = ease.pred

    recommendations, recommendations_score, popularity_list = recommend_top_k(
        pred=pred,
        user_item_matrix=X_train,
        test_jsonl_path=test_jsonl_path,
        user2idx_path=user2idx_path,
        item2idx_path=item2idx_path,
        item_interaction_sums=item_interaction_sums,
        k=200
    )
    logging.info(f"recommendations_internal 长度{len(recommendations)},recommendations_internal具体内容：{recommendations}")
    logging.info(f"popularity_list 长度{len(popularity_list)},具体内容：{popularity_list}")
    return recommendations, recommendations_score, popularity_list

def recommend_ease_external():
    # 1. 加载训练数据
    train_path = "CoSiGP_RLM/cf/data/external/redial/user_item_matrix.npz"
    X_train = load_sparse_matrix(train_path)
    X_train_norm = min_max_normalize_rows(X_train)
    # 2. 计算每列的交互总数（即每个项目的所有用户交互总数）
    # item_interaction_sums = np.array(X_train.sum(axis=0)).flatten()

    # 3. 初始化并训练 EASE 模型
    ease = EASE(l2_reg=150)
    ease.fit(X_train_norm)
    B = ease.B

    item2idx_path = "CoSiGP_RLM/cf/data/external/redial/item2idx.json"
    jsonl_path= "CoSiGP_RLM/data/test_processed_rec_redial_desc_replaced.jsonl"
    user_result = extract_prev_entity_by_lineno(jsonl_path, item2idx_path)
    recommendations, recommendations_score = recommend_top_k_for_users(user_result, item2idx_path, B, k=200, vector_len=B.shape[1])
    

    logging.info(f"recommendations_external 长度{len(recommendations)},recommendations_external具体内容：{recommendations}")
    return recommendations, recommendations_score

def main(mode:str=None, from_json:str=None, db_json_item:str=None, embeddings_path_item:str=None, to_json_item:str=None,
 base_model_path:str=None,target_model_path:str=None,  db_json_conv:str=None, embeddings_path_conv:str=None, to_json_conv:str=None, 
    stored_cand_lst:bool=True, is_lora:bool=True):
    
    set_seed(123)
    
    if is_lora:
        model = apply_lora(base_model_path,target_model_path)
    else:
        model = GritLM("GritLM/GritLM-7B", torch_dtype="auto")

    with open(from_json) as fd:
        lines = fd.readlines()
        data = [json.loads(line) for line in lines]
        print(len(data))

    if mode == "embedding":

        with open(db_json_item) as fi:
            db_item = json.load(fi)
        print("len(db_item):", len(db_item))
        

        queries = [example['context'][-512:] for example in data]
        print('queries length:',len(queries))


        all_names = list(db_item.keys())
        name2id = {all_names[index]: index for index in range(len(all_names))}  #"电影真实id字符串":序号id
        print("name2id:",len(name2id))
        id2name = {v:k for k,v in name2id.items()}
        # logging.info(f"name2id 长度{len(name2id)},具体内容：{name2id}")
        id2name2 = {}
        for v, k in name2id.items():
            # v 是 db 字典的键
            # db[v] 是对应的字符串
            value_str = db_item[v]
            # 用正则提取 Title: 和 Actors: 之间的内容
            match = re.search(r'Title:\s*(.*?)\s*Actors:', value_str)
            if match:
                movie_name = match.group(1).strip()
            else:
                movie_name = v  # 如果没有匹配到就用原始键
                print(v, "没有匹配到电影名称")
            id2name2[k] = movie_name

        docs = list(db_item.values())
        docs = [doc[:1024] for doc in docs]
        docs_len = [len(doc) for doc in docs]

        # logging.info(f"max docs：{np.max(docs_len)},mean docs:{np.mean(docs_len)},min docs:{np.min(docs_len)},doc length:{len(docs)}")

        query_instr_item="Retrieve semantically relevant movie items descriptions based on user's multi turn historical dialogue context:"
        doc_instr_item="Represent the item for retrieval:"
        
        if os.path.exists(embeddings_path_item):
            print("loading embeddings form file")
            d_rep = torch.load(embeddings_path_item)
        else:
            d_rep = model.encode(docs, instruction=gritlm_instruction(doc_instr_item))
            print('document shape:',torch.from_numpy(d_rep).shape)
            torch.save(d_rep, embeddings_path_item)
            print("saving embeddigns to file ...") 


        #get ground truth item ID
        rec_lists = []
        for example in tqdm(data):
            lst = []
            for item in example['rec_id']:
                if str(item) in all_names:
                    lst.append(name2id[str(item)])
            lst = list(set(lst))
            rec_lists.append(lst)
        
        ###############协同部分代码#################
        recommendations, recommendations_score, popularity_list = recommend_ease_internal()
        recommendations_external, recommendations_score_external = recommend_ease_external()
        
        ###############协同部分代码#################

        num_slice = 8
        step = int(len(queries) / num_slice) + 1
        print('query_step:',step)
        rank = []
        score_lst = []

        for i in range(0,len(queries),step):
            queries_slice = queries[i : i + step]
            rec_lists_slice = rec_lists[i : i + step]


            assert len(queries_slice) == len(rec_lists_slice)
        
            q_rep = model.encode(queries_slice, instruction=gritlm_instruction(query_instr_item))
 

            cos_sim = F.cosine_similarity(torch.from_numpy(q_rep).unsqueeze(1),torch.from_numpy(d_rep).unsqueeze(0),dim=-1)
            cos_sim = torch.where(torch.isnan(cos_sim),torch.full_like(cos_sim,0),cos_sim)
            

            topk_sim_values,topk_sim_indices = torch.topk(cos_sim,k=200,dim=-1)
            rank_slice = topk_sim_indices.tolist()
            score_lst_slice = topk_sim_values.tolist()

            rank += rank_slice
            score_lst += score_lst_slice
            print('length rank:',len(rank))
        
        # logging.info(f"rec_lists 长度{len(rec_lists)},具体内容：{rec_lists}")
        # logging.info(f"rank 长度：{len(rank)},具体内容：{rank}")
        print("优化前召回率：")
        recall_score(rec_lists,rank,ks=[1,5,10,20,50])

        # 保存rec_lists和rank到json文件（id转为name）
        # 将 rec_lists 中的 id 转换回 name
        rec_lists_names = [[id2name[id_] for id_ in sublist] for sublist in rec_lists]
        logging.info(f"rec_lists_names 长度{len(rec_lists_names)},具体内容：{rec_lists_names}")
        # 将 rank 中的 id 转换回 name
        rank_names = [[id2name[id_] for id_ in sublist] for sublist in rank]
        logging.info(f"rank_names 长度{len(rank_names)},具体内容：{rank_names}")

        # with open("/ai/reficrAll/dataAll/reficr_data3/embed/inspired/movie_pro/rec_lists_name.json", "w", encoding="utf-8") as f1:
        #     json.dump(rec_lists_names, f1, ensure_ascii=False, indent=2)
        # with open("/ai/reficrAll/dataAll/reficr_data3/embed/inspired/movie_pro/reficr_top200_name.json", "w", encoding="utf-8") as f2:
        #     json.dump(rank_names, f2, ensure_ascii=False, indent=2)
        # with open("/ai/reficrAll/dataAll/reficr_data3/embed/inspired/movie_pro/reficr_top200_name_score.json", "w", encoding="utf-8") as f2:
        #     json.dump(score_lst, f2, ensure_ascii=False, indent=2)
        result = get_last_turn_line_numbers(from_json)
        print("优化后召回率：")
        # hebin3_recall(rank_names, recommendations,  popularity_list,  result,rec_lists_names, ks=[1,5,10,20,50], verbose=True)
        hebin4_recall(rank_names, score_lst, recommendations, recommendations_score,popularity_list, 
                        recommendations_external, recommendations_score_external, result,
                        rec_lists_names, 
                        ks=[1,5,10,20,50], verbose=True)

        if stored_cand_lst:

            for i in range(len(rank)):

                ranked_list = {j:id2name2[j] for j in rank[i]}

                data[i]["rec_id_end"] = rec_lists[i]
                data[i]["cand_list"] = ranked_list

            with open(to_json_item,"w",encoding="utf-8") as fwr:
                for example in data:
                    fwr.write(json.dumps(example))
                    fwr.write("\n")

        with open(db_json_conv) as fi:
            db_conv = json.load(fi)
        print("len(db_conv):", len(db_conv))

        query_instr_conv="Given a user's conversation history, retrieve conversations from other users with similar intents"
        doc_instr_conv="Represent the conversation context for similar user intention retrieval"

        conv_docs = []
        for dict_conv in db_conv.values():
            context = add_roles(dict_conv['context'])
            conv_docs.append(context)
        print("conv_doc:", conv_docs[0])    
        print('length of conv_docs:',len(conv_docs))

        if os.path.exists(embeddings_path_conv):
            print("loading embeddings form file")
            conv_d_rep = torch.load(embeddings_path_conv)
        else:
            conv_d_rep = model.encode(conv_docs, instruction=gritlm_instruction(doc_instr_conv))
            print('conv doc shape:',torch.from_numpy(conv_d_rep).shape)
            torch.save(conv_d_rep, embeddings_path_conv)
            print("saving embeddigns to file ...")

        conv_q_rep = model.encode(queries, instruction=gritlm_instruction(query_instr_conv))
        print('conv queries shape:',torch.from_numpy(conv_q_rep).shape)
        #normalize
        conv_d_rep = F.normalize(torch.from_numpy(conv_d_rep), p=2, dim=1)
        conv_q_rep = F.normalize(torch.from_numpy(conv_q_rep), p=2, dim=1)

        #compute similarity
        conv_cos_similarities = torch.mm(conv_q_rep, conv_d_rep.t())
        


        cos_similarities = conv_cos_similarities
        topk_conv_values,topk_conv_indices = torch.topk(cos_similarities,k=1,dim=-1)
        conv_indices = topk_conv_indices.tolist()
        print("cos_similarities:",cos_similarities.shape)
        print("topk_conv_values:",topk_conv_values.shape)
        print("topk_conv_indices:",topk_conv_indices.shape)
        for i in range(len(conv_indices)):
            #print(conv_indices[i][0])
            re_kg = db_conv[str(conv_indices[i][0])]
            sim_value = topk_conv_values[i][0]
            #print("re_kg:",re_kg)
            #print("sim_value:",sim_value)
            data[i]["re_kg"] = re_kg
            data[i]["sim_value"] = sim_value.item()

        with open(to_json_conv,"w",encoding="utf-8") as fr:
            for example in data:
                fr.write(json.dumps(example))
                fr.write("\n")

    if mode == "generation":
        tag_Dialoge="Dialoge_Manage"
        gen_instr_Dialoge="Analyze the conversation context: {}\nDetermine the user's intention and recommend a system dialogue action. Provide your explanation and suggested action in the following format:- Explanation: \n- Suggested Action: <a>dialogue action</a>"
        to_json_Dialoge="CoSiGP_RLM/save/test_processed_action2.jsonl"

        outputs_Dialoge = get_instruction(data,tag_Dialoge,gen_instr_Dialoge)

        pred_Dialoge = []
        for messages in tqdm(outputs_Dialoge):
            start_time = time.time()  # 开始计时

            encoded = model.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
            encoded = encoded.to(model.device)
            gen = model.generate(encoded, max_new_tokens=1024, do_sample=False, pad_token_id=2)
            decoded = model.tokenizer.batch_decode(gen)
            print(decoded[0].encode("utf-8").decode("latin1"))
            logging.info(f"{decoded[0].encode('utf-8').decode('latin1')}")

            generated = decoded[0].split("<|assistant|>\n")[-1].replace("</s>","").replace("\n","").strip()
            pred_Dialoge.append(generated)

            end_time = time.time()  # 结束计时
            generation_time = end_time - start_time  # 计算生成时间
            print(f"生成一条消息的时间: {generation_time:.2f} 秒")

        if tag_Dialoge == "Dialoge_Manage":
            
            assert len(pred_Dialoge) == len(data)

            with open(to_json_Dialoge,"w",encoding="utf-8") as fout:
                for e_id in range(len(data)):
                    #print("pred[e_id]:", pred[e_id])
                    data[e_id]["action"] = pred_Dialoge[e_id]
                    fout.write(json.dumps(data[e_id],ensure_ascii=False))
                    fout.write("\n")

        tag_Response="Response_Gen"
        gen_instr_Response="Act as a friendly, knowledgeable movie expert for personalized recs. Follow these rules:\n-Conversational Context: {} - Use to understand the user’s latest request and stay coherent with prior talks.\n-Recommended Items: {} - movie titles only. Use your fine-tuned movie knowledge to explain (e.g., plot highlights, genre traits, standout elements). \n-Response Flow: First, briefly acknowledge the user’s last message. For each movie: Enclose the title in `<item></item>`, then add a natural explanation. End with an engaging question (e.g., \"Do any of these interest you? Want a different vibe?\"). \n-Without Items: Don’t invent movies—ask clarifying questions (e.g., \"What genres do you like? Any favorite actors?\").Respond naturally and succinctly"
       
        to_json_Response="CoSiGP_RLM/save/test_processed_gen2.jsonl"
        outputs_Response = get_instruction(data,tag_Response,gen_instr_Response)

        pred_Response = []
        for messages in tqdm(outputs_Response):

            encoded = model.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
            encoded = encoded.to(model.device)
            gen = model.generate(encoded, max_new_tokens=1024, do_sample=False, pad_token_id=2)
            decoded = model.tokenizer.batch_decode(gen)
            print(decoded[0].encode("utf-8").decode("latin1"))
            logging.info(f"{decoded[0].encode('utf-8').decode('latin1')}")

            generated = decoded[0].split("<|assistant|>\n")[-1].replace("</s>","").replace("\n","").strip()
            pred_Response.append(generated)

        if tag_Response == "Response_Gen":

            assert len(pred_Response) == len(data)

            with open(to_json_Response,"w",encoding="utf-8") as fout:
                for e_id in range(len(data)):
                   
                    if len(data[e_id]["rec"]) == 0:
                        data[e_id]["rec_tag"] = 0
                    else:
                        data[e_id]["rec_tag"] = 1

                    data[e_id]["pred"] = pred_Response[e_id]
                    fout.write(json.dumps(data[e_id],ensure_ascii=False))
                    fout.write("\n")

if __name__ == '__main__':
    CLI(main)
