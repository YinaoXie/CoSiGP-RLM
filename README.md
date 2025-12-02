# CoSiGP_RLM
## Dependency
`pip install -r requirements.txt`


## CoSiGP Training
external/internal
Step 1: `python pre_train.py`
Step 2: `python train.py`

## RLM Training
Refer to the work of (https://github.com/yt556677/ReFICR) and (https://github.com/ContextualAI/gritlm)

### Data
External scoring datasets can be downloaded from the following links: (https://grouplens.org/datasets/movielens/)

Internal interaction data will be uploaded later.

### Model Weight Download
We will upload our model weight to huggingface later.


cd CoSiGP_RLM
### Recommendation
#### Performance of candidate items before and after retrieval enhancement

`CUDA_VISIBLE_DEVICES=0 python main.py --config config/embedding/inspired_config.yaml`
`CUDA_VISIBLE_DEVICES=0 python main.py --config config/embedding/redial_config.yaml`

### Generation
#### Response Generation
`CUDA_VISIBLE_DEVICES=0 python main.py --config config/generation/inspired_config.yaml`
`CUDA_VISIBLE_DEVICES=0 python main.py --config config/generation/redial_config.yaml`



