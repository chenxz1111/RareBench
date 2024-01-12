MODEL="llama2-7b"


CUDA_VISIBLE_DEVICES=2 python main.py --model=$MODEL --dataset_name="PUMCH_ADM" --dataset_path="./datasets/PUMCH/PUMCH_ADM.json" --few_shot="dynamic"

CUDA_VISIBLE_DEVICES=2 python main.py --model=$MODEL --dataset_name="PUMCH_ADM" --dataset_path="./datasets/PUMCH/PUMCH_ADM.json" --few_shot="random"

