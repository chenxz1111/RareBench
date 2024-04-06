MODEL="mistral-7b"

CUDA_VISIBLE_DEVICES=1 python main.py --model=$MODEL --dataset_name="MME" --dataset_path="./datasets/Multi-Country/MME.json" --results_folder="./results/Multi-Country" --few_shot="dynamic"

CUDA_VISIBLE_DEVICES=1 python main.py --model=$MODEL --dataset_name="RAMEDIS" --dataset_path="./datasets/Multi-Country/RAMEDIS.json" --results_folder="./results/Multi-Country" --few_shot="dynamic"

CUDA_VISIBLE_DEVICES=1 python main.py --model=$MODEL --dataset_name="LIRICAL" --dataset_path="./datasets/Multi-Country/LIRICAL.json" --results_folder="./results/Multi-Country" --few_shot="dynamic"


CUDA_VISIBLE_DEVICES=1 python main.py --model=$MODEL --dataset_name="PUMCH_MDT" --dataset_path="./datasets/PUMCH/PUMCH_MDT.json" --results_folder="./results/PUMCH" --few_shot="dynamic"

CUDA_VISIBLE_DEVICES=1 python main.py --model=$MODEL --dataset_name="PUMCH_L" --dataset_path="./datasets/PUMCH/PUMCH_L.json" --results_folder="./results/PUMCH" --few_shot="dynamic"



CUDA_VISIBLE_DEVICES=1 python main.py --model=$MODEL --dataset_name="MME" --dataset_path="./datasets/Multi-Country/MME.json" --results_folder="./results/Multi-Country" --few_shot="random"

CUDA_VISIBLE_DEVICES=1 python main.py --model=$MODEL --dataset_name="RAMEDIS" --dataset_path="./datasets/Multi-Country/RAMEDIS.json" --results_folder="./results/Multi-Country" --few_shot="random"

CUDA_VISIBLE_DEVICES=1 python main.py --model=$MODEL --dataset_name="LIRICAL" --dataset_path="./datasets/Multi-Country/LIRICAL.json" --results_folder="./results/Multi-Country" --few_shot="random"


CUDA_VISIBLE_DEVICES=1 python main.py --model=$MODEL --dataset_name="PUMCH_MDT" --dataset_path="./datasets/PUMCH/PUMCH_MDT.json" --results_folder="./results/PUMCH" --few_shot="random"

CUDA_VISIBLE_DEVICES=1 python main.py --model=$MODEL --dataset_name="PUMCH_L" --dataset_path="./datasets/PUMCH/PUMCH_L.json" --results_folder="./results/PUMCH" --few_shot="random"

# python main.py --model="mistral-7b" --dataset_name="LIRICAL" --dataset_path="./datasets/Multi-Country/LIRICAL.json" --results_folder="./results/Multi-Country" --eval