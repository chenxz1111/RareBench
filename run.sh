MODEL="biomistral-7b"

CUDA_VISIBLE_DEVICES=0 python main.py --model=$MODEL

CUDA_VISIBLE_DEVICES=0 python main.py --model=$MODEL --dataset_name="MME" --results_folder="./results/Multi-Country" 

CUDA_VISIBLE_DEVICES=0 python main.py --model=$MODEL --dataset_name="HMS" --results_folder="./results/Multi-Country" 

CUDA_VISIBLE_DEVICES=0 python main.py --model=$MODEL --dataset_name="RAMEDIS" --results_folder="./results/Multi-Country" 

CUDA_VISIBLE_DEVICES=0 python main.py --model=$MODEL --dataset_name="LIRICAL" --results_folder="./results/Multi-Country" 