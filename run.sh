MODEL="mistral-7b"
DEVICE = 3

CUDA_VISIBLE_DEVICES=$DEVICE python main.py --model=$MODEL --dataset_name="MME" --dataset_path="./datasets/Multi-Country/MME.json" --results_folder="./results/Multi-Country"

CUDA_VISIBLE_DEVICES=$DEVICE python main.py --model=$MODEL --dataset_name="HMS" --dataset_path="./datasets/Multi-Country/HMS.json" --results_folder="./results/Multi-Country"

CUDA_VISIBLE_DEVICES=$DEVICE python main.py --model=$MODEL --dataset_name="PUMCH_ADM" --dataset_path="./datasets/PUMCH/PUMCH_ADM.json"

CUDA_VISIBLE_DEVICES=$DEVICE python main.py --model=$MODEL --dataset_name="PUMCH_MDT" --dataset_path="./datasets/PUMCH/PUMCH_MDT.json"

CUDA_VISIBLE_DEVICES=$DEVICE python main.py --model=$MODEL --dataset_name="PUMCH_L" --dataset_path="./datasets/PUMCH/PUMCH_L.json"

CUDA_VISIBLE_DEVICES=$DEVICE python main.py --model=$MODEL --dataset_name="RAMEDIS" --dataset_path="./datasets/Multi-Country/RAMEDIS.json" --results_folder="./results/Multi-Country"