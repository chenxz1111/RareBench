from llm_utils.api import Openai_api_handler, Zhipuai_api_handler
import argparse
from utils.mydataset import RareDataset
from utils.evaluation import diagnosis_evaluate
import os
from prompt import RarePrompt
import json


def run_task(task_type, dataset:RareDataset, handler, results_folder):
    rare_prompt = RarePrompt()
    if task_type == "diagnosis":
        patient_info_type = dataset.dataset_type
        os.makedirs(results_folder, exist_ok=True)
        print("Begin diagnosis.....")
        print("total patient: ", len(dataset.patient))
        for i, patient in enumerate(dataset.patient):
            result_file = os.path.join(results_folder, f"patient_{i}.json")
            if os.path.exists(result_file):
                continue
            patient_info = patient[0]
            golden_diagnosis = patient[1]
            system_prompt, prompt = rare_prompt.diagnosis_prompt(patient_info_type, patient_info)
            predict_diagnosis = handler.get_completion(system_prompt, prompt)
            predict_rank = diagnosis_evaluate(predict_diagnosis, golden_diagnosis)
            res = {
                "patient_info": patient_info,
                "golden_diagnosis": golden_diagnosis,
                "predict_diagnosis": predict_diagnosis,
                "predict_rank": predict_rank
            }
            with open(result_file, "w", encoding="utf-8-sig") as f:
                json.dump(res, f, indent=4, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_type', type=str, default="diagnosis", choices=["diagnosis"])
    parser.add_argument('--dataset_name', type=str, default="PUMCH_ADM")
    parser.add_argument('--dataset_type', type=str, default="PHENOTYPE", choices=["EHR", "PHENOTYPE"])
    # parser.add_argument('--dataset_path', default='./datasets/PUMCH/test.json')
    parser.add_argument('--dataset_path', default='./test.json')
    parser.add_argument('--results_folder', default='./results/PUMCH')
    parser.add_argument('--model', type=str, default="gpt4", choices=["gpt4", "chatgpt", "chatglm_turbo", "chatglm3-6b", "llama2-7b", "llama2-13b", "llama2-70b"])

    args = parser.parse_args()

    if args.model in ["gpt4", "chatgpt"]:
        handler = Openai_api_handler(args.model)
    elif args.model in ["chatglm_turbo"]:
        handler = Zhipuai_api_handler(args.model)

    dataset = RareDataset(args.dataset_name, args.dataset_path, args.dataset_type)

    results_folder = os.path.join(args.results_folder, args.dataset_name, args.model+"_"+args.task_type)
    run_task(args.task_type, dataset, handler, results_folder)

if __name__ == "__main__":
    main()