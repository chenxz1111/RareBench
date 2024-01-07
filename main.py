from llm_utils.api import Openai_api_handler, Zhipuai_api_handler
from llm_utils.local_llm import Local_llm_handler
import argparse
from utils.mydataset import RareDataset
from utils.evaluation import diagnosis_evaluate
import os
from prompt import RarePrompt
import json
import numpy as np
import re

np.random.seed(42)

def diagnosis_metric_calculate(folder):
    # handler = Openai_api_handler("gpt4")
    handler = Openai_api_handler("chatgpt")
    # handler = Openai_api_handler("chatgpt_instruct")
    # handler = Zhipuai_api_handler("chatglm_turbo")
    CNT = 0
    metric = {}
    recall_top_k = []
    for file in os.listdir(folder):
        file = os.path.join(folder, file)
        res = json.load(open(file, "r", encoding="utf-8-sig"))
        
        predict_rank = res["predict_rank"]
        if res['predict_diagnosis'] is None:
            print(file, "predict_diagnosis is None")
        if predict_rank is None:
            predict_rank = diagnosis_evaluate(res["predict_diagnosis"], res["golden_diagnosis"], handler)
            res["predict_rank"] = predict_rank
            json.dump(res, open(file, "w", encoding="utf-8-sig"), indent=4, ensure_ascii=False)

        if "否" in predict_rank:
            recall_top_k.append(11)
        else:
            print(file)
            pattern = r'\b(?:10|[1-9])\b'
            predict_rank = re.findall(pattern, predict_rank)
            if len(predict_rank) == 0 or len(predict_rank) > 1:
                CNT += 1
                res["predict_rank"] = None
                json.dump(res, open(file, "w", encoding="utf-8-sig"), indent=4, ensure_ascii=False)
                continue
                raise Exception("predict_rank error")
            predict_rank = predict_rank[0]
            recall_top_k.append(int(predict_rank))
        
    metric['recall_top_1'] = len([i for i in recall_top_k if i <= 1]) / len(recall_top_k)
    metric['recall_top_3'] = len([i for i in recall_top_k if i <= 3]) / len(recall_top_k)
    metric['recall_top_10'] = len([i for i in recall_top_k if i <= 10]) / len(recall_top_k)
    metric['medain_rank'] = np.median(recall_top_k)
    print(folder)
    print(metric)
    print("predict_rank error: ", CNT)
    print("evaluate tokens: ", handler.gpt4_tokens, handler.chatgpt_tokens, handler.chatgpt_instruct_tokens)
        
def generate_random_few_shot_id(exclude_id, total_num, k_shot=3):
    few_shot_id = []
    while len(few_shot_id) < k_shot:
        id = np.random.randint(0, total_num)
        if id not in few_shot_id and id not in exclude_id:
            few_shot_id.append(id)
    return few_shot_id


def run_task(task_type, dataset:RareDataset, handler, results_folder, few_shot):
    rare_prompt = RarePrompt()
    if task_type == "diagnosis":
        patient_info_type = dataset.dataset_type
        os.makedirs(results_folder, exist_ok=True)
        print("Begin diagnosis.....")
        print("total patient: ", len(dataset.patient))
        for i, patient in enumerate(dataset.patient):
            if handler is None:
                break
            result_file = os.path.join(results_folder, f"patient_{i}.json")
            if os.path.exists(result_file):
                continue
            patient_info = patient[0]
            golden_diagnosis = patient[1]
            few_shot_info = []
            if few_shot:
                few_shot_id = generate_random_few_shot_id([i], len(dataset.patient))
                for id in few_shot_id:
                    few_shot_info.append((dataset.patient[id][0], dataset.patient[id][1]))

            system_prompt, prompt = rare_prompt.diagnosis_prompt(patient_info_type, patient_info, few_shot_info)
            predict_diagnosis = handler.get_completion(system_prompt, prompt)
            if predict_diagnosis is None:
                print(f"patient {i} predict diagnosis is None")
                continue
            # predict_rank = diagnosis_evaluate(predict_diagnosis, golden_diagnosis)
            predict_rank = None
            res = {
                "patient_info": patient_info,
                "golden_diagnosis": golden_diagnosis,
                "predict_diagnosis": predict_diagnosis,
                "predict_rank": predict_rank
            }
            json.dump(res, open(result_file, "w", encoding="utf-8-sig"), indent=4, ensure_ascii=False)
            print(f"patient {i} finished")
            print("total tokens: ", handler.gpt4_tokens, handler.chatgpt_tokens, handler.chatgpt_instruct_tokens)
        
        # diagnosis_metric_calculate(results_folder)
    elif task_type == "mdt":
        pass
        


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_type', type=str, default="diagnosis", choices=["diagnosis", "mdt"])
    parser.add_argument('--dataset_name', type=str, default="PUMCH_MDT")
    parser.add_argument('--dataset_type', type=str, default="PHENOTYPE", choices=["EHR", "PHENOTYPE", "MDT"])
    parser.add_argument('--dataset_path', default='./datasets/PUMCH/PUMCH_MDT.json')
    # parser.add_argument('--dataset_path', default='./test.json')
    parser.add_argument('--results_folder', default='./results/PUMCH')
    parser.add_argument('--model', type=str, default="gpt4", choices=["gpt4", "chatgpt", "chatglm_turbo", "chatglm3-6b", "llama2-7b", "llama2-13b", "llama2-70b"])
    parser.add_argument('--few_shot', type=bool, default=False)

    args = parser.parse_args()

    try:
        if args.model in ["gpt4", "chatgpt"]:
            handler = Openai_api_handler(args.model)
        elif args.model in ["chatglm_turbo"]:
            handler = Zhipuai_api_handler(args.model)
        elif args.model in ["chatglm3-6b", "llama2-7b", "llama2-13b", "llama2-70b"]:
            handler = Local_llm_handler(args.model)
    except Exception as e:
        handler = None

    dataset = RareDataset(args.dataset_name, args.dataset_path, args.dataset_type)
    few_shot = "_few_shot" if args.few_shot else ""
    results_folder = os.path.join(args.results_folder, args.dataset_name, args.model+"_"+args.task_type+few_shot)
    run_task(args.task_type, dataset, handler, results_folder, args.few_shot)

    if args.model in ["gpt4", "chatgpt"]:
        print(f"OpenAI API total tokens: {handler.gpt4_tokens}", f"ChatGPT API total tokens: {handler.chatgpt_tokens}")

if __name__ == "__main__":
    main()