from llm_utils.api import Openai_api_handler, Zhipuai_api_handler, Gemini_api_handler
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

def diagnosis_metric_calculate(folder, judge_model="gpt4"):
    handler = Openai_api_handler(judge_model)
    
    # handler = Openai_api_handler("chatgpt")
    # handler = Gemini_api_handler("gemini_pro")
    # handler = Openai_api_handler("chatgpt_instruct")
    # handler = Zhipuai_api_handler("chatglm_turbo")
    CNT = 0
    metric = {}
    recall_top_k = []

    Pediatrics = range(0, 15)
    Neurology = range(30, 45)
    Cardiology = range(15, 30)
    Nephrology = range(45, 60)
    Hematology = range(60, 75)


    for file in os.listdir(folder):
    #   if int(file.split("_")[1].split(".")[0]) in Pediatrics:
        file = os.path.join(folder, file)
        res = json.load(open(file, "r", encoding="utf-8-sig"))
    
        predict_rank = res["predict_rank"]
        if res['predict_diagnosis'] is None:
            print(file, "predict_diagnosis is None")
        
        if predict_rank not in ["否", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]:
            print(file)
            CNT += 1
            # res["predict_rank"] = predict_rank[0]
            # res["predict_rank"] = "否"
            # res["predict_rank"] = None
            # json.dump(res, open(file, "w", encoding="utf-8-sig"), indent=4, ensure_ascii=False)
        
        
        if predict_rank is None:
            
            predict_rank = diagnosis_evaluate(res["predict_diagnosis"], res["golden_diagnosis"], handler)
            res["predict_rank"] = predict_rank
            json.dump(res, open(file, "w", encoding="utf-8-sig"), indent=4, ensure_ascii=False)
        # res["predict_rank"] = None
        # json.dump(res, open(file, "w", encoding="utf-8-sig"), indent=4, ensure_ascii=False)
        # continue
        if "否" in predict_rank:
            recall_top_k.append(11)
        else:
            # print(file)
            pattern = r'\b(?:10|[1-9])\b'
            predict_rank = re.findall(pattern, predict_rank)
            if len(predict_rank) == 0 or len(predict_rank) > 1:
                
                res["predict_rank"] = None
                print(file)
                # input()
                # json.dump(res, open(file, "w", encoding="utf-8-sig"), indent=4, ensure_ascii=False)
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

def generate_dynamic_few_shot_id(methods, exclude_id, dataset, k_shot=3):
    few_shot_id = []

    patient = dataset.load_hpo_code_data()
    if methods == "dynamic":
        phe2embedding = json.load(open("mapping/phe2embedding.json", "r", encoding="utf-8-sig"))
    elif methods == "medprompt":
        phe2embedding = json.load(open("mapping/medprompt_emb.json", "r", encoding="utf-8-sig"))
    ic_dict = json.load(open("mapping/ic_dict.json", "r", encoding="utf-8-sig"))
    if methods == "medprompt":
        ic_dict = {k: 1 for k, _ in ic_dict.items()}
   
    exclude_patient = patient[exclude_id]
    exclude_patient_embedding = np.array([np.array(phe2embedding[phe]) for phe in exclude_patient[0] if phe in phe2embedding])
    exclude_patient_ic = np.array([ic_dict[phe] for phe in exclude_patient[0] if phe in phe2embedding])
    exclude_patient_embedding = np.sum(exclude_patient_embedding * exclude_patient_ic.reshape(-1, 1), axis=0) / np.sum(exclude_patient_ic)
    candidata_embedding_list = []
    for i, p in enumerate(patient):
        phe_embedding = np.array([np.array(phe2embedding[phe]) for phe in p[0] if phe in phe2embedding])
        ic_coefficient_list = np.array([ic_dict[phe] for phe in p[0] if phe in phe2embedding])
        phe_embedding = np.sum(phe_embedding * ic_coefficient_list.reshape(-1, 1), axis=0) / np.sum(ic_coefficient_list)
        candidata_embedding_list.append(phe_embedding)
    candidata_embedding_list = np.array(candidata_embedding_list)
    cosine_sim = np.dot(candidata_embedding_list, exclude_patient_embedding) 
    cosine_sim = np.argsort(cosine_sim)[::-1]
    for i in cosine_sim:
        if i not in few_shot_id and i != exclude_id:
            few_shot_id.append(i)
        if len(few_shot_id) == k_shot:
            break
    # print(exclude_id, few_shot_id)
    # print(patient[exclude_id][1])
    # print('----------------')
    # for id in few_shot_id:
    #     print(patient[id][1])
    # print('\n')
    return few_shot_id


def run_task(task_type, dataset:RareDataset, handler, results_folder, few_shot, cot, judge_model, eval=False):
    few_shot_dict = {}
    rare_prompt = RarePrompt()
    if task_type == "diagnosis":
        patient_info_type = dataset.dataset_type
        os.makedirs(results_folder, exist_ok=True)
        print("Begin diagnosis.....")
        print("total patient: ", len(dataset.patient))
        ERR_CNT = 0
        questions = []
        for i, patient in enumerate(dataset.patient):
            if handler is None:
                print("handler is None")
                break
            result_file = os.path.join(results_folder, f"patient_{i}.json")
            if os.path.exists(result_file):
                continue
            patient_info = patient[0]
            golden_diagnosis = patient[1]
            few_shot_info = []
            if few_shot == "random":
                few_shot_id = generate_random_few_shot_id([i], len(dataset.patient))
                
                few_shot_dict[i] = few_shot_id
                for id in few_shot_id:
                    few_shot_info.append((dataset.patient[id][0], dataset.patient[id][1]))
            elif few_shot == "dynamic" or few_shot == "medprompt":
                few_shot_id = generate_dynamic_few_shot_id(few_shot, i, dataset)
                # print(few_shot_id)
                # input()
                # quit()
                few_shot_dict[str(i)] = [str(idx) for idx in few_shot_id]
                for id in few_shot_id:
                    few_shot_info.append((dataset.patient[id][0], dataset.patient[id][1]))

            system_prompt, prompt = rare_prompt.diagnosis_prompt(patient_info_type, patient_info, cot, few_shot_info)
            # print(system_prompt + prompt)
            # quit()
            questions.append(system_prompt + prompt)
            if few_shot == "auto-cot":
                autocot_example = json.load(open("mapping/autocot_example.json", "r", encoding="utf-8-sig"))
                system_prompt = "Here a some examples: " + autocot_example[handler.model_name] + system_prompt
                prompt = prompt + "Let us think step by step.\n"
            
            predict_diagnosis = handler.get_completion(system_prompt, prompt)
            if predict_diagnosis is None:
                print(f"patient {i} predict diagnosis is None")
                ERR_CNT += 1
                continue
            
            predict_rank = None
            res = {
                "patient_info": patient_info,
                "golden_diagnosis": golden_diagnosis,
                "predict_diagnosis": predict_diagnosis,
                "predict_rank": predict_rank
            }
            json.dump(res, open(result_file, "w", encoding="utf-8-sig"), indent=4, ensure_ascii=False)
            print(f"patient {i} finished")
            if type(handler) == Openai_api_handler:
                print("total tokens: ", handler.gpt4_tokens, handler.chatgpt_tokens, handler.chatgpt_instruct_tokens)
        if eval:
            diagnosis_metric_calculate(results_folder, judge_model=judge_model)
        print("diagnosis ERR_CNT: ", ERR_CNT)
    elif task_type == "mdt":
        pass
    # json.dump(questions, open(os.path.join("", "questions.json"), "w", encoding="utf-8-sig"), indent=4, ensure_ascii=False)
    # json.dump(few_shot_dict, open(os.path.join("dynamic_few-shot", f"{dataset.dataset_name}.json"), "w", encoding="utf-8-sig"), indent=4, )
        


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_type', type=str, default="diagnosis", choices=["diagnosis", "mdt"])
    parser.add_argument('--dataset_name', type=str, default="PUMCH_ADM")
    parser.add_argument('--dataset_type', type=str, default="PHENOTYPE", choices=["EHR", "PHENOTYPE", "MDT"])
    parser.add_argument('--dataset_path', default='./datasets/PUMCH/PUMCH_ADM.json')
    parser.add_argument('--results_folder', default='./results/PUMCH')
    parser.add_argument('--model', type=str, default="gpt4", choices=["gpt4", "chatgpt", "glm4", "glm3_turbo", "gemini_pro", "mistral-7b", "chatglm3-6b", "llama2-7b", "llama2-13b", "llama2-70b", "clinical-T5", "huatuogpt2-7b", "biomistral-7b", "medalpaca-7b"])
    parser.add_argument('--judge_model', type=str, default="gpt4", choices=["gpt4", "chatgpt"])
    parser.add_argument('--few_shot', type=str, default="none", choices=["none", "random", "dynamic", "medprompt", "auto-cot"])
    parser.add_argument('--cot', type=str, default="none", choices=["none", "zero-shot"])
    parser.add_argument('--eval', action='store_true')

    args = parser.parse_args()

    # try:
    if args.model in ["gpt4", "chatgpt"]:
        handler = Openai_api_handler(args.model)
    elif args.model in ["glm4", "glm3_turbo"]:
        handler = Zhipuai_api_handler(args.model)
    elif args.model in ["gemini_pro"]:
        handler = Gemini_api_handler(args.model)
    elif args.model in ["mistral-7b", "chatglm3-6b", "llama2-7b", "llama2-13b", "llama2-70b", "clinical-T5", "huatuogpt2-7b", "biomistral-7b", "medalpaca-7b"]:
        handler = Local_llm_handler(args.model)
    # except Exception as e:
    #     handler = None

    dataset = RareDataset(args.dataset_name, args.dataset_path, args.dataset_type)
    
    if args.few_shot == "none":
        few_shot = ""
    elif args.few_shot == "random":
        few_shot = "_few_shot"
    elif args.few_shot == "dynamic":
        few_shot = "_dynamic_few_shot"
    elif args.few_shot == "medprompt":
        few_shot = "_medprompt"
    elif args.few_shot == "auto-cot":
        few_shot = "_auto-cot"
    if args.cot == "none":
        cot = ""
    elif args.cot == "zero-shot":
        cot = "_cot"
    results_folder = os.path.join(args.results_folder, args.dataset_name, args.model+"_"+args.task_type+few_shot+cot)
    run_task(args.task_type, dataset, handler, results_folder, args.few_shot, args.cot, args.judge_model, args.eval)

    if args.model in ["gpt4", "chatgpt"]:
        print(f"OpenAI API total tokens: {handler.gpt4_tokens}", f"ChatGPT API total tokens: {handler.chatgpt_tokens}")

if __name__ == "__main__":
    main()