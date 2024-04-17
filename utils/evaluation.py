from llm_utils.api import Openai_api_handler, Zhipuai_api_handler

def diagnosis_evaluate(predict_diagnosis, golden_diagnosis, handler):
    if predict_diagnosis is None:
        raise Exception("Predict diagnosis is None")
        
    predict_diagnosis = predict_diagnosis.replace("\n\n\n", "")

    system_prompt = "You are a classifier."
    prompt = "我现在会给你十种预测疾病，如果预测诊断在标准诊断中。请输出预测的位次，否则输出“否”，只输出“否”或“1-10”的数字，不要输出额外的内容。"
    prompt += f'预测诊断：{predict_diagnosis}\n'
    prompt += f'标准诊断：{golden_diagnosis}\n'
    
    
    print("Begin evaluation.....")
    rank = handler.get_completion(system_prompt, prompt)
    rank = rank.replace("\n", "")
    # if len(rank) > 1:
    #     return None
    return rank
    