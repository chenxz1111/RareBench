from llm_utils.api import Openai_api_handler, Zhipuai_api_handler

def diagnosis_evaluate(predict_diagnosis, golden_diagnosis):
    if predict_diagnosis is None:
        raise Exception("Predict diagnosis is None")
        
    predict_diagnosis = predict_diagnosis.replace("\n\n\n", "")

    system_prompt = "You are an expert in the field of rare disease."
    prompt = '如果预测诊断在标准诊断中。请输出预测的位次，否则输出“否”，不要输出额外的内容。'
    prompt += f'预测诊断：{predict_diagnosis}'
    prompt += f'标准诊断：{golden_diagnosis}'
    handler = Openai_api_handler("gpt4")
    # handler = Zhipuai_api_handler("chatglm_turbo")
    print("Begin evaluation.....")
    rank = handler.get_completion(system_prompt, prompt)
    return rank
    