from llm_utils.api import Openai_api_handler, Zhipuai_api_handler

def diagnosis_evaluate(predict_diagnosis, golden_diagnosis, handler):
    if predict_diagnosis is None:
        raise Exception("Predict diagnosis is None")
        
    predict_diagnosis = predict_diagnosis.replace("\n\n\n", "")

    system_prompt = "You are a specialist in the field of rare diseases."
    prompt = 'I will now give you ten predicted diseases if the predicted diagnosis is in the standard diagnosis. Please output the predicted rank, otherwise output "No", only output "No" or "1-10" numbers, if the predicted disease has multiple conditions, only output the top rank. Output only "No" or one number, no additional output.'
    prompt += f'Predicted diseases: {predict_diagnosis}\n'
    prompt += f'Standard diagnosis: {golden_diagnosis}\n'
    
    print("Begin evaluation.....")
    rank = handler.get_completion(system_prompt, prompt)
    rank = rank.replace("\n", "")
    
    return rank
    