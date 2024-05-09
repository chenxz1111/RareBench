import json
from typing import List, Optional, Tuple

class RarePrompt:
    def __init__(self) -> None:
        self.system_prompt = "You are a specialist in the field of rare diseases."
        self.diagnosis_system_prompt = self.system_prompt + \
                                    " You will be provided and asked about a complicated clinical case; read it carefully and then provide a diverse and comprehensive differential diagnosis."
        
        self.num_mapping = json.load(open("mapping/num.json", "r"))

    def diagnosis_prompt(self, patient_info_type, patient_info: str, cot: str, few_shot: Optional[List[Tuple[str, str]]] = None) -> Tuple[str, str]:
        # patient_info_type = "EHR" or "PHENOTYPE"
        if patient_info_type == "EHR" or patient_info_type == "MDT":
            info_type = "electronic medical record"
        elif patient_info_type == "PHENOTYPE":
            info_type = "phenotype"
        prompt = ""
        if few_shot and len(few_shot) > 0:
            prompt += f"Let me give you {len(few_shot)} examples first:\n"
            for i, shot in enumerate(few_shot):
                prompt += f"The {self.num_mapping[str(i+1)]} patient has a rare disease [{shot[1]}], and his/her {info_type} is as follows: [{shot[0]}].\n"
            prompt += "Next is the patient case you need to diagnose:"
        prompt += f"Patient's {info_type}: {patient_info}\n"
        prompt += "Enumerate the top 10 most likely diagnoses. Be precise, listing one diagnosis per line, and try to cover many unique possibilities (at least 10). "
        if cot == "zero-shot":
            prompt += "Let us think step by step, you must think more steps. "
        prompt += "The top 10 diagnoses are:"
        
        # elif cot == "none":
            
        return (self.diagnosis_system_prompt, prompt)
    
    
    
