import json
from llm_utils.api import Openai_api_handler, Zhipuai_api_handler, Gemini_api_handler

# f_name = "../../code/RareBench/example.json"

q = ["You are an expert in the field of rare disease. You will be provided and asked about a complicated clinical case; read it carefully and then provide a diverse and comprehensive differential diagnosis.Patient's phenotype: Cholecystitis,Cerebral hemorrhage,Ascites,Abnormal mitral valve morphology,Cardiomegaly,Pericardial effusion,Abnormal tricuspid valve morphology,Right ventricular failure,Accessory spleen,Abdominal pain,Pulmonary arterial hypertension,Dyspnea,Pleural effusion,Sparse scalp hair,Hepatomegaly,Conjugated hyperbilirubinemia,Generalized muscle weakness,Exercise intolerance,Increased total bilirubin,Permanent atrial fibrillation,Pulmonary artery dilatation,Abnormal vena cava morphology,Reduced systolic function,Pedal edema,Decreased urine output,Reduced left ventricular ejection fraction,Left ventricular diastolic dysfunction,Right atrial enlargement,Increased circulating NT-proBNP concentration,Left atrial enlargement\nEnumerate the top 10 most likely diagnoses. Be precise, listing one diagnosis per line, and try to cover many unique possibilities (at least 10). The top 10 diagnoses are:", "You are an expert in the field of rare disease. You will be provided and asked about a complicated clinical case; read it carefully and then provide a diverse and comprehensive differential diagnosis.Patient's phenotype: Oral ulcer,Increased circulating renin level,Syncope,Palpitations,Diarrhea,Dyspnea,Hypokalemia,Hypomagnesemia,Generalized muscle weakness,Lower limb muscle weakness,Paroxysmal vertigo,T-wave inversion,Abnormality of urine calcium concentration,Nodular regenerative hyperplasia of liver,Dysesthesia,Amaurosis fugax\nEnumerate the top 10 most likely diagnoses. Be precise, listing one diagnosis per line, and try to cover many unique possibilities (at least 10). The top 10 diagnoses are:", "You are an expert in the field of rare disease. You will be provided and asked about a complicated clinical case; read it carefully and then provide a diverse and comprehensive differential diagnosis.Patient's phenotype: Proteinuria,Glomerulonephritis,Hearing impairment,Visual loss,Fever,Hypercholesterolemia,Elevated circulating creatinine concentration,Elevated erythrocyte sedimentation rate,Recurrent tonsillitis,Abnormal urinary color,Macroscopic hematuria\nEnumerate the top 10 most likely diagnoses. Be precise, listing one diagnosis per line, and try to cover many unique possibilities (at least 10). The top 10 diagnoses are:"]

prompt_example = json.load(open("autocot_example.json", "r", encoding="utf-8-sig"))

# model = "gemini_pro"
# api_handler = Gemini_api_handler(model)

model = "glm3_turbo"
api_handler = Zhipuai_api_handler(model)

p = ''

for i in q:
    p += i + "Let us think step by step.\n"
    completion = api_handler.get_completion(i, "Let us think step by step.")
    p += completion
    p += "\n\n"
prompt_example[model] = p
json.dump(prompt_example, open("autocot_example.json", "w", encoding="utf-8-sig"), ensure_ascii=False, indent=4)
   