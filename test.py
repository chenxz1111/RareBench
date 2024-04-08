import json
from llm_utils.api import Openai_api_handler, Zhipuai_api_handler, Gemini_api_handler
from llm_utils.local_llm import Local_llm_handler

f_name = "mapping/example.json"

q = json.load(open(f_name, "r", encoding="utf-8-sig"))

prompt_example = json.load(open("mapping/autocot_example.json", "r", encoding="utf-8-sig"))

# model = "gemini_pro"
# api_handler = Gemini_api_handler(model)

# model = "glm3_turbo"
# api_handler = Zhipuai_api_handler(model)

model = 'mistral-7b'
api_handler = Local_llm_handler(model)

p = ''

for i in q:
    p += i
    completion = api_handler.get_completion(i, '')
    p += completion
    p += "\n\n\n"
prompt_example[model] = p
json.dump(prompt_example, open("mapping/autocot_example.json", "w", encoding="utf-8-sig"), ensure_ascii=False, indent=4)
   