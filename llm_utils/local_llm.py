from transformers import AutoTokenizer, AutoModel, LlamaForCausalLM
import json
import time

class Local_llm_handler:
    def __init__(self, model_name):
        self.model_name = model_name
        with open("mapping/local_llm_path.json", "r") as f:
            local_llm_path = json.load(f)
        self.model_path = local_llm_path[model_name]
        # self.model_path = "/home/chenxuanzhong/llama/new_llama-2-7b"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.model_name == "chatglm3-6b":
            self.model = AutoModel.from_pretrained(self.model_path)
        elif self.model_name in ["llama2-7b", "llama2-13b", "llama2-70b"]:
            self.model = LlamaForCausalLM.from_pretrained(self.model_path)
        
        self.model.eval()

    def get_completion(self, system_prompt, prompt, seed=42):
        try:
            t = time.time()
            if self.model_name == "chatglm3-6b":
                result, history = self.model.chat(self.tokenizer, system_prompt + prompt, history=[])
            elif self.model_name in ["llama2-7b", "llama2-13b", "llama2-70b"]:
                inputs = self.tokenizer(system_prompt + prompt, return_tensors="pt")
                generate_ids = self.model.generate(inputs.input_ids)
                result = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                result = result.replace(system_prompt + prompt, "")
            print(f'Local LLM {self.model_name} time: {time.time() - t}')
            return result
        except Exception as e:
            print(e)
            return None

    