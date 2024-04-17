from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, T5ForConditionalGeneration
import json
import time
import torch
from transformers.generation.utils import GenerationConfig

class Local_llm_handler:
    def __init__(self, model_name):
        self.model_name = model_name
        with open("mapping/local_llm_path.json", "r") as f:
            local_llm_path = json.load(f)
        self.model_path = local_llm_path[model_name]
        return None
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.model_name == "chatglm3-6b":
            self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True, device='cuda')
        elif self.model_name in ["mistral-7b", "llama2-7b", "llama2-13b", "llama2-70b", "biomistral-7b", "medalpaca-7b"]:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
            
            print(f"Using device: {self.device}")
        elif self.model_name in ["huatuogpt2-7b"]:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
            self.model.generation_config = GenerationConfig.from_pretrained(self.model_path)

        elif self.model_name in ["clinical-T5"]:
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_path, from_flax=True)
        self.model.eval()

    def get_completion(self, system_prompt, prompt, seed=42):
        try:
            t = time.time()
            if self.model_name == "chatglm3-6b":
                result, history = self.model.chat(self.tokenizer, system_prompt + prompt, history=[])
            elif self.model_name in ["llama2-7b", "llama2-13b", "llama2-70b", "medalpaca-7b"]:
                inputs = self.tokenizer(system_prompt + prompt, return_tensors="pt")
                inputs = inputs.to(self.device)
                self.model.to(self.device)
                generate_ids = self.model.generate(inputs.input_ids, max_new_tokens=1000)
                result = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                result = result.replace(system_prompt + prompt, "")
            elif self.model_name in ["mistral-7b", "biomistral-7b"]:
                messages = [
                    {"role": "user", "content": system_prompt + prompt}
                ]
                model_inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt")
                model_inputs = model_inputs.to(self.device)
                self.model.to(self.device)

                generated_ids = self.model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
                result = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                result = result.replace(system_prompt + prompt, "")
                result = result.replace("[INST]", "")
                result = result.replace("[/INST]", "")
            elif self.model_name in ["huatuogpt2-7b"]:
                self.model.to(self.device)
                messages = [
                    {"role": "user", "content": system_prompt + prompt}
                ]
                result = self.model.HuatuoChat(self.tokenizer, messages)
                
            elif self.model_name in ["clinical-T5"]:
                inputs = self.tokenizer(system_prompt + prompt, return_tensors="pt")
                inputs = inputs.to(self.device)
                self.model.to(self.device)
                generate_ids = self.model.generate(inputs.input_ids, max_new_tokens=4096)
                result = self.tokenizer.decode(generate_ids[0])
                print(result)
                
            print(f'Local LLM {self.model_name} time: {time.time() - t}')
            return result
        except Exception as e:
            print(e)
            return None

    