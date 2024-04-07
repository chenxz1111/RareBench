import openai
import zhipuai
import google.generativeai as genai
import time

class Openai_api_handler:
    def __init__(self, model) -> None:
        # Put your own key in the llm_utils/gpt_key.txt file
        with open('llm_utils/gpt_key.txt', 'r') as f:
            openai.api_key = f.readline().strip()
        if model == 'gpt4':
            self.model = "gpt-4-1106-preview"
        elif model == 'chatgpt':
            self.model = "gpt-3.5-turbo-1106"
        elif model == 'chatgpt_instruct':
            self.model = "gpt-3.5-turbo-instruct"

        self.gpt4_tokens = 0
        self.chatgpt_tokens = 0
        self.chatgpt_instruct_tokens = 0

    def get_completion(self, system_prompt, prompt, seed=42):
        try:
            t = time.time()
            
            if self.model == "gpt-4-1106-preview" or self.model == "gpt-3.5-turbo-1106":
                completion = openai.chat.completions.create(
                    model=self.model,
                    seed=seed,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ]
                )
                result = {
                    'system_prompt': system_prompt,
                    'question': prompt,
                    'model': str(completion.model),
                    'seed': seed,
                    'usage': {
                        'input_tokens': int(completion.usage.prompt_tokens),
                        'output_tokens': int(completion.usage.completion_tokens),
                    },
                    'answer': str(completion.choices[0].message.content),
                }
            elif self.model == "gpt-3.5-turbo-instruct":
                completion = openai.completions.create(
                    model=self.model,
                    seed=seed,
                    prompt = system_prompt + prompt
                )
                result = {
                    'question': system_prompt + prompt,
                    'model': str(completion.model),
                    'seed': seed,
                    'usage': {
                        'input_tokens': int(completion.usage.prompt_tokens),
                        'output_tokens': int(completion.usage.completion_tokens),
                    },
                    'answer': str(completion.choices[0].text),
                }
            
            
            if self.model == "gpt-4-1106-preview":
                self.gpt4_tokens += int(completion.usage.prompt_tokens) + int(completion.usage.completion_tokens)
            elif self.model == "gpt-3.5-turbo-1106":
                self.chatgpt_tokens += int(completion.usage.prompt_tokens) + int(completion.usage.completion_tokens)
            elif self.model == "gpt-3.5-turbo-instruct":
                self.chatgpt_instruct_tokens += int(completion.usage.prompt_tokens) + int(completion.usage.completion_tokens)
            print("Input tokens: ", completion.usage.prompt_tokens, "Output tokens: ", completion.usage.completion_tokens)
            print(f'OpenAI API time: {time.time() - t}')
            return result['answer']
        except Exception as e:
            print(e)
            return None

    def get_embedding(self, text, model="text-embedding-ada-002"):
        text = text.replace("\n", " ")
        try:
            response = openai.embeddings.create(input = [text], model=model)
            result = {
                'text': text,
                'model': model,
                'embedding': response.data[0].embedding,
                'usage': {
                    'input_tokens': int(response.usage.prompt_tokens),
                    'total_tokens': int(response.usage.total_tokens),
                }
            }
            return result
        except Exception as e:
            print(e)
            return None
        
class Zhipuai_api_handler:
    def __init__(self, model) -> None:
        # Put your own key in the llm_utils/glm_key.txt file
        with open('llm_utils/glm_key.txt', 'r') as f:
            zhipuai.api_key = f.readline().strip()
        if model == 'glm4':
            self.model = "glm-4"
        elif model == 'glm3_turbo':
            self.model = "glm-3-turbo"
        self.model_name = model

    def get_completion(self, system_prompt, prompt, seed=42):
        try:
            t = time.time()
            response = zhipuai.model_api.sse_invoke(
                model=self.model,
                prompt= system_prompt + prompt,
                temperature=0.15,
                top_p=0.7
            )

            result = ''
            for event in response.events():
                if event.event == "add":
                    result += event.data
            print(f'{self.model} API time: {time.time() - t}')
            return result
        except Exception as e:
            print(e)
            return None
        

class Gemini_api_handler:
    def __init__(self, model) -> None:
        # Put your own key in the llm_utils/gemini_key.txt file
        with open('llm_utils/gemini_key.txt', 'r') as f:
            genai.configure(api_key=f.readline().strip(), transport='rest')  
        if model == 'gemini_pro':
            self.model_name = "gemini_pro"
            self.model = genai.GenerativeModel('gemini-pro')

    def get_completion(self, system_prompt, prompt, seed=42):
        try:
            t = time.time()
            response = self.model.generate_content(system_prompt+prompt)
            result = response.text
            
            print(f'Gemini API time: {time.time() - t}')
            return result
        except Exception as e:
            print(e)
            return None