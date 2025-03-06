from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import os
import torch.nn.functional as F

class Model:

    def __init__(self, model_name):

        model_map = {
            "llama3.1:8b": "meta-llama/Llama-3.1-8b",
            "llama3:70b": "meta-llama/Llama-3.1-70B"
        }

        self.platform = os.environ.get("PLATFORM", "cluster")

        self.tokenizer = AutoTokenizer.from_pretrained(model_map[model_name])
        self.tokenizer.padding_side = "left" 
        self.tokenizer.pad_token = self.tokenizer.eos_token 

        print("NUMBER OF GPUs:", torch.cuda.device_count())

        if self.platform == "cluster":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0, 
            )
            self.LLM = AutoModelForCausalLM.from_pretrained(
                model_map[model_name], 
                trust_remote_code=True,
                device_map="balanced",
                quantization_config=quantization_config,
                max_memory={i: "40GiB" for i in range(torch.cuda.device_count())}
            )
        else:
            self.LLM = AutoModelForCausalLM.from_pretrained(
                model_map[model_name], 
                trust_remote_code=True,
            )
            self.LLM.to("cuda")

    def generate(self, prompt, max_new_tokens=50, **generate_kwargs):
        """
        Generate text based on a prompt.
        
        Parameters:
          prompt (str): The input text to generate from.
          max_new_tokens (int): The maximum number of new tokens to generate.
          generate_kwargs: Additional keyword arguments for the model's generate() method.
          
        Returns:
          str: The generated text output.
        """
        self.LLM.eval()
        with torch.no_grad():
            if isinstance(prompt, list) and len(prompt) > 1:
                inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
                if self.platform == "local":
                    inputs = inputs.to("cuda")

                outputs = self.LLM.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    **generate_kwargs
                )
                return [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            else:
                if isinstance(prompt, list):
                    prompt = prompt[0]
                inputs = self.tokenizer(prompt, return_tensors="pt")
                if self.platform == "local":
                    inputs = inputs.to("cuda")
                outputs = self.LLM.generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    **generate_kwargs
                )
                return [self.tokenizer.decode(outputs[0], skip_special_tokens=True)]

    def predict(self, prompt, options):
        """
        Predict the most probable option based on the prompt.
        
        Parameters:
          prompt (str): The input text to base the prediction on.
          options (list): A list of option strings to choose from.
          
        Returns:
          int: The index of the most probable option.
        """
        self.LLM.eval()
        with torch.no_grad():
            if isinstance(prompt, list) and len(prompt) > 1:
                inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
                if self.platform == "local":
                    inputs = inputs.to("cuda")
                outputs = self.LLM(**inputs)
                logits = outputs.logits[:, -1, :]
                probabilities = F.softmax(logits, dim=-1)
                token_ids = [self.tokenizer(option, add_special_tokens=False).input_ids[0] for option in options]
                results = []
                for probs in probabilities:
                    option_probs = [probs[token_id].item() for token_id in token_ids]
                    results.append(int(torch.argmax(torch.tensor(option_probs))))
                return results
            else:
                if isinstance(prompt, list):
                    prompt = prompt[0]
                inputs = self.tokenizer(prompt, return_tensors="pt")
                if self.platform == "local":
                    inputs = inputs.to("cuda")
                outputs = self.LLM(**inputs)
                logits = outputs.logits[:, -1, :]
                probabilities = F.softmax(logits, dim=-1)
                token_ids = [self.tokenizer(option, add_special_tokens=False).input_ids[0] for option in options]
                option_probs = [probabilities[0, token_id].item() for token_id in token_ids]
                return [int(torch.argmax(torch.tensor(option_probs)))]
