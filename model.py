from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
from accelerate import Accelerator

class Model:

    def __init__(self, model_name, device=None):

        model_map = {
            "llama3.1:8b": "meta-llama/Llama-3.1-8b",
            "llama3:70b": "meta-llama/Llama-3.1-70B"
        }

        self.tokenizer = AutoTokenizer.from_pretrained(model_map[model_name])
        self.LLM = AutoModelForCausalLM.from_pretrained(
            model_map[model_name], 
            trust_remote_code=True,
        )
        self.accelerator = Accelerator()
        self.LLM = self.accelerator.prepare(self.LLM)
        

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
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self.accelerator.device) for key, value in inputs.items()}
        
        self.LLM.eval()
        with torch.no_grad():
            outputs = self.LLM.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                **generate_kwargs
            ).cpu() 
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def predict(self, prompt, options):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self.accelerator.device) for key, value in inputs.items()}

        n_options = len(options)
        choices = [f"{chr(65 + i)}" for i in range(n_options)]
        
        self.LLM.eval()
        with torch.no_grad():
            outputs = self.LLM(**inputs)
            logits = outputs.logits[:, -1, :].cpu()
        
        probabilities = F.softmax(logits, dim=-1)

        token_ids = self.tokenizer(choices, add_special_tokens=False).input_ids
        token_ids = [token[0] for token in token_ids]
        
        token_probs = {choice: probabilities[0, token_id].item() for choice, token_id in zip(choices, token_ids)}
        pred = max(token_probs, key=token_probs.get)
        pred = choices.index(pred)

        return pred

