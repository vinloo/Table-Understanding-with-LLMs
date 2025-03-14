from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import os
import torch.nn.functional as F

class Model:

    def __init__(self, model_name):

        model_map = {
            "llama3.1:8b": "meta-llama/Llama-3.1-8b",
            "llama3:70b": "meta-llama/Llama-3.1-70B",
            "jellyfish": "NECOUDBFM/Jellyfish-13B"
        }

        self.platform = os.environ.get("PLATFORM", "cluster")

        self.tokenizer = AutoTokenizer.from_pretrained(model_map[model_name], revision="main")
        self.tokenizer.padding_side = "left" 
        self.tokenizer.pad_token = self.tokenizer.eos_token 

        n_gpus = torch.cuda.device_count()
        print("NUMBER OF GPUs:", n_gpus)

        if self.platform == "cluster":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4"
            )
            self.LLM = AutoModelForCausalLM.from_pretrained(
                model_map[model_name], 
                trust_remote_code=True,
                device_map="auto",
                quantization_config=quantization_config,
            )
        else:
            self.LLM = AutoModelForCausalLM.from_pretrained(
                model_map[model_name], 
                trust_remote_code=True,
            )
            self.LLM.to("cuda")

        self.device = next(self.LLM.parameters()).device

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
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.LLM.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                **generate_kwargs
            )
        torch.cuda.empty_cache()
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

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
    
        n_options = len(options)
        choices = [f"{chr(65 + i)}" for i in range(n_options)]

        if isinstance(prompt, list):
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        else:
            inputs = self.tokenizer([prompt], return_tensors="pt") 

        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():    
            outputs = self.LLM(**inputs)
            del inputs
            logits = outputs.logits[:, -1, :].detach().cpu()
        
        probabilities = F.softmax(logits, dim=-1)
        token_ids = [self.tokenizer(f" {choice}", add_special_tokens=False).input_ids[0] for choice in choices]


        preds = []
        for probs in probabilities: 
            option_probs = [probs[token_id].item() for token_id in token_ids]
            preds.append(int(torch.argmax(torch.tensor(option_probs)).item()))

        torch.cuda.empty_cache()

        return preds
                
