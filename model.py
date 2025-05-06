from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import os
import torch.nn.functional as F
import pandas as pd
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.llama.modeling_llama import LlamaAttention
from torch import nn
from typing import Optional, Tuple
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from peft import PeftModel, PeftConfig

class Model:

    def __init__(self, model_name, LoRA_path=None):

        model_map = {
            "llama3.1:8b": "meta-llama/Llama-3.1-8b",
            "llama3:70b": "meta-llama/Llama-3.1-70B",
            "jellyfish": "NECOUDBFM/Jellyfish-13B",
            "deepseek-llama8b": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        }

        self.platform = os.environ.get("PLATFORM", "cluster")
        self.model_type = model_name
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

        if LoRA_path is not None:
            self.LLM = PeftModel.from_pretrained(self.LLM, LoRA_path)
            print("LoRA loaded from:", LoRA_path, flush=True)

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

        if self.model_type.lower() == 'jellyfish':
            token_ids = [self.tokenizer(choice, add_special_tokens=False).input_ids[0] for choice in choices]
        else:
            token_ids = [self.tokenizer(f" {choice}", add_special_tokens=False).input_ids[0] for choice in choices]


        preds = []
        for probs in probabilities: 
            option_probs = [probs[token_id].item() for token_id in token_ids]
            preds.append(int(torch.argmax(torch.tensor(option_probs)).item()))

        torch.cuda.empty_cache()

        return preds
                


    def generate_with_tabular_attention(self, prompt_pre, table, prompt_post, separator=',', max_new_tokens=50, **generate_kwargs):
        if not prompt_pre.startswith(self.tokenizer.bos_token):
            prompt_pre = self.tokenizer.bos_token + prompt_pre

        tokens_pre = self.tokenizer.tokenize(prompt_pre, return_tensors="pt")

        content_ids = set()

        rows_ids = [[] for _ in range(table.shape[0] + 1)]
        cols_ids = [[] for _ in range(table.shape[1])]
        tokens_table = []

        newline_tokens = self.tokenizer.tokenize("\n", return_tensors="pt")
        separator_tokens = self.tokenizer.tokenize(separator, return_tensors="pt")
        newline_n_tokens = len(newline_tokens)
        separator_n_tokens = len(separator_tokens)

        id_count = len(tokens_pre)

        for col_idx, header in enumerate(table.columns):
            cell_tokens = self.tokenizer.tokenize(str(header))
            
            cell_start = id_count
            cell_idxs = list(range(cell_start, cell_start + len(cell_tokens)))
            
            tokens_table.extend(cell_tokens)
            content_ids.update(cell_idxs)
            
            rows_ids[0].append(cell_idxs)
            cols_ids[col_idx].append(cell_idxs)
            
            id_count += len(cell_tokens)
            
            if col_idx < len(table.columns) - 1:
                tokens_table.extend(separator_tokens)
                id_count += separator_n_tokens

        tokens_table.extend(newline_tokens)
        id_count += newline_n_tokens

        for row_idx, (_, row) in enumerate(table.iterrows()):
            for col_idx, cell in enumerate(row):
                cell_tokens = self.tokenizer.tokenize(str(cell))
                
                cell_start = id_count
                cell_idxs = list(range(cell_start, cell_start + len(cell_tokens)))
                
                tokens_table.extend(cell_tokens)
                content_ids.update(cell_idxs)
                
                rows_ids[row_idx + 1].append(cell_idxs)
                cols_ids[col_idx].append(cell_idxs)
                
                id_count += len(cell_tokens)
                
                if col_idx < len(row) - 1:
                    tokens_table.extend(separator_tokens)
                    id_count += separator_n_tokens
            
            tokens_table.extend(newline_tokens)
            id_count += newline_n_tokens

        if newline_n_tokens > 0:
            tokens_table = tokens_table[:-newline_n_tokens]
            id_count -= newline_n_tokens

        tokens_post = self.tokenizer.tokenize(prompt_post, return_tensors="pt")
        tokens = tokens_pre + tokens_table + tokens_post

        text_ids = list(range(len(tokens_pre))) + list(
            range(len(tokens_pre) + len(tokens_table), len(tokens_pre) + len(tokens_table) + len(tokens_post))
        )
        table_ids = list(range(len(tokens_pre), len(tokens_pre) + len(tokens_table)))
        attention_pairs = set()

        for row in rows_ids:
            for i, cell in enumerate(row):
                for ti, token_id in enumerate(cell):

                    for token_id2 in cell[ti + 1:]:
                        attention_pairs.add((token_id, token_id2))

                    for cell2 in row[i + 1:]:
                        for token_id2 in cell2:
                            attention_pairs.add((token_id, token_id2))

        for col in cols_ids:
            for i, cell in enumerate(col):
                for token_id in cell:
                    for cell2 in col[i + 1:]:
                        for token_id2 in cell2:
                            attention_pairs.add((token_id, token_id2))

        layer_head_pairs = [(31, 14), (0, 24), (0, 3), (0, 2), (0, 26), (0, 0), (0, 25), (0, 6), (0, 10), (1, 0), (0, 27), (0, 1), (0, 4), (0, 7), (1, 21), (1, 10), (1, 11), (0, 5), (1, 26), (1, 31), (1, 29), (1, 20), (1, 24), (1, 28), (0, 16), (1, 25), (1, 17), (1, 6), (1, 5), (0, 11), (1, 13), (1, 3), (1, 30), (0, 12), (0, 17), (0, 9), (1, 8), (1, 16), (1, 27), (0, 13), (1, 18), (1, 4), (0, 30), (0, 8), (0, 20), (0, 21), (0, 14), (1, 1), (0, 18), (1, 19), (0, 23), (1, 15), (1, 23), (1, 12), (0, 22), (1, 14), (0, 19), (1, 22), (0, 28), (0, 15), (12, 12), (1, 7), (15, 6), (16, 30), (15, 5), (1, 9), (31, 23), (13, 28), (30, 25), (12, 15), (14, 13), (14, 26), (15, 24), (11, 26), (13, 3), (31, 25), (12, 3), (7, 27), (10, 19), (15, 10), (11, 24), (1, 2), (13, 2), (15, 8), (7, 8), (14, 8), (8, 23), (30, 18), (9, 2), (13, 22), (31, 12), (31, 1), (16, 22), (8, 31), (14, 14), (11, 14), (6, 20), (5, 17), (12, 30), (9, 10)]

        atten_score_mask = torch.zeros((1, self.LLM.config.num_attention_heads, len(tokens), len(tokens)), dtype=torch.float32, device=self.device)
        for i_idx, i in enumerate(table_ids):
            for j in table_ids[i_idx + 1:]:
                if (i,j) not in attention_pairs and (j,i) not in attention_pairs and i != j:
                    atten_score_mask[:, :, i, j] = 1
                    atten_score_mask[:, :, j, i] = 1

        for layer_idx, head_idx in layer_head_pairs:
            if layer_idx > 20:
                continue

            layer = self.LLM.model.layers[layer_idx]
            original_attn = layer.self_attn
            
            layer_head_attention_score_mask = atten_score_mask.clone()
            mask = torch.ones(layer_head_attention_score_mask.shape[1], dtype=torch.bool)
            mask[head_idx] = False
            layer_head_attention_score_mask[:, mask, :, :] = 0

            tabular_attn = TabularLlamaAttention(self.LLM.config, layer_idx, layer_head_attention_score_mask.clone())
            tabular_attn.load_state_dict(original_attn.state_dict())
            layer.self_attn = tabular_attn

        self.LLM.to(self.device)

        # generate outputs
        inputs = torch.tensor([self.tokenizer.convert_tokens_to_ids(tokens)], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(inputs, device=self.device)
        

        inputs = inputs.contiguous()
        attention_mask = attention_mask.contiguous()

        self.LLM.eval()
        with torch.no_grad():
            result = self.LLM.generate(input_ids=inputs, attention_mask=attention_mask,
                pad_token_id=self.tokenizer.pad_token_id, **generate_kwargs)
        torch.cuda.empty_cache()

        output = self.tokenizer.decode(result[0], skip_special_tokens=True,)

        return output



class TabularLlamaAttention(LlamaAttention):

    def __init__(self, config: LlamaConfig, layer_idx: int, attention_score_mask):
        super().__init__(config, layer_idx)
        self.attention_score_mask = attention_score_mask


    def eager_attention_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float = 0.0,
        **kwargs,
    ):
        key_states = repeat_kv(key, self.num_key_value_groups)
        value_states = repeat_kv(value, self.num_key_value_groups)

        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        if attn_weights.shape == self.attention_score_mask.shape:
            attn_weights[self.attention_score_mask == 1] = -1e9


        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        return attn_output, attn_weights


    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attn_output, attn_weights = self.eager_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights