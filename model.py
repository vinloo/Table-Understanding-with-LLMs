from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import os
import torch.nn.functional as F
import pandas as pd
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.llama.modeling_llama import LlamaAttention
from torch import nn
from typing import Optional, Tuple, Unpack
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs

class Model:

    def __init__(self, model_name):

        model_map = {
            "llama3.1:8b": "meta-llama/Llama-3.1-8b",
            "llama3:70b": "meta-llama/Llama-3.1-70B",
            "jellyfish": "NECOUDBFM/Jellyfish-13B"
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
                


    def generate_with_tabular_attention(self, prompt_pre, prompt_table, prompt_post, separator='\u0488', max_new_tokens=50, **generate_kwargs):
        if isinstance(prompt_table, pd.DataFrame):
            prompt_table = prompt_table.to_csv(index=False, sep=separator)

        tokens_pre = [self.tokenizer.bos_token] + self.tokenizer.tokenize(prompt_pre, return_tensors="pt")
        tokens_table = self.tokenizer.tokenize(prompt_table, return_tensors="pt")
        tokens_post = self.tokenizer.tokenize(prompt_post, return_tensors="pt")
        tokens = tokens_pre + tokens_table + tokens_post

        text_ids = list(range(len(tokens_pre))) + list(range(len(tokens_pre) + len(tokens_table), len(tokens_pre) + len(tokens_table) + len(tokens_post)))
        table_ids = list(range(len(tokens_pre), len(tokens_pre) + len(tokens_table)))

        curr_row, curr_col = 0, 0
        newline_tokens = self.tokenizer.tokenize("\n", return_tensors="pt")
        separator_tokens = self.tokenizer.tokenize(separator, return_tensors="pt")

        content_ids = set()
        rows_ids = []
        cols_ids = []
        just_added_col = False

        for token_id in table_ids:
            if  tokens[token_id] in newline_tokens:
                curr_row += 1
                curr_col = 0

            if len(rows_ids) <= curr_row:
                rows_ids.append([])
            if len(cols_ids) <= curr_col:
                cols_ids.append([])
            
            if  tokens[token_id] in separator_tokens and not just_added_col:
                curr_col += 1
                just_added_col = True
            else:
                just_added_col = False

            if tokens[token_id] not in separator_tokens and tokens[token_id] not in newline_tokens:
                if len(rows_ids[curr_row]) <= curr_col:
                    rows_ids[curr_row].append([])
                if len(cols_ids[curr_col]) <= curr_row:
                    cols_ids[curr_col].append([])

                rows_ids[curr_row][curr_col].append(token_id)
                cols_ids[curr_col][curr_row].append(token_id)
                content_ids.add(token_id)
        
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

        # construct the attention score mask
        atten_score_mask = torch.zeros((1, self.LLM.config.num_attention_heads, len(tokens), len(tokens)), dtype=torch.float32, device=self.device)
        for i_idx, i in enumerate(table_ids):
                for j in table_ids[i_idx + 1:]:
                    if (i,j) not in attention_pairs and (j,i) not in attention_pairs:
                        atten_score_mask[:, :, i, j] = 1
                        atten_score_mask[:, :, j, i] = 1


        # replace the attention layers with the custom tabular attention
        self.LLM.config._attn_implementation = "eager"

        for i, layer in enumerate(self.LLM.model.layers):
            original_attn = layer.self_attn
            tabular_attn = TabularLlamaAttention(self.LLM.config, i, atten_score_mask)
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
            attn_weights[self.attention_score_mask == 1] = 1e-9


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
        **kwargs: Unpack[FlashAttentionKwargs],
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