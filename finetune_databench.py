from datasets import load_dataset, Dataset
from trl import GRPOTrainer, GRPOConfig
from benchmarks.databench import DataBench
import re
import dotenv
import string
import os
from huggingface_hub import login
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from peft import LoraConfig, get_peft_model
import evaluate
import pandas as pd
import hashlib
import json
from json.decoder import JSONDecodeError

dotenv.load_dotenv()
login(os.environ.get("HF_TOKEN"))

wandb.init(
    project=os.environ.get("WANDB_PROJECT", "table-understanding"),
    entity=os.environ.get("WANDB_ENTITY"),
    name="GRPO-DataBench",
    reinit=True,
)

dataset = load_dataset("cardiffnlp/databench", "semeval", split="train")
dataset = dataset.shuffle(seed=42)

def read_parquet_with_cache( file_path):
    """Reads a parquet file with caching."""
    dotenv.load_dotenv()
    cache_dir = os.getenv("PARQUET_CACHE_DIR", "cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Generate a unique hash for the file path
    file_hash = hashlib.md5(file_path.encode()).hexdigest()
    cached_file_path = os.path.join(cache_dir, f"{file_hash}.parquet")

    if os.path.exists(cached_file_path):
        return pd.read_parquet(cached_file_path)
    else:
        df = pd.read_parquet(file_path)
        df.to_parquet(cached_file_path)
        return df

formatted_dataset = []
for example in dataset:
    label = example.get("sample_answer")
    question = example.get("question")
    table_id = example['dataset']
    table = read_parquet_with_cache(f"hf://datasets/cardiffnlp/databench/data/{table_id}/sample.parquet")
    task = example.get("type")

    prompt = DataBench().get_prompt(
        table=table,
        question=question,
        experiment='serialize_markdown',
    )

    formatted_dataset.append({
        "prompt": prompt,
        "ground_truth": label,
        "task": task,
    })

dataset = Dataset.from_list(formatted_dataset)

def parse_pred(pred):
    # pred = pred.split("ASSISTANT: ")[1].strip()
    pred = '{"answer":' + pred

    pred = pred.replace("{{", "{")
    pred = pred.replace("}}", "}")

    try:
        # try to get from the first opeining bracket to the last closing bracket
        pred = json.loads(pred[pred.find("{"):pred.rfind("}")+1])
        pred = pred["answer"]
    except JSONDecodeError:
        # if the json is not well formatted, try to get the first object between : and , OR : and }
        pred = pred.split('answer":')[1].strip()
        try:
            pred = pred[pred.find("{")+1:pred.find(",")].strip()
        except ValueError:
            try:
                pred = pred[pred.find("{")+1:pred.find("}")].strip()
            except ValueError:
                print("Failed to parse answer")

    return str(pred)


def normalize_answer(s):
    def remove_punctuation(text):
        return text.translate(str.maketrans('', '', string.punctuation))
    
    def lower(text):
        return text.lower()
    
    def white_space_fix(text):
        return " ".join(text.split())

    return white_space_fix(remove_punctuation(lower(s)))


def databench_reward(completions, ground_truth, task, **kwargs):
    rewards = []
    rouge = evaluate.load("rouge")
    for pred, ref, tsk in zip(completions, ground_truth, task):
        parsed_pred = parse_pred(pred)
        
        norm_pred = normalize_answer(parsed_pred)
        norm_ref = normalize_answer(str(ref))
        
        if norm_pred == norm_ref:
            rewards.append(1.0)
            continue

        rouge_score = rouge.compute(predictions=[norm_pred], references=[norm_ref])["rougeL"]
        rewards.append(rouge_score)

    return rewards

trainer_args = GRPOConfig(
    output_dir="snellius/out/grpo-databench",
    report_to=["wandb"],
    logging_strategy="steps",
    logging_steps=1,  
    save_strategy="epoch",
    run_name="grpo-databench-serialize-md",
    per_device_train_batch_size=16,
    num_generations=16,
    max_completion_length=50,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    beta=0.05,
    gradient_accumulation_steps=4,
    # gradient_checkpointing=True,
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8b", revision="main")
tokenizer.padding_side = "left" 
tokenizer.pad_token = tokenizer.eos_token 

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8b", 
    trust_remote_code=True,
    device_map="auto",
)

lora_config = LoraConfig(
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
    r=64,
    lora_alpha=64,
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(
    model,
    lora_config,
)

torch.autograd.set_detect_anomaly(True)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=databench_reward,
    train_dataset=dataset,
    args = trainer_args,
    processing_class=tokenizer,
)
trainer.train()

wandb.finish()