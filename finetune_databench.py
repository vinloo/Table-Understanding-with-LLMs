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

def parse_pred_label(pred, label, task):
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

    if task == "boolean":
        if pred == 0:
            pred = False
        elif pred == 1:
            pred = True

        if label.lower() == "true":
            label = True
        elif label.lower() == "false":
            label = False

        if isinstance(pred, str):
            pred = pred.split(",")[0].strip().lower()
            if "true" in pred or "yes" in pred or "right" in pred or "1" in pred:
                pred = True
            elif "false" in pred or "no" in pred or "wrong" in pred or "0" in pred:
                pred = False
            else:
                pred = not label
                print("Failed to parse boolean")

    elif task == "number":
        if isinstance(pred, str):
            pred = pred.split(",")[0].strip().lower().replace('"', '')
            try:
                pred = float(pred)
            except ValueError:
                print("Failed to parse number")
        if label == "null" or label == "":
            label = None
        if label is not None:
            label = float(label)

    elif task == "category":
        pred = str(pred)
        pred = pred.split(",")[0].strip().lower()
        pred = ''.join(e for e in pred if e.isalnum())
        label = str(label)
        label = ''.join(e for e in label if e.isalnum()).lower()

    elif task == "list[number]":
        if isinstance(pred, str):
            pred = ''.join(e for e in pred if e.isalnum() or e in [",", "."])
            pred = pred.split(",")
            if not isinstance(pred, list):
                pred = [pred]
            pred = [x.strip().lower() for x in pred]
            pred_float = []
            for x in pred:
                try:
                    pred_float.append(float(x))
                except ValueError:
                    print("Failed to parse list of numbers")
            pred = pred_float
        if isinstance(label, str):
            if label not in ["", "null", "[]", "['']"]:
                label = ''.join(e for e in label if e.isalnum() or e in [",", "."])
                label = label.split(",")
                label = [x.strip().lower() for x in label]
                label = [float(x) for x in label]

    elif task == "list[category]":
        pred = str(pred)
        pred = ''.join(e for e in pred if e.isalnum() or e == ",")
        pred = pred.split(",")
        if not isinstance(pred, list):
            pred = [pred]
        pred = [x.strip().lower() for x in pred]
        if label not in ["", None, "null", "[]", [], "['']"]:
            label = ''.join(e for e in label if e.isalnum() or e == ",")
            label = label.split(",")
            label = [x.strip().lower() for x in label]

    return pred, label


def list_to_set(lst):
    result = set()
    for item in lst:
        try:
            result.add(item)
        except TypeError:
            continue  # Skip unhashable items
    return result


def databench_reward(completions, ground_truth, task, **kwargs):
    rewards = []
    for pred, ref, tsk in zip(completions, ground_truth, task):
        parsed_pred, parsed_ref = parse_pred_label(pred, ref, task=tsk)
        
        if isinstance(parsed_pred, list):
            if not isinstance(parsed_ref, list):
                parsed_ref = [parsed_ref]
            overlap = list_to_set(parsed_pred).intersection(list_to_set(parsed_ref))
            reward = len(overlap) / max(len(parsed_pred), len(parsed_ref))
            reward = (reward - 0.5) * 2 # scale to [-1, 1]
            rewards.append(reward)
        else:
            if parsed_pred == parsed_ref:
                rewards.append(1.0)
            else:
                rewards.append(-1.0)

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