from datasets import load_dataset, Dataset
from trl import GRPOTrainer, GRPOConfig
from benchmarks.tabfact import TabFact
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

dotenv.load_dotenv()
login(os.environ.get("HF_TOKEN"))

wandb.init(
    project=os.environ.get("WANDB_PROJECT", "table-understanding"),
    entity=os.environ.get("WANDB_ENTITY"),
    name="GRPO-TabFact",
    reinit=True,
)

dataset = load_dataset("wenhu/tab_fact", "tab_fact", trust_remote_code=True, split="train")
dataset = dataset.shuffle(seed=42)
dataset = dataset.select(range(1000))

formatted_dataset = []
for example in dataset:
    statement = example.get("statement")
    label = example.get("label")
    table_text = example.get("table_text")
    table_caption = example.get("table_caption")

    prompt = TabFact().get_prompt(
        statement=statement,
        table_text=table_text,
        table_caption=table_caption,
        experiment='serialize_markdown',
    )

    formatted_dataset.append({
        "prompt": prompt,
        "ground_truth": label
    })

dataset = Dataset.from_list(formatted_dataset)

def tabfact_reward(completions, ground_truth, **kwargs):
    rewards = []
    for pred, ref in zip(completions, ground_truth):
        pred = pred.lstrip().lower()

        if ref == 0:
            ref = "a"
        elif ref == 1:
            ref = "b"

        if pred.startswith(ref):
            reward = 1.0
        else:
            reward = -1.0

        rewards.append(reward)

    return rewards

trainer_args = GRPOConfig(
    output_dir="snellius/out/grpo-tabfact",
    report_to=["wandb"],
    logging_strategy="steps",
    logging_steps=1,  
    save_strategy="epoch",
    run_name="grpo-tabfact-serialize-md",
    per_device_train_batch_size=16,
    num_generations=16,
    max_completion_length=2,
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
    reward_funcs=tabfact_reward,
    train_dataset=dataset,
    args = trainer_args,
    processing_class=tokenizer,
)
trainer.train()

wandb.finish()