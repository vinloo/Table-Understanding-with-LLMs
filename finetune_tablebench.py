from datasets import load_dataset, Dataset
from trl import GRPOTrainer, GRPOConfig
from benchmarks.tablebench import TableBench
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
    name="GRPO-TableBench",
    reinit=True,
)

dataset = load_dataset("Multilingual-Multimodal-NLP/TableBench", revision="90593ad8af90f027f6f478b8c4c1981d9f073a83", split="test")
dataset = dataset.filter(lambda x: x['instruction_type'] == 'DP')
dataset = dataset.shuffle(seed=42)

formatted_dataset = []
for example in dataset:
    label = example.get("answer")
    question = example.get("question")
    table = example.get("table")
    formatter = example.get("answer_formatter")
    instruction = example.get("instruction")

    prompt = TableBench().get_prompt(
        table=table,
        question=question,
        formatter=formatter,
        experiment='serialize_markdown',
    )

    formatted_dataset.append({
        "prompt": prompt,
        "ground_truth": label
    })

dataset = Dataset.from_list(formatted_dataset)

def normalize_answer(s):
    def remove_punctuation(text):
        return text.translate(str.maketrans('', '', string.punctuation))
    
    def lower(text):
        return text.lower()
    
    def white_space_fix(text):
        return " ".join(text.split())

    return white_space_fix(remove_punctuation(lower(s)))

def tablebench_reward(completions, ground_truth, **kwargs):
    rewards = []
    rouge = evaluate.load("rouge")
    for pred, ref in zip(completions, ground_truth):
        match = re.search(r"(.+)", pred)
        pred = match.group(1) if match else ''
        
        norm_pred = normalize_answer(pred)
        norm_ref = normalize_answer(ref)
        
        if norm_pred == norm_ref:
            rewards.append(1.0)
            continue

        rouge_score = rouge.compute(predictions=[pred], references=[ref])["rougeL"]
        rewards.append(rouge_score)

    return rewards

trainer_args = GRPOConfig(
    output_dir="snellius/out/grpo-tablebench",
    report_to=["wandb"],
    logging_strategy="steps",
    logging_steps=1,  
    save_strategy="epoch",
    run_name="grpo-tablebench-serialize-md",
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
    reward_funcs=tablebench_reward,
    train_dataset=dataset,
    args = trainer_args,
    processing_class=tokenizer,
)
trainer.train()

wandb.finish()