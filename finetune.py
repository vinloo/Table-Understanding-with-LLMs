from datasets import load_dataset, Dataset
from trl import GRPOTrainer, GRPOConfig
from benchmarks.tablebench import TableBench
import re
import dotenv
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

def tablebench_reward(completions, ground_truth, **kwargs):
    rewards = []
    for pred, ref in zip(completions, ground_truth):
        match = re.search(r"(.+)", pred)
        if match:
            pred = match.group(1)
        else:
            pred = ''

        rouge = evaluate.load("rouge")
        rouge_score = rouge.compute(predictions=[pred], references=[ref])['rougeL']
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
    num_generations=2
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8b", revision="main")
tokenizer.padding_side = "left" 
tokenizer.pad_token = tokenizer.eos_token 

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8b", 
    trust_remote_code=True,
    device_map="auto",
    quantization_config=quantization_config,
)

lora_config = LoraConfig(
    target_modules=["q_proj", "k_proj"],
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
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