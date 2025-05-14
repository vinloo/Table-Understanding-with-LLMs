from datasets import load_dataset, Dataset
from trl import GRPOTrainer, GRPOConfig
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
import json
import pandas as pd

dotenv.load_dotenv()
login(os.environ.get("HF_TOKEN"))

wandb.init(
    project=os.environ.get("WANDB_PROJECT", "table-understanding"),
    entity=os.environ.get("WANDB_ENTITY"),
    name="GRPO-TableBench-deepseek-qwen32b",
    reinit=True,
)

dataset = load_dataset("Multilingual-Multimodal-NLP/TableBench", revision="90593ad8af90f027f6f478b8c4c1981d9f073a83", split="test")
dataset = dataset.filter(lambda x: x['instruction_type'] == 'DP')
dataset = dataset.shuffle(seed=42)

def get_prompt(table, question):
    table_json = json.loads(table)
    table = pd.DataFrame(table_json['data'], columns=table_json['columns'])
    serialized_table = table.to_markdown(index=False)

    prompt = ("You are a table analyst. Your task is to answer questions based on the table content.\n"
        f"Read the table below in Markdown format:\n"

        "[TABLE]\n"
        f"{serialized_table}\n\n"
        "First write up to 120 tokens of step-by-step reasoning.\n"
        "When you are completely finished, write:\n"
        "===== FINAL ANSWER START =====\n"
        "<your concise answer here>\n"
        "===== FINAL ANSWER END =====\n\n"
        f"Question: {question}")
    
    return prompt

formatted_dataset = []
for example in dataset:
    label = example.get("answer")
    question = example.get("question")
    table = example.get("table")
    prompt = get_prompt(
        table=table,
        question=question,
    )

    formatted_dataset.append({
        "prompt": prompt,
        "ground_truth": label
    })

dataset = Dataset.from_list(formatted_dataset)

rouge = evaluate.load("rouge")

def normalize_answer(s: str) -> str:
    """Lowercase, remove articles/punctuation, collapse whitespace."""
    import re
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = re.sub(r'[^a-z0-9]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def tablebench_reward(completions, ground_truth, **kwargs):
    rewards = []
    for pred, ref in zip(completions, ground_truth):
        # Strict format check
        match = re.search(
            r"===== FINAL ANSWER START =====\s*(.*?)\s*===== FINAL ANSWER END =====",
            pred, flags=re.S
        )
        if match:
            answer_text = match.group(1).strip()
            used_fallback = False
        else:
            # last non-empty line as fallback
            lines = [L for L in pred.splitlines() if L.strip()]
            answer_text = lines[-1].strip() if lines else ""
            used_fallback = True

        norm_pred = normalize_answer(answer_text)
        norm_ref  = normalize_answer(ref)

        # exact mat h
        if norm_pred == norm_ref and not used_fallback:
            reward = 1.0
        else:
            # rouge score for partial match
            rouge_score = rouge.compute(
                predictions=[norm_pred],
                references=[norm_ref]
            )["rougeL"]

            # penalty by scaling if used fallback
            max_base = 0.9 if not used_fallback else 0.6
            reward = max_base * rouge_score

        # penalty for using fallback
        if used_fallback:
            reward -= 0.3

        # penalty for too long reasoning
        reasoning_tokens = len(pred.split("===== FINAL ANSWER START =====")[0].split())
        over = max(reasoning_tokens - 100, 0)
        reward -= 0.0005 * over  # light nudge

        # clip reward
        reward = max(-1.0, min(1.0, reward))
        rewards.append(reward)

    return rewards

trainer_args = GRPOConfig(
    output_dir="snellius/out/grpo-tablebench-deepseek-qwen32b",
    report_to=["wandb"],
    logging_strategy="steps",
    logging_steps=1,  
    save_strategy="epoch",
    run_name="grpo-tablebench-deepseek-qwen32b",
    per_device_train_batch_size=8,
    num_generations=8,
    max_completion_length=150,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    beta=0.05,
    gradient_accumulation_steps=4,
    # gradient_checkpointing=True,
)

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", revision="main")
tokenizer.padding_side = "left" 
tokenizer.pad_token = tokenizer.eos_token 

model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
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