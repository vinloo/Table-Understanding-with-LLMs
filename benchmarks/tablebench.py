from datasets import load_dataset, Split
import evaluate
import re
from tqdm import tqdm

class TableBench:
    def run(self, model, batch_size=1):
        ds = load_dataset("Multilingual-Multimodal-NLP/TableBench")
        ds = ds.filter(lambda x: x['instruction_type'] == 'DP')
        split = Split.TEST
        if Split.TEST not in ds:
            split = Split.VALIDATION if Split.VALIDATION in ds else Split.TRAIN
        metrics = {name: evaluate.load(name) for name in ["rouge"]}
        predictions = {task: [] for task in ["FactChecking", "NumericalReasoning", "DataAnalysis"]}
        references = {task: [] for task in ["FactChecking", "NumericalReasoning", "DataAnalysis"]}
        for task in ["FactChecking", "NumericalReasoning", "DataAnalysis"]:
            ds_task = ds.filter(lambda x: x['qtype'] == task)
            prompts = []
            examples = []
            for example in tqdm(ds_task[split], desc=f"Processing {task} examples"):
                label = example.get("answer")
                question = example.get("question")
                prompt = example.get("instruction")
                prompts.append(prompt)
                examples.append((question, label))
                if len(prompts) == batch_size:
                    batch_preds = model.generate(prompts, max_new_tokens=50)
                    for pred, (question, label) in zip(batch_preds, examples):
                        pred = pred.split(question)[-1]
                        match = re.search(r"Final Answer: (.+)", pred)
                        if match:
                            pred = match.group(1)
                        else:
                            pred = ''
                        predictions[task].append(pred)
                        references[task].append(label)
                    prompts = []
                    examples = []
            if prompts:
                batch_preds = model.generate(prompts, max_new_tokens=50)
                for pred, (question, label) in zip(batch_preds, examples):
                    pred = pred.split(question)[-1]
                    match = re.search(r"Final Answer: (.+)", pred)
                    if match:
                        pred = match.group(1)
                    else:
                        pred = ''
                    predictions[task].append(pred)
                    references[task].append(label)
        results = {}
        for name, metric in metrics.items():
            for task in ["FactChecking", "NumericalReasoning", "DataAnalysis"]:
                results[f"rougeL/{task}"] = metric.compute(predictions=predictions[task], references=references[task])['rougeL']
        return results