from datasets import load_dataset, Split
import evaluate
import re
from tqdm import tqdm

class TableBench:
    def run(self, model):
        metric_names = ["rouge"]
        subtasks = ["FactChecking", "NumericalReasoning", "DataAnalysis"] # we do not include Visualization as this does not fit the research

        ds = load_dataset("Multilingual-Multimodal-NLP/TableBench")
        ds = ds.filter(lambda x: x['instruction_type'] == 'DP')

        split = Split.TEST
        if Split.TEST not in ds:
            split = Split.VALIDATION if Split.VALIDATION in ds else Split.TRAIN

        metrics = {name: evaluate.load(name) for name in metric_names}


        predictions = {
            "FactChecking": [],
            "NumericalReasoning": [],
            "DataAnalysis": [],
            "Visualization": []
        }
        references = {
            "FactChecking": [],
            "NumericalReasoning": [],
            "DataAnalysis": [],
            "Visualization": []
        }

        for task in subtasks:
            ds_task = ds.filter(lambda x: x['qtype'] == task)
            for example in tqdm(ds_task[split], total=len(ds_task[split])):
                label = example.get("answer")
                question = example.get("question")
                prompt = example.get("instruction")
                pred = model.generate(prompt, max_new_tokens=50).split(question)[-1]
                
                try:
                    match = re.search(r"Final Answer: (.+)", pred)
                    if match:
                        pred = match.group(1)
                    else:
                        pred = ''
                except Exception as e:
                    pred = ''

                predictions[task].append(pred)
                references[task].append(label)

        results = {}
        for name, metric in metrics.items():
            for task in subtasks:
                results[f"rougeL/{task}"] = metric.compute(predictions=predictions[task], references=references[task])['rougeL']

        return results