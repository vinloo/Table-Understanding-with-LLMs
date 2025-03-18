from datasets import load_dataset, Split
import evaluate
import re
from tqdm import tqdm
import textwrap
import pandas as pd
import json

class TableBench:

    def get_prompt(self, table, question, formatter, experiment):

        if experiment == "serialize_json":
            serialized_table = table
            format = "JSON"
        else:
            table_json = json.loads(table)
            table = pd.DataFrame(table_json['data'], columns=table_json['columns'])
            if experiment == "serialize_csv":
                serialized_table = table.to_csv(index=False)
                format = "CSV"
            elif experiment == "serialize_markdown":
                serialized_table = table.to_markdown(index=False)
                format = "Markdown"

        prompt = ("You are a table analyst. Your task is to answer questions based on the table content.\n"
        "\n\n"
        f"{formatter}"
        "\n\n"
        "Give the final answer to the question directly without any explanation.\n"
        "\n"
        f"Read the table below in {format} format:\n"
        "[TABLE]\n"
        f"{serialized_table}\n"
        "\n"
        "Let's get start!\n"
        f"Question: {question}\n\n"
        "Final Answer: ")
        
        return prompt

    def run(self, model, experiment, batch_size):
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
                table = example.get("table")
                formatter = example.get("answer_formatter")
                if experiment == "baseline":
                    prompt = example.get("instruction")
                else:
                    prompt = self.get_prompt(table, question, formatter, experiment)
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