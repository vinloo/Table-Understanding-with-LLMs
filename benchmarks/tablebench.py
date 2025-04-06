from datasets import load_dataset, Split, Dataset
import evaluate
import re
from tqdm import tqdm
import textwrap
import pandas as pd
import json

class TableBench:

    def get_prompt(self, table, question, formatter, experiment, shots=None):

        if experiment == "serialize_json":
            serialized_table = table
            format = "JSON"

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
        else:
            table_json = json.loads(table)
            table = pd.DataFrame(table_json['data'], columns=table_json['columns'])
            if experiment == "serialize_csv":
                serialized_table = table.to_csv(index=False)
                format = "CSV"
            elif experiment == "serialize_markdown":
                serialized_table = table.to_markdown(index=False)
                format = "Markdown"

            if experiment == "few-shot":
                serialized_table = table.to_markdown(index=False)
                format = "Markdown"
                prompt = ("You are a table analyst. Your task is to answer questions based on the table content.\n"
                "\n\n"
                f"{formatter}"
                "\n\n"
                "Give the final answer to the question directly without any explanation.\n\n"
                f"Below are {len(shots)} examples of how to answer the question based on the table content.\n"
                "\n\n")

                for i, (shot_table, shot_question, shot_formatter, shot_label) in enumerate(shots):
                    shot_table_json = json.loads(shot_table)
                    shot_table = pd.DataFrame(shot_table_json['data'], columns=shot_table_json['columns'])
                    serialized_shot_table = shot_table.to_markdown(index=False)
                    prompt += (f"Example {i + 1}:\n"
                    f"Read the table below in {format} format:\n"
                    "[TABLE]\n"
                    f"{serialized_shot_table}\n"
                    "\n"
                    f"Question: {shot_question}\n"
                    f"Final Answer: {shot_label}\n\n")


                prompt += (f"Now it's your turn. Read the table below in {format} format:\n"
                "[TABLE]\n"
                f"{serialized_table}\n"
                "\n"
                "Let's get start!\n"
                f"Question: {question}\n"
                "Final Answer: ")

            
            else:
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
    

    def get_shots(self, main_table, ds_task, n_shots):
        shot_ids = []
        shots = []
        while len(shots) < n_shots:
            example = ds_task.shuffle()[0]
            label = example.get("answer")
            question = example.get("question")
            table = example.get("table")
            formatter = example.get("answer_formatter")

            if table != main_table and example['id'] not in shot_ids:
                shots.append((table, question, formatter, label))
                shot_ids.append(example['id']) 

        return shots


    def run(self, model, experiment, batch_size, n_shots=5):
        metric_names = ["rouge"]
        subtasks = ["FactChecking", "NumericalReasoning", "DataAnalysis"] # we do not include Visualization as this does not fit the research

        ds = load_dataset("Multilingual-Multimodal-NLP/TableBench", revision="90593ad8af90f027f6f478b8c4c1981d9f073a83")
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
                    shots = None
                    if experiment == "few-shot":
                        shots = self.get_shots(table, ds_task[split], n_shots)
                    prompt = self.get_prompt(table, question, formatter, experiment, shots)

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