from datasets import load_dataset, Split, Dataset
import evaluate
import re
from tqdm import tqdm
import textwrap
import pandas as pd
import json
from .utils import serialize_sentence

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
            elif experiment in ["serialize_markdown", "eval_grpo", "cot-few-shot", "cot-zero-shot"]:
                serialized_table = table.to_markdown(index=False)
                format = "Markdown"
            elif experiment == "serialize_sentence":
                serialized_table = serialize_sentence(table)
                format = "Sentence"

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

            elif experiment in ["cot-few-shot"]:
                prompt = ("You are a table analyst. Your task is to answer questions based on the table content.\n"
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
                    "\n\n"
                    "First write up to 120 tokens of step-by-step reasoning.\n"
                    "When you are completely finished, write:\n"
                    f"Question: {shot_question}\n"
                    "<reasoning here>\n"
                    "===== FINAL ANSWER START =====\n"
                    f"{shot_label}\n"
                    "===== FINAL ANSWER END =====\n\n")
                
                prompt += ("Now it's your turn. Read the table below in Markdown format:\n""[TABLE]\n"
                    f"{serialized_table}\n\n"
                    "First write up to 120 tokens of step-by-step reasoning.\n"
                    "When you are completely finished, write:\n"
                    "===== FINAL ANSWER START =====\n"
                    "<your concise answer here>\n"
                    "===== FINAL ANSWER END =====\n\n"
                    f"Question: {question}")
                
            elif experiment in ["cot-zero-shot"]:#, "eval_grpo"]:
                prompt = ("You are a table analyst. Your task is to answer questions based on the table content.\n"
                            f"Read the table below in Markdown format:\n"

                            "[TABLE]\n"
                            f"{serialized_table}\n\n"
                            "First write up to 120 tokens of step-by-step reasoning.\n"
                            "When you are completely finished, write:\n"
                            "===== FINAL ANSWER START =====\n"
                            "<your concise answer here>\n"
                            "===== FINAL ANSWER END =====\n\n"
                            f"Question: {question}\n\n")
            
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
    

    def get_tabular_attention_prompt(self, question, formatter, separator=','):
        prompt = ("You are a table analyst. Your task is to answer questions based on the table content.\n"
                "\n\n"
                f"{formatter}"
                "\n\n"
                "Give the final answer to the question directly without any explanation.\n"
                "\n"
                f"Read the table below:\n"
                "[TABLE]\n")
        prompt_post =  ("\n\n"
                "Let's get start!\n"
                f"Question: {question}\n\n"
                "Final Answer: ")
        return prompt, prompt_post
    

    def get_shots(self, main_table, ds_task, n_shots, experiment):
        shot_ids = []
        shots = []
        while len(shots) < n_shots:
            example = ds_task.shuffle()[0]
            if experiment == "eval_grpo":
                label = example.get("response").split("Final Answer: ")[-1]
                formatter = example.get("instruction").split("\n\n\n")[1]
            else:
                label = example.get("answer")
                formatter = example.get("answer_formatter")

            question = example.get("question")
            table = example.get("table")

            if table != main_table and example['id'] not in shot_ids:
                shots.append((table, question, formatter, label))
                shot_ids.append(example['id']) 

        return shots


    def run(self, model, experiment, batch_size, n_shots=5):
        metric_names = ["rouge"]
        subtasks = ["FactChecking", "NumericalReasoning", "DataAnalysis"] # we do not include Visualization as this does not fit the research

        if experiment != "eval_grpo":
            ds = load_dataset("Multilingual-Multimodal-NLP/TableBench", revision="90593ad8af90f027f6f478b8c4c1981d9f073a83")
        else:
            ds = load_dataset("Multilingual-Multimodal-NLP/TableInstruct")

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

            ds_task_split = ds_task[split]
            if len(ds_task_split) > 500:
                ds_task_split = ds_task_split.shuffle(seed=42).select(range(500))

            for example in tqdm(ds_task_split, total=len(ds_task_split)):

                if experiment == "eval_grpo":
                    label = example.get("response").split("Final Answer: ")[-1]
                    formatter = example.get("instruction").split("\n\n\n")[1]
                else:
                    label = example.get("answer")
                    formatter = example.get("answer_formatter")


                question = example.get("question")
                table = example.get("table")
                
                if experiment == "baseline":
                    prompt = example.get("instruction")
                elif experiment != "tabular_attention":
                    shots = None
                    if experiment == "few-shot" or experiment == "cot-few-shot":
                        shots = self.get_shots(table, ds_task_split, n_shots, experiment)
                        print(shots, flush=True)
                    try:
                        prompt = self.get_prompt(table, question, formatter, experiment, shots)
                    except Exception as e:
                        print(f"Error in prompt generation: {e}", flush=True)
                        continue

                if experiment != "tabular_attention" and experiment not in ["cot-zero-shot", "cot-few-shot"]:#, "eval_grpo"]:
                    pred = model.generate(prompt, max_new_tokens=50).split(question)[-1]
                    # print(pred, flush=True)
                elif experiment in ["cot-zero-shot", "cot-few-shot"]:#, "eval_grpo"]:
                    pred = model.generate(prompt, max_new_tokens=150)
                else:
                    prompt_pre, prompt_post = self.get_tabular_attention_prompt(question, formatter)
                    table_json = json.loads(table)
                    table = pd.DataFrame(table_json['data'], columns=table_json['columns'])
                    try:
                        pred = model.generate_with_tabular_attention(prompt_pre, table, prompt_post, separator=',').split(question)[-1]
                    except Exception as e:
                        print(f"Error in model generation: {e}", flush=True)
                        continue
                
                if experiment in ["cot-few-shot", "cot-zero-shot"]:#, "eval_grpo"]:
                    print(f"UNFORMATTED: {pred}", flush=True)
                    try:
                        if experiment != "cot-few-shot":
                            split_index = 2
                        else:
                            split_index = n_shots + 2
                        pred = pred.split("===== FINAL ANSWER START =====")[split_index].split("===== FINAL ANSWER END =====")[0]
                    except Exception as e:
                        pred = pred.split("\n")[-1].strip()
                    print("----------------------------------------------", flush=True)
                    print(f"FORMATTED: {pred}", flush=True)
                    print("----------------------------------------------\n\n\n\n", flush=True)

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