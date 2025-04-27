from datasets import load_dataset, Split
import textwrap
import evaluate
import json
from json.decoder import JSONDecodeError
from tqdm import tqdm
import pandas as pd
from io import StringIO
import os
import hashlib
import dotenv
from .utils import serialize_sentence

class DataBench:

    def get_prompt(self, table, question, experiment, shots=None, shots_table=None):
        # Baseline experiment
        if experiment == "baseline":
            prompt = textwrap.dedent("""\
                You are an assistant tasked with answering the questions asked of a given CSV in JSON format. 
                You must answer in a single JSON with two fields:
                    * "answer": answer using information from
                    the provided CSV only.
                    * "columns_used": list of columns from the
                    CSV used to get the answer.
                Requirements:
                    * Only respond with the JSON.
                In the following CSV
                ```csv
                """)
            
            prompt += table.to_csv(index=False)
                
            prompt += textwrap.dedent(f"""\
                ```
                USER: {question}
                ASSISTANT: {{"answer":""")
        
        else:
            if experiment == "explicit_prompt":
                serialized_table = table.to_csv(index=False)
            elif experiment == "serialize_json":
                serialized_table = table.to_json(index=False)
            elif experiment == "serialize_markdown" or experiment == "eval_grpo":
                serialized_table = table.to_markdown(index=False)
            elif experiment == "serialize_sentence":
                serialized_table = serialize_sentence(table)

            if experiment == "few-shot":
                serialized_table = table.to_markdown(index=False)
                serialized_shot_table = shots_table.to_markdown(index=False)
                prompt = ("You are an assistant tasked with answering the questions asked of a given table in Markdown format.\n" 
                "You must answer in a single JSON with two fields:\n"
                '    * "answer": answer using information from\n'
                '    the provided CSV only.\n'
                "Requirements:\n"
                "    * Only respond with the JSON.\n\n"
                f"Below are {len(shots['question'])} examples of how to answer the question based on the table content.\n\n"
                f"Read the table below in Markdown format:\n"
                "[TABLE]\n"
                f"{serialized_shot_table}\n"
                "\n")

                for i, shot in pd.DataFrame(shots).iterrows(): 
                    
                    shot_label = shot["sample_answer"]
                    if shot_label is None:
                        shot_label = "nan"
                    elif shot_label in ["True", "False"]:
                        shot_label = shot_label.lower()
                    #int or float
                    elif shot_label.replace('.','',1).replace('-','',1).isdigit() and not (not shot_label.startswith('-') and "-" in shot_label):
                        shot_label = str(float(shot_label))
                    elif not shot_label.startswith("[") and not shot_label.endswith("]"):
                        if not shot_label.startswith('"') and not shot_label.endswith('"') and not shot_label.startswith("'") and not shot_label.endswith("'"):
                            shot_label = shot_label.lower()
                            shot_label = f'"{shot_label}"'
                    # do the same for items in a list
                    elif shot_label.startswith("[") and shot_label.endswith("]"):
                        new_shot_label = []
                        for item in shot_label[1:-1].split(","):
                            item = item.strip()
                            if item.replace('.','',1).replace('-','',1).isdigit() and not (not item.startswith('-') and "-" in item):
                                item = float(item)
                            elif item in ["True", "False"]:
                                item = item.lower()
                            elif not item.startswith('"') and not item.endswith('"') and not item.startswith("'"):
                                item = f'"{item}"'
                            new_shot_label.append(item)
                        shot_label = str(new_shot_label)

                    shot_label = shot_label.replace("'", '')

                    prompt += (f"Example {i + 1}:\n"
                    f"USER: {shot['question']}\n"
                    f'ASSISTANT: {{"answer": {shot_label}}}\n\n')


                prompt += (f"Now it's your turn. Read the table below in Markdown format:\n"
                "[TABLE]\n"
                f"{serialized_table}\n"
                "\n"
                f"USER: {question}\n"
                'ASSISTANT: {{"answer":')

            else:
                prompt = textwrap.dedent("""\
                    You are an AI assistant that answers questions based on the provided CSV dataset. 
                    Your response must be in JSON format with the following structure:

                    {{
                        "answer": "<answer using information from the CSV>",
                        "columns_used": ["<list of relevant columns used>"]
                    }}

                    Guidelines:
                    - Your answer must be derived strictly from the CSV data.
                    - List only the columns directly used to determine the answer.
                    - Respond **only** with the JSON. Do not include explanations.

                    Here is the CSV dataset:
                    ```csv
                    """)
                
                prompt += serialized_table

                prompt += textwrap.dedent(f"""\
                    ```
                    USER: {question}
                    ASSISTANT: {{"answer":""")

        return prompt
    

    def custom_accuracy(self, predictions, references):
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length.")

        correct = sum(p == r for p, r in zip(predictions, references))
        return correct / len(predictions) if references else 0.0


    def read_parquet_with_cache(self, file_path):
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

    def run(self, model, experiment, batch_size=None, n_shots=5):
        # batching does not work for this benchmark
        metric_names = ["accuracy"]

        if experiment != "eval_grpo":
            ds = load_dataset("cardiffnlp/databench", "qa", split="train")
        else:
            ds = load_dataset("cardiffnlp/databench", "semeval", split="dev")
        
        
        subtasks = ["list[number]", "boolean", "category", "number", "list[category]"]

        # split = Split.TEST
        # if Split.TEST not in ds:
        #     split = Split.VALIDATION if Split.VALIDATION in ds else Split.TRAIN

        metrics = {name: evaluate.load(name) for name in metric_names}


        predictions = {
            "boolean": [],
            "category": [],
            "number": [],
            "list[category]": [],
            "list[number]": []
        }
        references = {
            "boolean": [],
            "category": [],
            "number": [],
            "list[category]": [],
            "list[number]": []
        }

        for task in subtasks:
            ds_task = ds.filter(lambda x: x['type'] == task)
            shots = None
            shots_table = None
            for i, example in enumerate(tqdm(ds_task, total=len(ds_task))):
                label = example.get("sample_answer")
                question = example.get("question")
                
                table_id = example['dataset']
                table = self.read_parquet_with_cache(f"hf://datasets/cardiffnlp/databench/data/{table_id}/sample.parquet")

                
                if experiment == "few-shot" and i % 50 == 0:
                    shots_table_id = table_id
                    while shots_table_id == table_id:
                        shots_table_id = ds_task.shuffle()[0]['dataset']
                    shots_all_samples = ds_task.filter(lambda x: x['dataset'] == shots_table_id)
                    shots_table = self.read_parquet_with_cache(f"hf://datasets/cardiffnlp/databench/data/{shots_table_id}/sample.parquet")
                    
                    # shots_table = pd.read_parquet(f"hf://datasets/cardiffnlp/databench/data/{shots_table_id}/sample.parquet")
                    shots = {}
                    shots["question"] = shots_all_samples['question'][:n_shots]
                    shots["sample_answer"] = shots_all_samples['sample_answer'][:n_shots]

                prompt = self.get_prompt(table, question, experiment, shots=shots, shots_table=shots_table)

                if len(prompt) > 15000:
                    continue

                pred = model.generate(prompt, max_new_tokens=50)

                if experiment == "few-shot":
                    pred = pred.split("ASSISTANT: ")[len(shots["question"]) + 1].strip()
                else:
                    pred = pred.split("ASSISTANT: ")[1].strip()

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


                if isinstance(pred, list):
                    if not isinstance(label, list):
                        label = [label]
                    overlap = self.list_to_set(pred).intersection(self.list_to_set(label))
                    if len(overlap) == len(pred):
                        predictions[task].append(1)
                        references[task].append(1)
                    else:
                        predictions[task].append(0)
                        references[task].append(1)
                else:
                    predictions[task].append(pred)
                    references[task].append(label)
                
        results = {}
        for name, metric in metrics.items():
            for task in subtasks:
                preds = predictions[task]
                refs = references[task]

                results[f"{name}/{task}"] = self.custom_accuracy(predictions=preds, references=refs)

        return results
    
    def list_to_set(self, lst):
        result = set()
        for item in lst:
            try:
                result.add(item)
            except TypeError:
                continue  # Skip unhashable items
        return result