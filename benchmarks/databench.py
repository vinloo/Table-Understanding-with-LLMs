from datasets import load_dataset, Split
import textwrap
import evaluate
import json
from json.decoder import JSONDecodeError
from tqdm import tqdm
import pandas as pd

class DataBench:

    def get_prompt(self, table, question):
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
        
        prompt += table
            
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


    def run(self, model, batch_size=None):
        # batching does not work for this benchmark
        metric_names = ["accuracy"]
        ds = load_dataset("cardiffnlp/databench", "qa")
        subtasks = ["boolean", "category", "number", "list[category]", "list[number]"]

        split = Split.TEST
        if Split.TEST not in ds:
            split = Split.VALIDATION if Split.VALIDATION in ds else Split.TRAIN

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
            for example in tqdm(ds_task[split], total=len(ds_task[split])):
                label = example.get("sample_answer")
                question = example.get("question")
                
                table_id = example['dataset']
                table = pd.read_parquet(f"hf://datasets/cardiffnlp/databench/data/{table_id}/sample.parquet")
                table = table.to_csv(index=False)
                prompt = self.get_prompt(table, question)

                if len(prompt) > 10000:
                    continue

                pred = model.generate(prompt, max_new_tokens=50)
                pred = pred.split("ASSISTANT: ")[1].strip()

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
                    overlap = set(pred).intersection(set(label))
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