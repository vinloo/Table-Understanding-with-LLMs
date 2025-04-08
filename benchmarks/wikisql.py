from datasets import load_dataset, Split
import textwrap
import json
from json.decoder import JSONDecodeError
from tqdm import tqdm
import pandas as pd
import pandasql as ps
import re
import random
from .utils import serialize_sentence

class WikiSQL:

    def get_prompt(self, table, question, experiment, fewshot_table=None, shots=None):
        # Baseline experiment
        if experiment == "baseline":
            prompt = textwrap.dedent("""\
                You are an assistant tasked with answering the questions asked of a given CSV in JSON format. 
                You must answer in a single JSON with one field:
                    * "sql_query": the sql query required to answer the question.
                Requirements:
                    * Only respond with the JSON.
                    * The table's name to select from is 'table'.
                                    
                In the following CSV
                ```csv
                """)
            
            prompt += table.to_csv(index=False)
                
            prompt += textwrap.dedent(f"""\            
                ```
                USER: {question}
                ASSISTANT: {{"sql_query": \"""")
        
        else:
            if experiment == "explicit_prompt":
                serialized_table = table.to_csv(index=False)
            elif experiment == "serialize_json":
                serialized_table = table.to_json(index=False)
            elif experiment == "serialize_markdown":
                serialized_table = table.to_markdown(index=False)
            elif experiment == "serialize_sentence":
                serialized_table = serialize_sentence(table)

            if experiment == "few-shot":
                serialized_table = table.to_json(index=False)
                serialized_fewshot_table = fewshot_table.to_json(index=False)
                prompt = textwrap.dedent("""\
                    You are an AI assistant that translates natural language questions into SQL queries based on a provided CSV dataset. 
                    Your response must be in JSON format with the following structure:

                    {{
                        "sql_query": "<generated SQL query>"
                    }}

                    Guidelines:
                    - Use the table name 'table' in your query.
                    - Ensure the SQL accurately answers the question.
                    - Respond **only** with the JSON output. Do not include explanations.
                    
                    Below are {len(shots)} examples of similar questions with the following table:
                    """)
                prompt += serialized_fewshot_table
                for i, (shot_question, label) in enumerate(shots):
                    prompt += textwrap.dedent(f"""\
                                              
                        Example {i + 1}:
                        USER: {shot_question}
                        ASSISTANT: {{"sql_query": "{label}"}}
                        """)
                    
                prompt += textwrap.dedent(f"""\
                                          
                    Now, here is the dataset you will be working with:
                    """)
                prompt += serialized_table
                prompt += textwrap.dedent(f"""\
                    Now, answer the following user question:

                    USER: {question}
                    ASSISTANT: {{"sql_query": \"""")
            else:
                prompt = textwrap.dedent("""\
                    You are an AI assistant that translates natural language questions into SQL queries based on a provided CSV dataset. 
                    Your response must be in JSON format with the following structure:

                    {{
                        "sql_query": "<generated SQL query>"
                    }}

                    Guidelines:
                    - Use the table name 'table' in your query.
                    - Ensure the SQL accurately answers the question.
                    - Respond **only** with the JSON output. Do not include explanations.
                    
                    Here is the dataset:
                    """)
                
                prompt += serialized_table

                prompt += textwrap.dedent(f"""\            
                    USER: {question}
                    ASSISTANT: {{"sql_query": \"""")
        
        return prompt
    

    def execution_accuracy(self, predictions, references, tables):
        correct = 0
        total = len(predictions)
        for pred, ref, data in zip(predictions, references, tables):
            data # ensure table is in locals scope
            try:
                result_pred = ps.sqldf(pred, locals())
            except Exception as e:
                continue
            try:
                result_ref = ps.sqldf(ref, locals())
            except Exception as e:
                total -= 1
                continue
            if result_pred is not None and result_pred.equals(result_ref):
                correct += 1
        return correct / len(predictions) if total else 0.0


    def logical_form_accuracy(self, predictions, references):
        correct = sum(p == r for p, r in zip(predictions, references))
        return correct / len(predictions) if references else 0.0
    

    def get_shots(self, ds, n_shots):
        few_shot_split = Split.TRAIN if Split.TRAIN in ds else (Split.VALIDATION if Split.VALIDATION in ds else Split.TEST)
        for i in range(20):
            try:
                table_ids = ds[few_shot_split].data.to_pandas().apply(lambda x: x["table"]["id"], axis=1).unique()
                table_id = random.choice(table_ids)
                samples = ds[few_shot_split].filter(lambda x: x["table"]["id"] == table_id).select(range(n_shots))
                break
            except IndexError:
                if i == 19:
                    raise IndexError(f"No table with {n_shots} related questions found in 20 attempts")
                continue

        table_info = samples[0]["table"]
        fewshot_table = pd.DataFrame(table_info["rows"], columns=table_info["header"])
        shots = []
        
        for sample in samples:
            question = sample["question"]
            label = sample["sql"]["human_readable"]
            label = label.replace("FROM table", "FROM data")
            label = label.replace("FROM 'table'", "FROM data")
            label = label.replace('"', "'")

            conditions = sample.get("sql")["conds"]["condition"]
            for condition in conditions:
                if f"'{condition}'" not in label:
                    label = label.replace(f"{condition}", f"'{condition}'")
            for col in fewshot_table.columns:
                if f"{col}" in label and f"'{col}'" not in label:
                    label = label.replace(f"{col}", f"'{col}'")

            label = re.sub(r"COUNT '([^']*)'", r"COUNT('\1')", label)
            label = re.sub(r"MIN '([^']*)'", r"MIN('\1')", label)
            label = re.sub(r"MAX '([^']*)'", r"MAX('\1')", label)
            label = re.sub(r"SUM '([^']*)'", r"SUM('\1')", label)
            label = re.sub(r"AVG '([^']*)'", r"AVG('\1')", label)
            shots.append((question, label))

        return fewshot_table, shots


    def run(self, model, experiment, batch_size=None, n_shots=5):
        ds = load_dataset("Salesforce/wikisql", trust_remote_code=True)

        split = Split.TEST
        if Split.TEST not in ds:
            split = Split.VALIDATION if Split.VALIDATION in ds else Split.TRAIN

        # cap to 1000 examples
        ds[split] = ds[split].select(range(1000))

        predictions = []
        references = []
        tables = []


        shots = None
        fewshot_table = None

        for i, example in enumerate(tqdm(ds[split], total=len(ds[split]))):
            label = example.get("sql")["human_readable"]
            question = example.get("question")
            table_info = example.get("table")
            table = pd.DataFrame(table_info["rows"], columns=table_info["header"])

            if experiment == "few-shot" and i % 100 == 0:
                fewshot_table, shots = self.get_shots(ds, n_shots)
            
            # table_csv = table.to_csv(index=False)
            prompt = self.get_prompt(table, question, experiment, fewshot_table=fewshot_table, shots=shots)

            pred = model.generate(prompt, max_new_tokens=50)
            if experiment == "few-shot":
                pred = pred.split("ASSISTANT: ")[n_shots + 1].strip()
            else:
                pred = pred.split("ASSISTANT: ")[1].strip()

            try:
                # try to get from the first opeining bracket to the last closing bracket
                pred = json.loads(pred[pred.find("{"):pred.rfind("}")+1])
                pred = pred["sql_query"]
            except JSONDecodeError:
                # if the json is not well formatted, try to get the first object between : and , OR : and }
                pred = pred.split('sql_query":')[1].strip()
                try:
                    pred = pred[pred.find("{")+1:pred.find(",")].strip()
                except ValueError:
                    try:
                        pred = pred[pred.find("{")+1:pred.find("}")].strip()
                    except ValueError:
                        print("Failed to parse sql_query")


            pred = pred.replace('"', "'")
            pred = pred.replace("FROM 'table'", "FROM data")
            pred = pred.replace("FROM table", "FROM data")
            pred = pred.replace("from table", "from data")
            pred = pred.replace("from 'table'", "from data")
            label = label.replace("FROM table", "FROM data")
            label = label.replace("FROM 'table'", "FROM data")
            label = label.replace('"', "'")

            conditions = example.get("sql")["conds"]["condition"]
            for condition in conditions:
                if f"'{condition}'" not in label:
                    label = label.replace(f"{condition}", f"'{condition}'")
                if f"'{condition}'" not in pred and f"{condition}" in pred:
                    pred = pred.replace(f"{condition}", f"'{condition}'")
            for col in table.columns:
                if f"{col}" in label and f"'{col}'" not in label:
                    label = label.replace(f"{col}", f"'{col}'")
                if f"{col}" in pred and f"'{col}'" not in pred:
                    pred = pred.replace(f"{col}", f"'{col}'")

            # fix some operatores which are somehow wrong in the dataset labels
            label = re.sub(r"COUNT '([^']*)'", r"COUNT('\1')", label)
            label = re.sub(r"MIN '([^']*)'", r"MIN('\1')", label)
            label = re.sub(r"MAX '([^']*)'", r"MAX('\1')", label)
            label = re.sub(r"SUM '([^']*)'", r"SUM('\1')", label)
            label = re.sub(r"AVG '([^']*)'", r"AVG('\1')", label)

            predictions.append(pred)
            references.append(label)
            tables.append(table)
            
        results = {}
        results[f"exec_accuracy"] = self.execution_accuracy(predictions=predictions, references=references, tables=tables)
        results[f"lf_accuracy"] = self.logical_form_accuracy(predictions=predictions, references=references)

        return results