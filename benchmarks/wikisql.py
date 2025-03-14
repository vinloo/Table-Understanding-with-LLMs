from datasets import load_dataset, Split
import textwrap
import json
from json.decoder import JSONDecodeError
from tqdm import tqdm
import pandas as pd
import pandasql as ps
import re

class WikiSQL:

    def get_prompt(self, table, question, experiment):
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
            
            prompt += table
                
            prompt += textwrap.dedent(f"""\            
                ```
                USER: {question}
                ASSISTANT: {{"sql_query": \"""")
        
        # Explicit prompt experiment
        elif experiment == "explicit_prompt":
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
                
                Here is the CSV dataset:
                ```csv
                """)
            
            prompt += table

            prompt += textwrap.dedent(f"""\            
                ```
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


    def run(self, model, experiment, batch_size=None):
        ds = load_dataset("Salesforce/wikisql", trust_remote_code=True)

        split = Split.TEST
        if Split.TEST not in ds:
            split = Split.VALIDATION if Split.VALIDATION in ds else Split.TRAIN

        # cap to 1000 examples
        ds[split] = ds[split].select(range(1000))

        predictions = []
        references = []
        tables = []

        for example in tqdm(ds[split], total=len(ds[split])):
            label = example.get("sql")["human_readable"]
            question = example.get("question")
            table_info = example.get("table")
            table = pd.DataFrame(table_info["rows"], columns=table_info["header"])
            
            table_csv = table.to_csv(index=False)
            prompt = self.get_prompt(table_csv, question, experiment)

            pred = model.generate(prompt, max_new_tokens=50)
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