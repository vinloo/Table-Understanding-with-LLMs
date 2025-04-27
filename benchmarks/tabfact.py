from datasets import load_dataset, Split
import textwrap
import evaluate
from tqdm import tqdm
import pandas as pd
from io import StringIO
import random
from .utils import serialize_sentence

class TabFact:

    def get_prompt(self, statement, table_text, table_caption, experiment, fewshot_table=None, fewshot_caption=None, shots=None):
        # Baseline experiment
        if experiment == "baseline":
            prompt = textwrap.dedent(f"""\
                Statement: {statement}

                Table:
                """)
            prompt += table_text
            prompt += textwrap.dedent(f"""\
                                    
                Caption: {table_caption}

                Is the statement entailed or refuted by the table?

                Options:

                A) Refuted
                B) Entailed

                Answer: """)
        else:
            table = pd.read_csv(StringIO(table_text), sep='#')
            if experiment == "explicit_prompt":
                serialized_table = table_text
            elif experiment == "serialize_csv":
                serialized_table = table.to_csv(index=False)
            elif experiment == "serialize_json":
                serialized_table = table.to_json(index=False)
            elif experiment == "serialize_markdown" or experiment == "eval_grpo":
                serialized_table = table.to_markdown(index=False)
            elif experiment == "serialize_sentence":
                serialized_table = serialize_sentence(table)

            if experiment == "few-shot":
                serialized_table = table.to_markdown(index=False)
                serialized_fewshot_table = pd.read_csv(StringIO(fewshot_table), sep='#').to_markdown(index=False)
                prompt = textwrap.dedent(f"""\
                    You are given a table and a statement. Your task is to determine whether the statement 
                    is supported by the information in the table (Entailed) or contradicts it (Refuted). 

                    Below are {len(shots)} examples of similar questions with the following table:
                    """)
                prompt += serialized_fewshot_table
                prompt += textwrap.dedent(f"""\

                    Caption: {fewshot_caption}

                    Examples:
                    """)
                for i, (shot, label) in enumerate(shots):
                    prompt += textwrap.dedent(f"""\
                        Example {i + 1}:
                        Statement: {shot}

                        Based on the table, choose the most accurate option:

                        A) Refuted — The table contradicts the statement.
                        B) Entailed — The table supports the statement.

                        Answer: {label}

                        """)
                prompt += textwrap.dedent(f"""\
                    Now, here is the table and statement you have to answer:

                    Table:
                    """)
                prompt += serialized_table
                prompt += textwrap.dedent(f"""\

                    Caption: {table_caption}

                    Statement: {statement}

                    Based on the table, choose the most accurate option:

                    A) Refuted — The table contradicts the statement.
                    B) Entailed — The table supports the statement.

                    Answer: """)
                
            else:
                prompt = textwrap.dedent(f"""\
                    You are given a table and a statement. Your task is to determine whether the statement 
                    is supported by the information in the table (Entailed) or contradicts it (Refuted). 

                    Statement: {statement}

                    Table:
                    """)
                prompt += serialized_table
                prompt += textwrap.dedent(f"""\

                    Caption: {table_caption}

                    Based on the table, choose the most accurate option:

                    A) Refuted — The table contradicts the statement.
                    B) Entailed — The table supports the statement.

                    Answer: """)
            
            
        return prompt
    

    def get_shots(self, ds, n_shots):
        shots = []
        few_shot_split = Split.TRAIN if Split.TRAIN in ds else (Split.VALIDATION if Split.VALIDATION in ds else Split.TEST)
        for i in range(10):
            try:
                table_ids = ds[few_shot_split].unique("table_id")
                table_id = random.choice(table_ids)
                examples = ds[few_shot_split].filter(lambda x: x["table_id"] == table_id).select(range(n_shots))
                break
            except IndexError:
                if i == 9:
                    raise IndexError(f"No table with {n_shots} related questions found in 10 attempts")
                continue

        fewshot_table = examples[0]["table_text"]
        fewshot_caption = examples[0]["table_caption"]

        for example in examples:
            example_label = "A" if example["label"] == 0 else "B"
            shots.append((example["statement"], example_label))

        return shots, fewshot_table, fewshot_caption


    def run(self, model, experiment, batch_size=1, n_shots=5,):
        ds = load_dataset("wenhu/tab_fact", "tab_fact", trust_remote_code=True)
        split = Split.TEST
        if Split.TEST not in ds:
            split = Split.VALIDATION if Split.VALIDATION in ds else Split.TRAIN
        metrics = {name: evaluate.load(name) for name in ["accuracy"]}
        predictions = []
        references = []
        prompts = []
        labels = []

        shots = None
        fewshot_table = None
        fewshot_caption = None
        

        for i, example in enumerate(tqdm(ds[split], desc="Processing TabFact examples")):
            statement = example.get("statement")
            label = example.get("label")
            table_text = example.get("table_text")
            table_caption = example.get("table_caption")

            if experiment == "few-shot" and i % 100 == 0:
                shots, fewshot_table, fewshot_caption = self.get_shots(ds, n_shots)

            prompt = self.get_prompt(statement=statement, 
                                     table_text=table_text, 
                                     table_caption=table_caption, 
                                     experiment=experiment, 
                                     fewshot_table=fewshot_table, 
                                     fewshot_caption=fewshot_caption, 
                                     shots=shots)
            
            prompts.append(prompt)
            labels.append(label)
            if len(prompts) == batch_size:
                batch_preds = model.predict(prompts, options=["Refuted", "Entailed"])
                predictions.extend(batch_preds)
                references.extend(labels)
                prompts = []
                labels = []
        if prompts:
            batch_preds = model.predict(prompts, options=["Refuted", "Entailed"])
            predictions.extend(batch_preds)
            references.extend(labels)
        results = {}
        for name, metric in metrics.items():
            results[name] = metric.compute(predictions=predictions, references=references)
        return results
