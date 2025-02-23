from datasets import load_dataset, Split
import textwrap
import evaluate
from tqdm import tqdm

class TabFact:
    def get_prompt(self, statement, table_text, table_caption):
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
        return prompt

    def run(self, model):
        metric_names = ["accuracy"]

        ds = load_dataset("wenhu/tab_fact", "tab_fact", trust_remote_code=True)

        split = Split.TEST
        if Split.TEST not in ds:
            split = Split.VALIDATION if Split.VALIDATION in ds else Split.TRAIN

        metrics = {name: evaluate.load(name) for name in metric_names}

        predictions = []
        references = []

        for example in tqdm(ds[split], total=len(ds[split])):
            statement = example.get("statement")
            label = example.get("label")
            table_text = example.get("table_text")
            table_caption = example.get("table_caption")
            prompt = self.get_prompt(statement=statement, table_text=table_text, table_caption=table_caption)
            pred = model.predict(prompt, options=["Refuted", "Entailed"])
            predictions.append(pred)
            references.append(label)

        results = {}
        for name, metric in metrics.items():
            results[name] = metric.compute(predictions=predictions, references=references)

        return results
