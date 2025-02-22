from datasets import load_dataset, Split
import evaluate
from tqdm import tqdm

class MMLUPro:
    def get_prompt(self, question, options):
        choices = [f"{chr(65 + i)}) {options[i]}" for i in range(len(options))]

        prompt = f"Questions: {question}\n\nOptions:\n"

        for choice in choices:
            prompt += f"\n{choice}"

        prompt += "\n\nAnswer: "

        return prompt
    
    def run(self, model):
        metric_names = ["accuracy"]

        ds = load_dataset("TIGER-Lab/MMLU-Pro")

        split = Split.TEST
        if Split.TEST not in ds:
            split = Split.VALIDATION if Split.VALIDATION in ds else Split.TRAIN

        metrics = {name: evaluate.load(name) for name in metric_names}

        predictions = []
        references = []


        for example in tqdm(ds[split], total=len(ds[split])):
            question = example.get("question")
            options = example.get("options")
            label = example.get("answer")
            prompt = self.get_prompt(question, options)
            pred = model.predict(prompt, options=options)

            if isinstance(label, str):
                label = ord(label) - ord("A")

            predictions.append(pred)
            references.append(label)

        results = {}
        for name, metric in metrics.items():
            results[name] = metric.compute(predictions=predictions, references=references)

        return results
