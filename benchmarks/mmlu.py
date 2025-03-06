from datasets import load_dataset, Split
import evaluate
from tqdm import tqdm

class MMLU:

    def get_prompt(self, question, options):
        choices = [f"{chr(65 + i)}) {options[i]}" for i in range(len(options))]

        prompt = f"Questions: {question}\n\nOptions:\n"

        for choice in choices:
            prompt += f"\n{choice}"

        prompt += "\n\nAnswer: "

        return prompt

    def run(self, model, batch_size=1):
        ds = load_dataset("cais/mmlu", "all")

        split = Split.TEST
        if Split.TEST not in ds:
            split = Split.VALIDATION if Split.VALIDATION in ds else Split.TRAIN

        metrics = {name: evaluate.load(name) for name in ["accuracy"]}

        predictions = []
        references = []
        prompts = []
        options_list = []
        labels = []

        for example in tqdm(ds[split], desc="Processing MMLU examples"):
            question = example.get("question")
            options = example.get("choices")
            label = example.get("answer")
            prompt = self.get_prompt(question, options)
            prompts.append(prompt)
            options_list.append(options)
            labels.append(label)

            if len(prompts) == batch_size:
                batch_preds = model.predict(prompts, options=options_list[0])
                predictions.extend(batch_preds)
                references.extend(labels)
                prompts = []
                options_list = []
                labels = []

        if prompts:
            batch_preds = model.predict(prompts, options=options_list[0])
            predictions.extend(batch_preds)
            references.extend(labels)

        results = {}
        for name, metric in metrics.items():
            results[name] = metric.compute(predictions=predictions, references=references)

        return results
