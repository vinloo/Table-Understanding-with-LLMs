from datasets import load_dataset, Split
import evaluate
from tqdm import tqdm

class MMLUPro:
    def get_prompt(self, question, options, experiment):
        # baseline experiment
        if experiment == "baseline":
            choices = [f"{chr(65 + i)}) {options[i]}" for i in range(len(options))]

            prompt = f"Questions: {question}\n\nOptions:\n"

            for choice in choices:
                prompt += f"\n{choice}"

            prompt += "\n\nAnswer: "

        # explicit prompt experiment
        elif experiment == "explicit_prompt":
            choices = [f"{chr(65 + i)}) {options[i]}" for i in range(len(options))]

            prompt = (
                "You are given a multiple-choice question. Carefully read the question "
                "and select the correct answer from the provided options. Respond only "
                "with the letter corresponding to the correct choice.\n\n"
                f"Question: {question}\n\nOptions:\n"
            )

            for choice in choices:
                prompt += f"\n{choice}"

            prompt += "\n\nAnswer: "
            return prompt
    
    def run(self, model, experiment, batch_size=1):
        ds = load_dataset("TIGER-Lab/MMLU-Pro")

        split = Split.TEST
        if Split.TEST not in ds:
            split = Split.VALIDATION if Split.VALIDATION in ds else Split.TRAIN

        metrics = {name: evaluate.load(name) for name in ["accuracy"]}

        predictions = []
        references = []
        prompts = []
        options_list = []
        labels = []

        for example in tqdm(ds[split], desc="Processing MMLUPro examples"):
            question = example.get("question")
            options = example.get("options")
            label = example.get("answer")
            prompt = self.get_prompt(question, options, experiment)
            prompts.append(prompt)
            options_list.append(options)

            if isinstance(label, str):
                label = ord(label) - ord("A")

            labels.append(label)

            if len(prompts) == batch_size:
                longest_options = max(options_list, key=lambda opts: max(len(opt) for opt in opts))
                batch_preds = model.predict(prompts, options=longest_options)
                predictions.extend(batch_preds)
                references.extend(labels)
                prompts = []
                options_list = []
                labels = []

        if prompts:
            longest_options = max(options_list, key=lambda opts: max(len(opt) for opt in opts))
            batch_preds = model.predict(prompts, options=longest_options)
            predictions.extend(batch_preds)
            references.extend(labels)

        results = {}
        for name, metric in metrics.items():
            results[name] = metric.compute(predictions=predictions, references=references)

        return results
