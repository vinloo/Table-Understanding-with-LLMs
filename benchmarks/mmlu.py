from datasets import load_dataset, Split
import evaluate
import torch
from tqdm import tqdm

class MMLU:

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
        ds = load_dataset("cais/mmlu", "all")

        split = Split.TEST if Split.TEST in ds else (Split.VALIDATION if Split.VALIDATION in ds else Split.TRAIN)

        accuracy_metric = evaluate.load("accuracy")

        predictions = []
        references = []

        prompts = []
        options_list = []
        labels = []

        for example in tqdm(ds[split], desc="Processing MMLU examples"):

            question = example["question"]
            options = example["choices"]
            label = example["answer"]
            prompt = self.get_prompt(question, options, experiment)

            prompts.append(prompt)
            options_list.append(options)
            labels.append(label)

            if len(prompts) == batch_size:
                batch_preds = model.predict(prompts, options=options_list)
                predictions.extend(batch_preds)
                references.extend(labels)

                prompts = []
                options_list = []
                labels = []

        if prompts:
            batch_preds = model.predict(prompts, options=options_list)
            predictions.extend(batch_preds)
            references.extend(labels)

        results = accuracy_metric.compute(predictions=predictions, references=references)
        
        return results
