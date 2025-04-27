from datasets import load_dataset, Split
import evaluate
import torch
from tqdm import tqdm

class MMLU:

    def get_prompt(self, question, options, experiment, shots=None):
        # baseline experiment
        if experiment == "baseline" or experiment == "eval_grpo":
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

        elif experiment == "few-shot":
            prompt = (
                "You are given a multiple-choice question. Carefully read the question "
                "and select the correct answer from the provided options. Respond only "
                "with the letter corresponding to the correct choice.\n"
                f"Below are {len(shots)} examples of similar questions:\n\n"
            )

            if shots:
                for i, shot in enumerate(shots):
                    prompt += f"Example {i + 1}:\n"
                    prompt += f"{shot}\n\n"

            prompt += "Now, here is the question you have to answer:\n\n"
            prompt += f"Question: {question}\n\nOptions:\n"
            choices = [f"{chr(65 + i)}) {options[i]}" for i in range(len(options))]

            for choice in choices:
                prompt += f"\n{choice}"

            prompt += "\n\nAnswer: "

            

        return prompt

    def run(self, model, experiment, batch_size=1, n_shots=5):
        ds = load_dataset("cais/mmlu", "all")

        split = Split.TEST if Split.TEST in ds else (Split.VALIDATION if Split.VALIDATION in ds else Split.TRAIN)
        few_shot_split = Split.TRAIN if Split.TRAIN in ds else (Split.VALIDATION if Split.VALIDATION in ds else Split.TEST)

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

            shots = None
            if experiment == "few-shot":
                shots = []
                # take n_shots examples from ds[Split.TRAIN]
                train_ds = ds[few_shot_split]
                train_ds = train_ds.shuffle(seed=42)
                train_ds = train_ds.select(range(n_shots))
                examples = [train_ds[i] for i in range(n_shots)]
                for shot in examples:
                    shot_question = shot["question"]
                    shot_options = shot["choices"]
                    shot_label = shot["answer"]
                    shot_string = f"Question: {shot_question}\n\nOptions:\n"
                    for i, option in enumerate(shot_options):
                        shot_string += f"\n{chr(65 + i)}) {option}"
                    shot_string += "\n\nAnswer: " + chr(65 + shot_label)
                    shots.append(shot_string)

            prompt = self.get_prompt(question, options, experiment, shots=shots)

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
