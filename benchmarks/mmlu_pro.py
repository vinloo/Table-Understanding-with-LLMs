from datasets import load_dataset, Split
import evaluate
from tqdm import tqdm

class MMLUPro:
    def get_prompt(self, question, options, experiment, shots=None):
        # baseline experiment
        if experiment == "baseline" or experiment == "eval_grpo":
            choices = [f"{chr(65 + i)}) {options[i]}" for i in range(len(options))]

            prompt = f"Question: {question}\n\nOptions:\n"

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
        
        # few-shot experiment
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

        if experiment == "few-shot":
            few_shot_split = Split.TRAIN if Split.TRAIN in ds else (Split.VALIDATION if Split.VALIDATION in ds else Split.TEST)

        for example in tqdm(ds[split], desc="Processing MMLUPro examples"):
            question = example.get("question")
            options = example.get("options")
            label = example.get("answer")
            shots = None
            if experiment == "few-shot":
                shots = []
                train_ds = ds[few_shot_split].shuffle(seed=42).select(range(n_shots))
                examples_shot = [train_ds[i] for i in range(n_shots)]
                for shot in examples_shot:
                    shot_question = shot.get("question")
                    shot_options = shot.get("options")
                    shot_label = shot.get("answer")
                    # Convert shot_label if needed
                    if isinstance(shot_label, int):
                        answer_letter = chr(65 + shot_label)
                    else:
                        answer_letter = shot_label
                    shot_string = f"Question: {shot_question}\n\nOptions:\n"
                    for i, option in enumerate(shot_options):
                        shot_string += f"\n{chr(65 + i)}) {option}"
                    shot_string += "\n\nAnswer: " + answer_letter
                    shots.append(shot_string)
            prompt = self.get_prompt(question, options, experiment, shots=shots)
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
