import json
from datasets import load_dataset, Split
import evaluate
from tqdm import tqdm
from prompts import get_prompt

class BenchmarkRunner:
    def __init__(self, benchmark):
        with open("benchmarks.json", "r") as f:
            self.benchmark = benchmark
            self.config = json.load(f)[benchmark]

    def run(self, model):
        dataset_config = self.config

        dataset_name = dataset_config.get("dataset")
        subset = dataset_config.get("subset")
        task_type = dataset_config.get("task_type")
        input_key = dataset_config.get("input_key")
        label_key = dataset_config.get("label_key")
        metric_names = dataset_config.get("metrics", [])

        ds = load_dataset(dataset_name, subset) if subset else load_dataset(dataset_name)

        split = Split.TEST
        if Split.TEST not in ds:
            split = Split.VALIDATION if Split.VALIDATION in ds else Split.TRAIN

        metrics = {name: evaluate.load(name) for name in metric_names}

        predictions = []
        references = []

        for i, example in tqdm(enumerate(ds[split])):
            text = example.get(input_key)
            label = example.get(label_key)

            if text is None or label is None:
                continue

            if task_type == "multiple_choice":
                options_key = dataset_config.get("options_key")
                options = example.get(options_key)
                prompt = get_prompt(self.benchmark, question=text, options=options)
                pred = model.predict(prompt, options=options)

                if isinstance(label, str):
                    label = ord(label) - ord("A")

            else:
                prompt = get_prompt(self.benchmark, question=text)
                pred = model.generate(prompt)

            predictions.append(pred)
            references.append(label)
            if i == 300:
                break

        results = {}
        for name, metric in metrics.items():
            result = metric.compute(predictions=predictions, references=references)
            results[name] = result

        return results