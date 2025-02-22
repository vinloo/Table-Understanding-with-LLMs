import json
from datasets import load_dataset, Split
import evaluate
from tqdm import tqdm
from prompts import get_prompt

class BenchmarkRunner:
    def __init__(self, benchmark, debug):
        with open("benchmarks.json", "r") as f:
            self.benchmark = benchmark
            self.config = json.load(f)[benchmark]
            self.debug = debug

    def run(self, model):
        dataset_config = self.config

        dataset_name = dataset_config.get("dataset")
        subset = dataset_config.get("subset")
        input_key = dataset_config.get("input_key")
        label_key = dataset_config.get("label_key")
        metric_names = dataset_config.get("metrics", [])

        ds = load_dataset(dataset_name, subset, trust_remote_code=True) if subset else load_dataset(dataset_name, trust_remote_code=True)

        split = Split.TEST
        if Split.TEST not in ds:
            split = Split.VALIDATION if Split.VALIDATION in ds else Split.TRAIN

        metrics = {name: evaluate.load(name) for name in metric_names}

        predictions = []
        references = []

        for i, example in enumerate(tqdm(ds[split], total=len(ds[split]))):
            text = example.get(input_key)
            label = example.get(label_key)

            if text is None or label is None:
                continue

            if self.benchmark == "mmlu" or self.benchmark == "mmlu_pro":
                options_key = dataset_config.get("options_key")
                options = example.get(options_key)
                prompt = get_prompt(self.benchmark, question=text, options=options)
                pred = model.predict(prompt, options=options)

                if isinstance(label, str):
                    label = ord(label) - ord("A")
            elif self.benchmark == 'tabfact':
                table_text = example.get("table_text")
                table_caption = example.get("table_caption")
                prompt = get_prompt(self.benchmark, question=text, table_text=table_text, table_caption=table_caption)
                pred = model.predict(prompt, options=["Refuted", "Entailed"])

            else:
                prompt = get_prompt(self.benchmark, question=text)
                pred = model.generate(prompt)

            predictions.append(pred)
            references.append(label)

            if self.debug:
                if i == 0:
                    print("Debugging is enabled, printing first example:")
                    print("Prompt:", prompt)
                    print("Prediction:", pred)
                    print("Reference:", label)
                    print()
                elif i == 100:
                    print("Debugging is enabled, stopping after 100 examples.")
                    break
                

        results = {}
        for name, metric in metrics.items():
            result = metric.compute(predictions=predictions, references=references)
            results[name] = result

        return results