import argparse
import dotenv
import os
from huggingface_hub import login
from models import Llama3_1_8bModel
from runners.benchmark_runner import BenchmarkRunner

def main():
    dotenv.load_dotenv()
    login(os.environ.get("HF_TOKEN"))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=["llama3.1:8b"],
        required=True,
        help="Model to use for evaluation."
    )
    parser.add_argument(
        "-b",
        "--benchmark",
        type=str,
        required=True,
        help="Path to the benchmark config file."
    )
    args = parser.parse_args()

    if args.model == "llama3.1:8b":
        model = Llama3_1_8bModel()
    else:
        raise ValueError("Unsupported model selected.")
    
    runner = BenchmarkRunner(args.benchmark)
    results = runner.run(model)

    print("Evaluation Results:")
    for metric, value in results.items():
        print(f"{metric}: {value}")

if __name__ == "__main__":
    main()
