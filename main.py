import argparse
import dotenv
import os
from huggingface_hub import login
import wandb
import datetime
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
        choices=["tabfact", "tabularbenchmark", "tablebench", "databench", "wikisql", "mmlu", "mmlu_pro"],
        required=True,
        help="Path to the benchmark config file."
    )
    parser.add_argument(
        "-e",
        "--experiment",
        type=str,
        default="baseline",
        choices=["baseline"]
    )
    parser.add_argument(
        "--nolog",
        action="store_true",
        help="Disable logging to wandb."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode."
    )
    args = parser.parse_args()

    experiment_name = args.model + "-" + args.benchmark + "-" + args.experiment + "-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    wandb_mode = "disabled" if args.nolog else "online"


    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "table-understanding"),
        entity=os.environ.get("WANDB_ENTITY"),
        name=experiment_name,
        config=vars(args),
        reinit=True,
        mode=wandb_mode
    )

    if args.model == "llama3.1:8b":
        model = Llama3_1_8bModel()
    else:
        raise ValueError("Unsupported model selected.")
    
    runner = BenchmarkRunner(args.benchmark, args.debug)
    results = runner.run(model)

    print("Evaluation Results:")
    for metric, value in results.items():
        print(f"{metric}: {value}")

    wandb.log(results)

if __name__ == "__main__":
    main()
