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
    args = parser.parse_args()

    experiment_name = args.model + "-" + args.benchmark + "-" + args.experiment + "-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Initialize WandB with credentials and CLI parameters from .env and terminal respectively
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "table-understanding"),
        entity=os.environ.get("WANDB_ENTITY"),
        name=experiment_name,
        config=vars(args),
        reinit=True
    )

    if args.model == "llama3.1:8b":
        model = Llama3_1_8bModel()
    else:
        raise ValueError("Unsupported model selected.")
    
    runner = BenchmarkRunner(args.benchmark)
    results = runner.run(model)

    print("Evaluation Results:")
    for metric, value in results.items():
        print(f"{metric}: {value}")

    # Log results to WandB
    wandb.log(results)

if __name__ == "__main__":
    main()
