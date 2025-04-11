import argparse
import dotenv
import os
from huggingface_hub import login
import wandb
import datetime
from model import Model
from benchmarks import TableBench, TabFact, MMLU, MMLUPro, DataBench, WikiSQL


def main():
    dotenv.load_dotenv()
    login(os.environ.get("HF_TOKEN"))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=["llama3.1:8b", "llama3:70b", "jellyfish"],
        required=True,
        help="Model to use for evaluation."
    )
    parser.add_argument(
        "-b",
        "--benchmark",
        type=str,
        choices=["tabfact", "tablebench", "databench", "wikisql", "mmlu", "mmlu_pro"],
        required=True,
        help="Path to the benchmark config file."
    )
    parser.add_argument(
        "-e",
        "--experiment",
        type=str,
        default="baseline",
        choices=["baseline", "explicit_prompt", "serialize_markdown", "serialize_csv", "serialize_json", "serialize_sentence", "few-shot", "tabular_attention"],
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
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for processing prompts."
    )
    parser.add_argument(
        "--n_shots",
        type=int,
        default=3,
        help="Number of shots for few-shot learning."
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

    model = Model(args.model)
        
    if args.benchmark == "tablebench":
        benchmark = TableBench()
    elif args.benchmark == "tabfact":
        benchmark = TabFact()
    elif args.benchmark == "databench":
        benchmark = DataBench()
    elif args.benchmark == "wikisql":
        benchmark = WikiSQL()
    elif args.benchmark == "mmlu":
        benchmark = MMLU()
    elif args.benchmark == "mmlu_pro":
        benchmark = MMLUPro()
    else:
        raise ValueError("Unsupported benchmark selected.")

    results = benchmark.run(model, args.experiment, batch_size=args.batch_size, n_shots=args.n_shots)

    print("Evaluation Results:")
    for metric, value in results.items():
        print(f"{metric}: {value}")

    wandb.log(results)

if __name__ == "__main__":
    main()
