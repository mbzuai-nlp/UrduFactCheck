import os
import argparse
import json
from openfactcheck import OpenFactCheck, OpenFactCheckConfig

# Set up argument parser
parser = argparse.ArgumentParser(description="Evaluate fact-checking claims.")
parser.add_argument(
    "--dataset",
    type=str,
    help="Path to the dataset file.",
)
parser.add_argument(
    "--factchecker",
    type=str,
    help="Name of the fact-checker to use.",
)

args = parser.parse_args()
if args.dataset:
    DATASET_PATH = args.dataset
else:
    # Return an error if no dataset is provided
    raise ValueError("Please provide a dataset path using --dataset argument.")
if args.factchecker:
    FACTCHECKER = args.factchecker
else:
    # Return an error if no fact-checker is provided
    raise ValueError("Please provide a fact-checker name using --factchecker argument.")

if __name__ == "__main__":
    CONFIG_PATH = "config.json"
    PROCESSED_PATH = "../../datasets/processed/claims"
    DATASET_NAME = args.dataset
    DATASET_PATH = f"{PROCESSED_PATH}/{DATASET_NAME}/data_gpt-4o_annotated.json"

    # Set the environment variable for cost
    os.environ["SAVE_SERPER_COST"] = "True"
    os.environ["SERPER_COST_PATH"] = (
        f"evaluate_factchecker/{FACTCHECKER}/{DATASET_NAME}/serper_cost.jsonl"
    )
    os.environ["SAVE_MODEL_COST"] = "True"
    os.environ["MODEL_COST_PATH"] = (
        f"evaluate_factchecker/{FACTCHECKER}/{DATASET_NAME}/model_cost.jsonl"
    )

    # Create the output directory if it doesn't exist
    os.makedirs(f"evaluate_factchecker/{FACTCHECKER}/{DATASET_NAME}", exist_ok=True)

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    config = OpenFactCheckConfig(CONFIG_PATH)
    config.output_path = f"evaluate_factchecker/{FACTCHECKER}/{DATASET_NAME}"
    ofc = OpenFactCheck(config).ResponseEvaluator

    # Check if the results file already exists
    existing_results = {}
    cleaned_results = {}
    if os.path.exists(
        f"evaluate_factchecker/{FACTCHECKER}/{DATASET_NAME}/results.jsonl"
    ):
        with open(
            f"evaluate_factchecker/{FACTCHECKER}/{DATASET_NAME}/results.jsonl",
            "r",
            encoding="utf-8",
        ) as f:
            results = [json.loads(line) for line in f.readlines()]

            # Convert the results to a dictionary
            for result in results:
                for claim, data in result.items():
                    existing_results[claim] = data

        for claim, existing_result in existing_results.items():
            if isinstance(existing_result["response"], bool):
                print(f"Claim already exists: {claim}")
                cleaned_results[claim] = existing_result

    # Rewrite the cleaned results to the file
    with open(
        f"evaluate_factchecker/{FACTCHECKER}/{DATASET_NAME}/results.jsonl",
        "w",
        encoding="utf-8",
    ) as f:
        for claim, data in cleaned_results.items():
            f.write(
                json.dumps(
                    {
                        claim: {
                            "response": data["response"],
                            "factchecker": FACTCHECKER,
                            "dataset": DATASET_NAME,
                        }
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    for row in dataset:
        if row["claim_urdu"] in cleaned_results:
            print(f"Claim already exists: {row['claim_urdu']}")
            continue

        claim = row["claim_urdu"]

        response = ofc.evaluate(
            response=claim,
        )

        with open(
            f"evaluate_factchecker/{FACTCHECKER}/{DATASET_NAME}/results.jsonl",
            "a",
        ) as f:
            f.write(
                json.dumps(
                    {
                        claim: {
                            "response": response,
                            "factchecker": FACTCHECKER,
                            "dataset": DATASET_NAME,
                        }
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
