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

args = parser.parse_args()
if args.dataset:
    DATASET_PATH = args.dataset
else:
    # Return an error if no dataset is provided
    raise ValueError("Please provide a dataset path using --dataset argument.")

if __name__ == "__main__":
    CONFIG_PATH = "config.json"
    PROCESSED_PATH = "../../datasets/processed/claims"
    DATASET_NAME = args.dataset
    DATASET_PATH = f"{PROCESSED_PATH}/{DATASET_NAME}/data_gpt-4o_annotated.json"

    # Set the environment variable for cost
    os.environ["SAVE_SERPER_COST"] = "True"
    os.environ["SERPER_COST_PATH"] = (
        f"evaluate_factchecker/{DATASET_NAME}/serper_cost.jsonl"
    )
    os.environ["SAVE_MODEL_COST"] = "True"
    os.environ["MODEL_COST_PATH"] = (
        f"evaluate_factchecker/{DATASET_NAME}/model_cost.jsonl"
    )

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    config = OpenFactCheckConfig(CONFIG_PATH)
    config.output_path = f"evaluate_factchecker/{DATASET_NAME}"
    ofc = OpenFactCheck(config).ResponseEvaluator

    for row in dataset:
        claim = row["claim_urdu"]

        response = ofc.evaluate(
            response=claim,
        )

        print("Overall Result: ", response)
        break
