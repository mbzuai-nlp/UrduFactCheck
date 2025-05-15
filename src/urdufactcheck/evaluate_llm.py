import os
import argparse
import json
from openfactcheck import OpenFactCheck, OpenFactCheckConfig

# Set up argument parser
parser = argparse.ArgumentParser(
    description="Evaluate fact-checking generated_answer_urdus."
)
parser.add_argument(
    "--dataset",
    type=str,
    help="Path to the dataset file.",
)
parser.add_argument(
    "--model",
    type=str,
    help="Name of the model to use.",
)

args = parser.parse_args()
if not args.dataset:
    raise ValueError("Please provide a dataset path using --dataset argument.")
if not args.model:
    raise ValueError("Please provide a model name using --model argument.")

if __name__ == "__main__":
    CONFIG_PATH = "config.json"
    EVALUATION_PATH = "../../datasets/evaluation/qa"
    DATASET_NAME = args.dataset
    MODEL_NAME = args.model
    DATASET_PATH = f"{EVALUATION_PATH}/{DATASET_NAME}/{MODEL_NAME}.json"

    # Set the environment variable for cost
    os.environ["SAVE_SERPER_COST"] = "True"
    os.environ["SERPER_COST_PATH"] = (
        f"evaluate_llm/{DATASET_NAME}/{MODEL_NAME}/serper_cost.jsonl"
    )
    os.environ["SAVE_MODEL_COST"] = "True"
    os.environ["MODEL_COST_PATH"] = (
        f"evaluate_llm/{DATASET_NAME}/{MODEL_NAME}/model_cost.jsonl"
    )

    # Create the output directory if it doesn't exist
    os.makedirs(f"evaluate_llm/{DATASET_NAME}/{MODEL_NAME}", exist_ok=True)

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    config = OpenFactCheckConfig(CONFIG_PATH)
    config.output_path = f"evaluate_llm/{DATASET_NAME}/{MODEL_NAME}"
    ofc = OpenFactCheck(config).ResponseEvaluator

    # Check if the results file already exists
    existing_results = {}
    cleaned_results = {}
    if os.path.exists(f"evaluate_llm/{DATASET_NAME}/{MODEL_NAME}/results.jsonl"):
        with open(
            f"evaluate_llm/{DATASET_NAME}/{MODEL_NAME}/results.jsonl",
            "r",
            encoding="utf-8",
        ) as f:
            results = [json.loads(line) for line in f.readlines()]

            # Convert the results to a dictionary
            for result in results:
                for generated_answer_urdu, data in result.items():
                    existing_results[generated_answer_urdu] = data

        for generated_answer_urdu, existing_result in existing_results.items():
            if isinstance(existing_result["response"], bool):
                print(f"generated_answer_urdu already exists: {generated_answer_urdu}")
                cleaned_results[generated_answer_urdu] = existing_result

        # Rewrite the cleaned results to the file
        with open(
            f"evaluate_llm/{DATASET_NAME}/{MODEL_NAME}/results.jsonl",
            "w",
            encoding="utf-8",
        ) as f:
            for generated_answer_urdu, data in cleaned_results.items():
                f.write(
                    json.dumps(
                        {
                            generated_answer_urdu: {
                                "response": data["response"],
                                "model": MODEL_NAME,
                                "dataset": DATASET_NAME,
                            }
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    for row in dataset:
        if row["generated_answer_urdu"] in cleaned_results:
            print(
                f"generated_answer_urdu already exists: {row['generated_answer_urdu']}"
            )
            continue

        generated_answer_urdu = row["generated_answer_urdu"]
        question_urdu = row["question_urdu"]

        response = ofc.evaluate(
            question=question_urdu,
            response=question_urdu + " " + generated_answer_urdu,
        )

        with open(
            f"evaluate_llm/{DATASET_NAME}/{MODEL_NAME}/results.jsonl",
            "a",
        ) as f:
            f.write(
                json.dumps(
                    {
                        generated_answer_urdu: {
                            "response": response,
                            "model": MODEL_NAME,
                            "dataset": DATASET_NAME,
                        }
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
