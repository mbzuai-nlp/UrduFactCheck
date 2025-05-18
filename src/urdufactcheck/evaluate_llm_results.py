import os
import json
import pandas as pd
from sklearn.metrics import classification_report


# Calculate Model Cost
def calculate_model_cost(
    num_input_tokens,
    num_output_tokens,
    input_cost_per_1M_tokens,
    output_cost_per_1M_tokens,
):
    input_cost = (num_input_tokens / 1_000_000) * input_cost_per_1M_tokens
    output_cost = (num_output_tokens / 1_000_000) * output_cost_per_1M_tokens
    total_cost = input_cost + output_cost
    return round(total_cost, 2)


# Calculate Serper Cost
def calculate_serper_cost(num_credits, cost_per_credit):
    return round(num_credits * cost_per_credit, 2)


costs_by_model = {
    "gpt-4o": {
        "input_cost_per_1M_tokens": 2.5,  # $2.5 for 1M tokens
        "output_cost_per_1M_tokens": 10,  # $10 for 1M tokens
    },
    "gpt-4o-mini": {
        "input_cost_per_1M_tokens": 0.15,  # $0.15 for 1M tokens
        "output_cost_per_1M_tokens": 0.60,  # $0.60 for 1M tokens
    },
    "gpt-4.1": {
        "input_cost_per_1M_tokens": 2,  # $2.5 for 1M tokens
        "output_cost_per_1M_tokens": 8,  # $10 for 1M tokens
    },
    "gpt-4.1-mini": {
        "input_cost_per_1M_tokens": 0.4,  # $0.4 for 1M tokens
        "output_cost_per_1M_tokens": 1.6,  # $1.6 for 1M tokens
    },
}

serper_cost_per_credit = 52.5 / 50000  # $52.5 for 50k credits

DATASET_PATH = "../../datasets/processed/claims"
OUTPUT_PATH = "evaluate_factchecker"
RESULTS_FILE = "results.jsonl"
MODEL_COST_FILE = "model_cost.jsonl"
SERPER_COST_FILE = "serper_cost.jsonl"

# Gather all dataset names and output subdirs
datasets = [
    d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))
]
output_dirs = [
    d for d in os.listdir(OUTPUT_PATH) if os.path.isdir(os.path.join(OUTPUT_PATH, d))
]

# Prepare containers
evaluate_factchecker_results = {}

for dataset in datasets:
    for output_dir in output_dirs:
        # e.g. "rarr_gpt4o" → factchecker_name="rarr", model_name="gpt4o"
        factchecker_name, model_name = output_dir.split("_", 1)

        # Ensure nested dicts exist
        evaluate_factchecker_results.setdefault(factchecker_name, {}).setdefault(
            dataset, {}
        ).setdefault(model_name, {})

        # Build file paths
        base_dir = os.path.join(OUTPUT_PATH, output_dir, dataset)
        results_path = os.path.join(base_dir, RESULTS_FILE)
        model_cost_path = os.path.join(base_dir, MODEL_COST_FILE)
        serper_cost_path = os.path.join(base_dir, SERPER_COST_FILE)

        # ----- Load evaluation results -----
        if os.path.exists(results_path):
            # Load the dataset
            gold_path = os.path.join(
                DATASET_PATH, dataset, "data_gpt-4o_annotated.json"
            )
            with open(gold_path, "r", encoding="utf-8") as gf:
                gold_records = json.load(gf)

            # Load the results as json
            evaluated_results = {}
            with open(results_path, "r") as f:
                results = [json.loads(line) for line in f]

            # Convert the results to a dictionary
            for result in results:
                for claim, data in result.items():
                    evaluated_results[claim] = data

            gold_labels = []
            predicted_labels = []
            for data in gold_records:
                # Get the claim
                claim = data["claim_urdu"]

                # Get the gold label
                if data["label"] == "false":
                    gold_label = False
                elif data["label"] == "true":
                    gold_label = True

                # Get the claim from the results
                result = evaluated_results.get(claim, None)
                if result is None:
                    continue

                # Get the predicted label
                if result["response"] == "False":
                    predicted_label = False
                elif result["response"] == "True":
                    predicted_label = True

                gold_labels.append(gold_label)
                predicted_labels.append(result["response"])

            # Compute classification report
            try:
                report = classification_report(
                    gold_labels,
                    predicted_labels,
                    output_dict=True,
                    zero_division=0,
                )
                evaluate_factchecker_results[factchecker_name][dataset][model_name][
                    "classification_report"
                ] = report
            except ValueError as e:
                print(
                    f"[Warning] Error computing classification report for {factchecker_name} on {dataset} with model {model_name}: {e}"
                )
                evaluate_factchecker_results[factchecker_name][dataset][model_name][
                    "classification_report"
                ] = None

        # ----- Compute model API cost -----
        if os.path.exists(model_cost_path):
            df_cost = pd.read_json(model_cost_path, lines=True)
            if model_name in costs_by_model:
                # Check if all df_cost["model"] are the same
                if not df_cost["model"].eq(model_name).all():
                    print(
                        f"[Warning] Model name mismatch in {model_cost_path}: expected {model_name}, found {df_cost['model_name'].unique()}"
                    )
                params = costs_by_model[model_name]
                cost = calculate_model_cost(
                    num_input_tokens=df_cost["prompt_tokens"].sum(),
                    num_output_tokens=df_cost["completion_tokens"].sum(),
                    input_cost_per_1M_tokens=params["input_cost_per_1M_tokens"],
                    output_cost_per_1M_tokens=params["output_cost_per_1M_tokens"],
                )
                evaluate_factchecker_results[factchecker_name][dataset][model_name][
                    "model_cost"
                ] = cost
            else:
                print(f"[Warning] No cost parameters for model '{model_name}'")

        # ----- Compute Serper (search) cost -----
        if os.path.exists(serper_cost_path):
            df_serp = pd.read_json(serper_cost_path, lines=True)
            cost = calculate_serper_cost(
                num_credits=df_serp["google_serper_credits"].sum(),
                cost_per_credit=serper_cost_per_credit,
            )
            evaluate_factchecker_results[factchecker_name][dataset][model_name][
                "serper_cost"
            ] = cost

with open("evaluate_factchecker_results.json", "w") as f:
    json.dump(evaluate_factchecker_results, f, indent=4)

# Convert the nested dict into a flat DataFrame
df_results = pd.DataFrame.from_dict(
    {
        (factchecker, dataset, model): values
        for factchecker, datasets in evaluate_factchecker_results.items()
        for dataset, models in datasets.items()
        for model, values in models.items()
    },
    orient="index",
)
df_results.reset_index(inplace=True)
df_results.columns = ["factchecker", "dataset", "model"] + list(df_results.columns[3:])

# Flatten the `classification_report` dict into its own columns
report_df = pd.json_normalize(df_results["classification_report"])
report_df.columns = report_df.columns.str.replace(
    r"[.\s-]", "_", regex=True
).str.replace("__", "_", regex=False)

# Drop the original dict‐column and concatenate the flattened metrics
df_results = pd.concat(
    [df_results.drop(columns=["classification_report"]), report_df], axis=1
)

# Keep only the macro‐avg metrics (plus identifiers)
macro_cols = [c for c in df_results.columns if c.startswith("macro_avg")]
df_macro = df_results[["factchecker", "dataset", "model"] + macro_cols]

# Rename for brevity
df_macro = df_macro.rename(
    columns={
        "macro_avg_precision": "precision",
        "macro_avg_recall": "recall",
        "macro_avg_f1_score": "f1_score",
        "macro_avg_support": "support",
    }
)

# Drop the `support` column
df_macro = df_macro.drop(columns=["support"])

# Round the numeric columns to 2 decimal places
df_macro[["precision", "recall", "f1_score"]] = df_macro[
    ["precision", "recall", "f1_score"]
].round(2)

# Add the `model_cost` and `serper_cost` columns
df_macro["model_cost"] = df_results["model_cost"]
df_macro["serper_cost"] = df_results["serper_cost"]

# Add the `total_cost` column
df_macro["total_cost"] = df_macro["model_cost"] + df_macro["serper_cost"]

# Save to CSV
df_macro.to_csv("evaluate_factchecker_results.csv", index=False)
