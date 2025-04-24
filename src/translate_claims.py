import os
import json
from datetime import datetime
from utils.logging import get_logger

from graph_fewshot import graph

logger = get_logger()


def classify():
    # File paths.
    raw_file = "../datasets/raw/claims/factcheckbench/data.jsonl"
    output_file = f"../datasets/processed/claims/factcheckbench/data_{os.environ['MODEL_NAME']}.jsonl"

    # Read raw data.
    with open(raw_file, "r", encoding="utf-8") as json_file:
        rows = [json.loads(line) for line in json_file]

    enriched_claims = []
    # processed_ids = set()

    # # If output file exists, load already processed rows.
    # if os.path.exists(output_file):
    #     with open(output_file, "r", encoding="utf-8") as f:
    #         enriched_claims = json.load(f)
    #         for row in enriched_claims:
    #             # Assuming each row has a unique 'id'
    #             processed_ids.add(row.get("id"))

    # Process rows that haven't been enriched.
    for idx, row in enumerate(rows, 1):
        # if row.get("id") in processed_ids:
        #     continue

        claim_text = row.get("claim", "")
        label_text = row.get("label", "")
        tstamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        logger.info(f"[{tstamp}]: Processing claim {idx}/{len(rows)}: {claim_text}")

        for retry in range(5):
            try:
                enriched_claim = graph.invoke(
                    {"claim": claim_text, "label": label_text}
                )
                row.update(enriched_claim)
                enriched_claims.append(row)
                # Save after each successful classification.
                with open(output_file, "w", encoding="utf-8") as f:
                    for record in enriched_claims:
                        f.write(json.dumps(record, ensure_ascii=False, indent=4) + "\n")
                break
            except Exception as e:
                logger.warning(
                    f"[Retry {retry + 1}]: Error processing question {row.get('claim', 'Unknown ID')}: {str(e)}"
                )
                if retry < 4:
                    logger.info("Retrying...")


if __name__ == "__main__":
    classify()
