import os
import json
from datetime import datetime
from utils.logging import get_logger

from graph_fewshot import graph

logger = get_logger()


def classify():
    # File paths.
    raw_file = "../datasets/raw/simpleqa/simpleqa.json"
    output_file = (
        f"../datasets/processed/simpleqa/simpleqa_{os.environ['MODEL_NAME']}.json"
    )

    # Read raw data.
    with open(raw_file, "r", encoding="utf-8") as json_file:
        rows = json.load(json_file)

    enriched_questions = []
    processed_ids = set()

    # If output file exists, load already processed rows.
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            enriched_questions = json.load(f)
            for row in enriched_questions:
                # Assuming each row has a unique 'id'
                processed_ids.add(row.get("id"))

    # Process rows that haven't been enriched.
    for idx, row in enumerate(rows, 1):
        if row.get("id") in processed_ids:
            continue

        question_text = row.get("question", "")
        answer_text = row.get("answer", "")
        tstamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        logger.info(
            f"[{tstamp}]: Processing question {idx}/{len(rows)}: {question_text}"
        )

        for retry in range(5):
            try:
                enriched_question = graph.invoke(
                    {"question": question_text, "answer": answer_text}
                )
                row.update(enriched_question)
                enriched_questions.append(row)
                # Save after each successful classification.
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(enriched_questions, f, ensure_ascii=False, indent=4)
                break
            except Exception as e:
                logger.warning(
                    f"[Retry {retry + 1}]: Error processing question {row.get('id', 'Unknown ID')}: {str(e)}"
                )
                if retry < 4:
                    logger.info("Retrying...")


if __name__ == "__main__":
    classify()
