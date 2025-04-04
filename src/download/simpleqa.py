import os
import json
import pandas as pd


def download() -> None:
    """
    Download the SimpleQA dataset, convert it from CSV to JSON using pandas,
    add an "id" key at the start of each entry, and save as JSON.
    """
    name = "SimpleQA"
    url = (
        "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"
    )
    json_path = "../../datasets/raw/simpleqa/simpleqa.json"

    if os.path.exists(json_path):
        print(f"Dataset {name} already exists at {json_path}, skipping download.")
        return

    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    try:
        print(f"Downloading dataset from {url}...")
        # Read CSV directly from URL using pandas
        df = pd.read_csv(url)

        # Add an "id" column as string representation of the row index and insert it at the beginning
        df.insert(0, "id", df.index.astype(str))

        # Convert the DataFrame to a list of dictionaries
        data = df.to_dict(orient="records")

        # Save the JSON file with indentation for readability
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        print(f"Saved JSON dataset with IDs to {json_path}")
    except Exception as e:
        print(f"Failed to download or convert dataset from {url}: {e}")


if __name__ == "__main__":
    download()
