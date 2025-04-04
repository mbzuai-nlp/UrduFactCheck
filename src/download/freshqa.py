import os
import re
import requests
import pandas as pd
from datetime import datetime


def download():
    # Check if the raw file already exists.
    if os.path.exists("../../datasets/raw/freshqa/freshqa.json"):
        print("Raw file already exists. Skipping download.")
        return

    # Fetch the README file containing the links.
    URL = "https://raw.githubusercontent.com/freshllms/freshqa/main/README.md"
    response = requests.get(URL)

    regex = (
        r"\[FreshQA\s+(?P<date>(?P<month>[A-Za-z]+)\s+(?P<day>\d{1,2}),\s+(?P<year>\d{4}))\]\("
        r"(?P<url>https:\/\/docs\.google\.com\/spreadsheets\/d\/(?P<id>[A-Za-z0-9_-]+)\/edit\?usp=sharing)\)"
    )

    def parse_date(date_str):
        for fmt in ("%B %d, %Y", "%b %d, %Y"):
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        raise ValueError(f"Date {date_str!r} does not match expected formats")

    matches = re.finditer(regex, response.text)

    entries = []
    for match in matches:
        date_str = match.group("date")
        parsed_date = parse_date(date_str)
        spreadsheet_id = match.group("id")
        url = match.group("url")
        entries.append((parsed_date, url, spreadsheet_id))

    if entries:
        latest_date, latest_url, latest_id = max(entries, key=lambda x: x[0])
        print(f"Latest date: {latest_date.strftime('%B %d, %Y')}")
        print(f"Latest sheet URL: {latest_url}")

        # Construct the CSV export URL.
        csv_url = (
            f"https://docs.google.com/spreadsheets/d/{latest_id}/export?format=csv"
        )
        print(f"CSV export URL: {csv_url}")

        # # Download the CSV file.
        # csv_response = requests.get(csv_url)
        # csv_response.raise_for_status()  # Raise an error if the request failed.

        # Read CSV content directly into pandas DataFrame
        df = pd.read_csv(csv_url, header=1)

        # Remove the first two rows; the next row will serve as the header.
        df = df.iloc[2:]

        # Reset the index after removing the rows
        df.reset_index(drop=True, inplace=True)

        # Save the DataFrame to JSON
        json_path = "../../datasets/raw/freshqa/freshqa.json"
        df.to_json(
            json_path, orient="records", lines=False, force_ascii=False, indent=4
        )

        print(f"Downloaded and converted CSV to JSON as '{json_path}'")
    else:
        print("No entries found.")


if __name__ == "__main__":
    download()
