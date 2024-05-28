import base64
import json
import time
from typing import Dict, List
import pandas as pd
import requests
from tqdm import tqdm
from pathlib import Path
from scoring.asr_eval import asr_eval
from dotenv import load_dotenv
import os
import sys

load_dotenv()

TEAM_NAME = os.getenv("TEAM_NAME")
TEAM_TRACK = os.getenv("TEAM_TRACK")


def main():
    line_limit = sys.maxsize
    if len(sys.argv) > 1:
        # First argument is the line limit
        arg1 = sys.argv[1]
        if arg1.isnumeric():
            line_limit = int(arg1)
            print(f"Limiting to {line_limit} test cases.")

    on_gcp = None not in [TEAM_NAME, TEAM_TRACK]

    if on_gcp:
        # For running on GCP
        input_dir = Path(f"/home/jupyter/{TEAM_TRACK}")
        results_dir = Path(f"/home/jupyter/{TEAM_NAME}")
    else:
        # For running locally
        input_dir = Path("advanced")
        results_dir = Path("results")
    
    results_dir.mkdir(parents=True, exist_ok=True)
    instances = []

    with open(input_dir / "asr.jsonl", "r") as f:
        for idx, line in enumerate(f):
            if idx >= line_limit:
                break
            if line.strip() == "":
                continue
            instance = json.loads(line.strip())
            with open(input_dir / "audio" / instance["audio"], "rb") as file:
                audio_bytes = file.read()
                instances.append(
                    {**instance, "b64": base64.b64encode(audio_bytes).decode("ascii")}
                )

    # Wait for the server to start
    counter = 0
    retries = 100
    while counter < retries:
        counter += 1
        print(f"> Checking server health... ({counter}/{retries} attempts)", end="\r")
        try:
            res = requests.get("http://localhost:5001/health")
            if res.ok:
                print("\n✓ Server is ready!")
                break
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(2)

    results = run_batched(instances)
    df = pd.DataFrame(results)
    # calculate eval
    eval_result = asr_eval(
        [result["transcript"] for result in results],
        [result["prediction"] for result in results],
    )
    df.to_csv(results_dir / f"asr_whisper-small.en_cXXX_{str(eval_result)[:8]}.csv", index=False)
    print(f"1-WER: {eval_result}")


def run_batched(
    instances: List[Dict[str, str | int]], batch_size: int = 4
) -> List[Dict[str, str | int]]:
    # split into batches
    results = []
    for index in tqdm(range(0, len(instances), batch_size)):
        _instances = instances[index : index + batch_size]
        response = requests.post(
            "http://localhost:5001/stt",
            data=json.dumps(
                {
                    "instances": [
                        {"key": _instance["key"], "b64": _instance["b64"]}
                        for _instance in _instances
                    ]
                }
            ),
        )
        _results = response.json()["predictions"]
        results.extend(
            [
                {
                    "key": _instances[i]["key"],
                    "transcript": _instances[i]["transcript"],
                    "prediction": _results[i],
                }
                for i in range(len(_instances))
            ]
        )
    return results


if __name__ == "__main__":
    main()
