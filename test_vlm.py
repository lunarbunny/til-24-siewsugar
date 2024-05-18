import base64
import json
import sys
import time
from typing import Dict, List
import pandas as pd
import requests
from tqdm import tqdm
from pathlib import Path
from scoring.vlm_eval import vlm_eval
from dotenv import load_dotenv
import os

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
    truths = []
    counter = 0

    with open(input_dir / "vlm.jsonl", "r") as f:
        for idx, line in enumerate(tqdm(f, desc="Reading vlm.jsonl")):
            if idx >= line_limit:
                break
            if line.strip() == "":
                continue
            instance = json.loads(line.strip())
            with open(input_dir / "images" / instance["image"], "rb") as file:
                image_bytes = file.read()
                for annotation in instance["annotations"]:
                    instances.append(
                        {
                            "key": counter,
                            "caption": annotation["caption"],
                            "b64": base64.b64encode(image_bytes).decode("ascii"),
                        }
                    )
                    truths.append(
                        {
                            "key": counter,
                            "caption": annotation["caption"],
                            "bbox": annotation["bbox"],
                        }
                    )
                    counter += 1

    assert len(truths) == len(instances)

    # Wait for the server to start
    counter = 0
    retries = 100
    while counter < retries:
        counter += 1
        print(f"> Checking server health... ({counter}/{retries} attempts)", end="\r")
        try:
            res = requests.get("http://localhost:5004/health")
            if res.ok:
                print("\n✓ Server is ready!")
                break
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(2)

    results = run_batched(instances)
    df = pd.DataFrame(results).merge(pd.DataFrame(truths), on="key")
    df.rename(columns={"bbox_x": "predict", "bbox_y": "truth"}, inplace=True)
    df["iou"] = df.apply(
        lambda row: vlm_eval([row["predict"]], [row["truth"]]), axis=1
    )
    df = df[["key", "predict", "truth", "iou", "caption"]]
    assert len(truths) == len(results)
    df.to_csv(results_dir / "vlm_results.csv", index=False)
    # calculate eval
    eval_result = vlm_eval(
        [truth["bbox"] for truth in truths],
        [result["bbox"] for result in results],
    )
    print(f"IoU@0.5: {eval_result}")


def run_batched(
    instances: List[Dict[str, str | int]], batch_size: int = 4
) -> List[Dict[str, str | int]]:
    # split into batches
    results = []
    for index in tqdm(range(0, len(instances), batch_size)):
        _instances = instances[index : index + batch_size]
        response = requests.post(
            "http://localhost:5004/identify",
            data=json.dumps(
                {
                    "instances": [
                        {field: _instance[field] for field in ("key", "caption", "b64")}
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
                    "bbox": _results[i],
                }
                for i in range(len(_instances))
            ]
        )
    return results


if __name__ == "__main__":
    main()
