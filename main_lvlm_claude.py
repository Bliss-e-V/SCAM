"""
Contains main function to generate benchmark CSVs for LVLMs accessed via Anthropic Claude API.

NOTE: Your .env file should contain the Anthropic API key as ANTHROPIC_API_KEY=<your_key>.
"""

import os
import time
from warnings import simplefilter
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from anthropic import Anthropic
from pathlib import Path
import datasets

from utils import (
    get_dataset,
    get_prompt_templates,
    create_prompt,
    process_answer,
)

simplefilter(action="ignore", category=DeprecationWarning)
load_dotenv()
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
client = Anthropic(api_key=anthropic_api_key)


def process_item(item, model_name, prompt_templates):
    results = []  # Accumulate results for all prompts
    image = item["image_preprocessed"]
    object_label = item["object_label"]
    attack_word = item["attack_word"]

    # Infer media type (we only encounter .jpeg, .jpg, and .png here)
    image_path = item["id"]
    ext = os.path.splitext(image_path)[1].lower()
    if ext == ".png":
        media_type = "image/png"
    else:
        media_type = "image/jpeg"

    for template in prompt_templates:
        prompt, object_label_first = create_prompt(
            object_label,
            attack_word,
            prefix=template["prefix"],
            suffix=template["suffix"],
        )
        response = client.messages.create(
            model=model_name,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        time.sleep(1.2)  # Claude rate limit: 50 requests per minute
        answer = response.content[0].text if response.content else ""
        processed_answer = process_answer(object_label, attack_word, answer, object_label_first)
        results.append(
            {
                "type": item["type"],
                "id": item["id"],
                "object_label": object_label,
                "attack_word": attack_word,
                "postit_area_pct": item["postit_area_pct"],
                "prompt_id": template["id"],
                "prompt": prompt,
                "object_label_first": object_label_first,
                "answer": answer,
                "processed_answer": processed_answer,
                "exact_model_string": model_name,
                "total_tokens_used": response.usage.input_tokens + response.usage.output_tokens,
            }
        )
    return results


def main(n_samples=None):
    max_workers = 1
    prompt_templates = get_prompt_templates([1, 5])  # [1, 2, 3, 4, 5, 6, 7])
    print(f"INFO: Testing {len(prompt_templates)} prompts.")

    dir = "data"
    hf_path = os.path.join(dir, "hf_datasets")
    os.makedirs(hf_path, exist_ok=True)
    os.environ["HF_HOME"] = hf_path
    datasets.config.DOWNLOADED_DATASETS_PATH = Path(hf_path)
    os.environ["hfS_CACHE"] = hf_path
    print(f"Setting HuggingFace datasets path to: {hf_path}")

    model_names = [
        # "claude-opus-4-20250514", # Too expensive...
        "claude-sonnet-4-20250514",
    ]
    preprocess = "base64"
    eval_datasets = [
        "SCAM",
        "SynthSCAM",
        "NoSCAM",
        "RTA100",
        "PAINT",
    ]

    # res_dir = "model_evals_temp" # TEST only
    res_dir = "model_evals"
    overwrite = True

    # n_samples = 2 # TEST only

    # Set up folder for saving results
    res_dir = os.path.join(dir, res_dir)
    os.makedirs(res_dir, exist_ok=True)

    for eval_dataset in eval_datasets:
        dataset = get_dataset(dir, eval_dataset, preprocess)
        if n_samples is not None:
            subset = dataset[:n_samples]
            if isinstance(subset, dict):
                # Convert dict of lists to list of dicts
                dataset = [dict(zip(subset, t)) for t in zip(*subset.values())]
            else:
                dataset = subset

        for model_name in model_names:
            output_name = f"{eval_dataset}--{model_name}.csv"
            output_csv = os.path.join(res_dir, output_name)

            # Checkpoint/resume logic
            already_processed_ids = set()
            old_results = []
            if os.path.exists(output_csv):
                try:
                    old_df = pd.read_csv(output_csv)
                    if "id" in old_df.columns:
                        already_processed_ids = set(old_df["id"].astype(str))
                        old_results = old_df.to_dict(orient="records")
                        print(
                            f"Resuming: {len(already_processed_ids)} already processed in {output_name}."
                        )
                except Exception as e:
                    print(f"Warning: Could not load existing CSV {output_csv}: {e}")

            # Filter dataset to only missing items
            items_to_process = [
                item for item in dataset if str(item["id"]) not in already_processed_ids
            ]
            if not items_to_process:
                print(f"All items already processed for {output_name}. Skipping.")
                continue

            results = []
            futures = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for item in items_to_process:
                    futures.append(
                        executor.submit(process_item, item, model_name, prompt_templates)
                    )
                try:
                    for future in tqdm(futures, desc="Processing images", total=len(futures)):
                        results.extend(future.result())
                except Exception as e:
                    print(f"An error occurred: {e}. Saving results collected so far.")
                    for future in futures:
                        future.cancel()

            # Merge old and new results, then overwrite CSV
            all_results = old_results + results
            pd.DataFrame(all_results).to_csv(output_csv, index=False)
            print(f"Results saved to {output_csv} (total: {len(all_results)})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_samples", type=int, default=None, help="Number of samples to process for testing."
    )
    args = parser.parse_args()
    main(n_samples=args.n_samples)
