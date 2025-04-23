"""
Contains main function to generate benchmark CSVs for LVLMs accessed via OpenAI API.

NOTE: Your .env file should contain the OpenAI API key as OPENAI_API_KEY=<your_key>.
"""

import os
from warnings import simplefilter
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from openai import OpenAI
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
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)


def process_item(item, model_name, prompt_templates):
    results = []  # Accumulate results for all prompts
    image = item["image_preprocessed"]
    object_label = item["object_label"]
    attack_word = item["attack_word"]

    for template in prompt_templates:
        # Set up prompt
        prompt, object_label_first = create_prompt(
            object_label,
            attack_word,
            prefix=template["prefix"],
            suffix=template["suffix"],
        )

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image}",
                                "detail": "low",  # since SCAM images are resized to 512x512 anyway
                            },
                        },
                    ],
                }
            ],
        )

        answer = response.choices[0].message.content
        processed_answer = process_answer(
            object_label, attack_word, answer, object_label_first
        )

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
                "exact_model_string": response.model,
                "total_tokens_used": response.usage.total_tokens,
            }
        )
    return results


def main():
    """main function to manage the experiments"""
    # Set max workers that work with your OpenAI API key
    max_workers = 1

    prompt_templates = get_prompt_templates([1, 2, 3, 4, 5, 6, 7])
    print(f"INFO: Testing {len(prompt_templates)} prompts.")

    dir = "data"

    # Configure HuggingFace datasets to use a PVC-accessible location based on the dir argument
    hf_path = os.path.join(dir, "hf_datasets")
    os.makedirs(hf_path, exist_ok=True)
    os.environ["HF_HOME"] = hf_path
    datasets.config.DOWNLOADED_DATASETS_PATH = Path(hf_path)
    os.environ["hfS_CACHE"] = hf_path
    print(f"Setting HuggingFace datasets path to: {hf_path}")

    model_names = ["gpt-4o", "gpt-4o-mini"]
    preprocess = "base64"
    eval_datasets = [
        "SCAM",
        "SynthSCAM",
        "NoSCAM",
        "RTA100",
        "PAINT",
    ]

    # res_dir = "model_evals_temp"
    res_dir = "model_evals"
    overwrite = True

    # Set up folder for saving results
    res_dir = os.path.join(dir, res_dir)
    os.makedirs(res_dir, exist_ok=True)

    for eval_dataset in eval_datasets:
        dataset = get_dataset(dir, eval_dataset, preprocess)

        for model_name in model_names:
            output_name = f"{eval_dataset}--{model_name}.csv"
            if not overwrite and os.path.exists(os.path.join(res_dir, output_name)):
                print(f"File {output_name} already exists. Skipping.")
                return

            results = []
            futures = []

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for item in dataset:
                    # Submit each API call as a separate thread
                    futures.append(
                        executor.submit(
                            process_item, item, model_name, prompt_templates
                        )
                    )
                try:
                    for future in tqdm(
                        futures, desc="Processing images", total=len(futures)
                    ):
                        # print(future)
                        # print(future.result())
                        results.extend(future.result())
                except Exception as e:
                    print(f"An error occurred: {e}. Saving results collected so far.")
                    # Optionally cancel remaining tasks:
                    for future in futures:
                        future.cancel()

            output_csv = os.path.join(res_dir, output_name)
            pd.DataFrame(results).to_csv(output_csv, index=False)
            print(f"Results saved to {output_csv}")


if __name__ == "__main__":
    main()
