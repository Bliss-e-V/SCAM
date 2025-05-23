"""
Contains main function to generate benchmark CSVs for LVLMs accessed via ollama.
"""

import argparse
import signal
import sys
import os
import time
import httpx
import subprocess
from tqdm import tqdm
import pandas as pd
from warnings import simplefilter
from ollama import chat
from pathlib import Path
import datasets

from utils import (
    get_dataset,
    get_prompt_templates,
    create_prompt,
    process_answer,
)

simplefilter(action="ignore", category=DeprecationWarning)


def timeout_handler(signum, frame):
    raise TimeoutError


signal.signal(signal.SIGALRM, timeout_handler)


def start_ollama():
    print("Starting Ollama...")
    process = subprocess.Popen(
        ["ollama", "start"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    # Wait longer for Ollama to start and be ready
    for i in range(30):  # 30 seconds with status updates
        if i % 5 == 0:  # Show message every 5 seconds
            print(f"Waiting for Ollama to start... ({i}/30 seconds)")
        time.sleep(1)

        # Check if Ollama is responding
        try:
            # Simple ping to check if the server is ready
            subprocess.check_output(
                ["ollama", "list"], stderr=subprocess.STDOUT, text=True
            )
            print("Ollama is ready!")
            return True
        except subprocess.CalledProcessError:
            # Ollama is not ready yet
            pass

    print("Warning: Ollama might not be fully started yet")
    return False


def kill_first_process(match_str):
    # Get process list
    output = subprocess.check_output(["ps", "aux"], text=True)
    # Filter matching lines, excluding the grep process
    lines = [
        line
        for line in output.splitlines()
        if match_str in line and "grep" not in line
    ]
    if lines:
        first_line = lines[0]
        # PID is the second column
        pid = first_line.split()[1]
        subprocess.run(["kill", "-9", pid])
        print(f"Killed process {pid} for '{match_str}'.")
    else:
        print(f"No process found matching '{match_str}'.")


def main():
    """main function to manage the experiments"""
    prompt_templates = get_prompt_templates([1, 5])  # 1,2,3,4,5,6,7
    print(f"INFO: Testing {len(prompt_templates)} prompts.")

    if len(sys.argv) == 1:
        print("Using default parameters.")
        args = {
            "dir": "data",
            "model_name": "llava:34b",  # gemma3:4b, gemma3:12b, gemma3:27b, llava:7b, llava:13b, llava:34b, llava-llama3, llama3.2-vision:90b
            "eval_dataset": "SCAM",  # SynthSCAM, NoSCAM, RTA100, PAINT
            "sleep_for_ollama_pull": False,
        }
        # Using temp results dir for testing
        res_dir = "model_evals_temp"
    else:
        # Read arguments
        args = parse_input()
        res_dir = "model_evals"

    print("\n #### Arguments set: #####\n")
    for key, value in args.items():
        print(f"\t{key}={value}")
    print("\n #########################\n")

    dir = args["dir"]

    # Configure HuggingFace datasets to use a PVC-accessible location based on the dir argument
    hf_path = os.path.join(dir, "hf_datasets")
    os.makedirs(hf_path, exist_ok=True)
    os.environ["HF_HOME"] = hf_path
    datasets.config.DOWNLOADED_DATASETS_PATH = Path(hf_path)
    os.environ["hfS_CACHE"] = hf_path
    print(f"Setting HuggingFace datasets path to: {hf_path}")

    model_name = args["model_name"]
    preprocess = "base64"
    eval_dataset = args["eval_dataset"]
    overwrite = True

    # Start Ollama service in the background
    start_ollama()

    # If this is the first time running this model, it will need to be pulled.:
    if args["sleep_for_ollama_pull"]:
        time.sleep(600)
        # Now, connect to the maschine (e.g., kubernetes pod) and `ollama pull`
        #  the model if not done already.

    # Set up folder for saving results
    res_dir = os.path.join(dir, res_dir)
    os.makedirs(res_dir, exist_ok=True)

    output_name = f"{eval_dataset}--{model_name}.csv"
    # Replace ":" in model_name by "-" (MacOS...)
    output_name = output_name.replace(":", "-")
    if not overwrite and os.path.exists(os.path.join(res_dir, output_name)):
        print(f"File {output_name} already exists. Skipping.")
        return
    results = []
    output_csv = os.path.join(res_dir, output_name)

    dataset = get_dataset(dir, eval_dataset, preprocess)

    print("Processing images...")
    count = 0
    for item in tqdm(dataset, desc="Processing images"):
        object_label = item["object_label"]
        attack_word = item["attack_word"]
        print(f"Processing item ({object_label}, {attack_word})")

        for template in prompt_templates:

            # Set up prompt
            prompt, object_label_first = create_prompt(
                object_label,
                attack_word,
                prefix=template["prefix"],
                suffix=template["suffix"],
            )

            wait_sec = 30
            if count > 0:
                signal.alarm(wait_sec)  # Set timeout to wait_sec seconds
            else:
                signal.alarm(0)  # No timeout in this case, because model needs
                #  time to set up and load into GPU

            try:
                print("Getting model response...")

                response = chat(
                    model=model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                            "images": [item["image_preprocessed"]],
                        },
                    ],
                )
                signal.alarm(0)  # Cancel alarm if finished in time
            except (TimeoutError, httpx.ReadTimeout):
                print(
                    "Response took longer than wait_sec seconds... killing processes..."
                )
                kill_first_process("ollama_models")
                kill_first_process("ollama start")

                # Start ollama again with improved waiting
                ollama_ready = start_ollama()

                # Try multiple times to connect to Ollama
                max_retries = 3
                for retry in range(max_retries):
                    try:
                        print(
                            f"Attempt {retry+1}/{max_retries} to get model response..."
                        )
                        response = chat(
                            model=model_name,
                            messages=[
                                {
                                    "role": "user",
                                    "content": prompt,
                                    "images": [item["image_preprocessed"]],
                                },
                            ],
                        )
                        print("Successfully received response.")
                        break
                    except ConnectionError as ce:
                        print(f"Connection error: {ce}")
                        if retry < max_retries - 1:
                            print(f"Waiting 10 more seconds before retry...")
                            time.sleep(10)
                        else:
                            print(
                                "All retries failed. Saving partial results and exiting."
                            )
                            pd.DataFrame(results).to_csv(output_csv, index=False)
                            sys.exit(1)
            print("Received response.")
            time.sleep(0.75)  # to avoid too fast requests

            exact_model_string = model_name
            # TODO: print response to check if there is also a thinking part
            print("response:", response)
            print("response message:", response.message)

            answer = response.message.content

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
                    "exact_model_string": exact_model_string,
                    # "total_tokens_used": total_tokens_used, # ?
                }
            )
            count += 1
            if count % 100 == 0:
                pd.DataFrame(results).to_csv(output_csv, index=False)
                print(f"Temporary results saved to {output_csv}")
        # break

    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"Final results saved to {output_csv}")


def parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--d",
        "--dir",
        default="data",
        type=str,
        help="Data directory (results are also stored in there)",
    )
    parser.add_argument(
        "--m",
        "--model_name",
        default="",
        type=str,
        help="Model name (e.g., 'llava:34b')",
    )
    parser.add_argument(
        "--ed",
        "--eval_dataset",
        default="SCAM",
        type=str,
        help="Name of dataset to evaluate  (e.g., 'SCAM')",
    )
    # sleep_for_ollama_pull
    parser.add_argument(
        "--s",
        "--sleep_for_ollama_pull",
        default=False,
        type=lambda x: x.lower() == "true",
        help="Sleep for 10 minutes to allow for manual model pull from Ollama pod.",
    )

    args = parser.parse_args()

    return {
        "dir": args.d,
        "model_name": args.m,
        "eval_dataset": args.ed,
        "sleep_for_ollama_pull": args.s,
    }


if __name__ == "__main__":
    main()
