"""
Contains main function to generate benchmark CSVs for LVLMs accessed via LLaVA
modules directly.
"""

import argparse
import torch
import os
import json
import signal
import sys
from tqdm import tqdm
import pandas as pd
from warnings import simplefilter
from pathlib import Path
import datasets
import base64
import io
from PIL import Image

# Import LLaVA-specific modules
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    tokenizer_image_token,
    process_images,
    get_model_name_from_path,
)

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


def query_llava_model(
    prompt, image, model, tokenizer, image_processor, temperature=0.0
):
    """
    Query the LLaVA model directly using the Python modules,
    without using a conversation template for simplicity
    """
    try:
        # Prepare the prompt for LLaVA with image token
        if model.config.mm_use_im_start_end:
            # Format with image start/end tokens if model requires it
            qs = (
                DEFAULT_IM_START_TOKEN
                + DEFAULT_IMAGE_TOKEN
                + DEFAULT_IM_END_TOKEN
                + "\n"
                + prompt
            )
        else:
            # Simple format with just the image token
            qs = DEFAULT_IMAGE_TOKEN + "\n" + prompt

        # print(f"Formatted prompt: {qs}")

        # Set up conversation format
        conv = conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # print(f"Final conversation prompt: {prompt}")

        # Directly tokenize the prompt without conversation template
        input_ids = (
            tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )

        # print(f"Input length: {input_ids.shape}")

        # Process the image
        image_tensor = process_images([image], image_processor, model.config)[0]

        # Generate response
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[image.size],
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                max_new_tokens=1024,
                use_cache=True,
            )

        # print(f"Output shape: {output_ids.shape}")

        # Decode the output preserving all text
        # full_output = tokenizer.decode(output_ids[0], skip_special_tokens=False)
        # print(f"Raw full output with special tokens: {repr(full_output)}")

        # Try normal decoding
        standard_output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
            0
        ].strip()
        print(f"Standard decoded output: {repr(standard_output)}")

        return standard_output

    except Exception as e:
        print(f"Error in query_llava_model: {e}")
        raise


def main():
    """main function to manage the experiments"""
    prompt_templates = get_prompt_templates([1, 5])  # 1,2,3,4,5,6,7
    print(f"INFO: Testing {len(prompt_templates)} prompts.")

    if len(sys.argv) == 1:
        print("Using default parameters.")
        args = {
            "dir": "/data",
            "model_name": "custom-llava",
            "eval_dataset": "SCAM",
            "model_path": "/llava-checkpoints/llava-v1.5-7b-openai-clip-vit-large-patch14-336-bs16-bf16-zero3-12.8",
            "model_base": None,
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

    # Configure HuggingFace datasets
    hf_path = os.path.join(dir, "hf_datasets")
    os.makedirs(hf_path, exist_ok=True)
    os.environ["HF_HOME"] = hf_path
    datasets.config.DOWNLOADED_DATASETS_PATH = Path(hf_path)
    os.environ["HF_CACHE"] = hf_path
    print(f"Setting HuggingFace datasets path to: {hf_path}")

    # Load the LLaVA model
    print("Loading LLaVA model...")
    disable_torch_init()
    model_path = args["model_path"]
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name
    )
    print(f"Model loaded: {model_name}")

    # Set up evaluation parameters
    eval_dataset = args["eval_dataset"]
    model_name = args["model_name"]
    preprocess = "pil"  # Use PIL images directly instead of base64
    overwrite = True

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

            wait_sec = 120  # Timeout for model inference
            if count > 0:
                signal.alarm(wait_sec)
            else:
                signal.alarm(0)  # No timeout for first inference

            try:
                print("Getting model response...")

                # Query LLaVA model directly
                answer = query_llava_model(
                    prompt,
                    item["image_preprocessed"],
                    model,
                    tokenizer,
                    image_processor,
                )

                signal.alarm(0)  # Cancel alarm if finished in time
            except (TimeoutError, torch.cuda.OutOfMemoryError) as e:
                print(
                    f"Response took longer than {wait_sec} seconds or ran out of memory... retrying..."
                )

                # Try one more time with a longer timeout
                try:
                    signal.alarm(wait_sec * 2)
                    answer = query_llava_model(
                        prompt,
                        item["image_preprocessed"],
                        model,
                        tokenizer,
                        image_processor,
                    )
                    signal.alarm(0)
                except Exception as e2:
                    print(f"Second attempt also failed: {e2}")
                    answer = f"ERROR: {str(e2)}"
                    signal.alarm(0)
            except Exception as e:
                print(f"Error occurred: {e}")
                answer = f"ERROR: {str(e)}"
                signal.alarm(0)

            print(f"Received response. Answer is '{answer}'")

            exact_model_string = model_name

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
                }
            )

            count += 1
            if count % 10 == 0:  # Save more frequently
                pd.DataFrame(results).to_csv(output_csv, index=False)
                print(f"Temporary results saved to {output_csv}")

    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"Final results saved to {output_csv}")


def parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        "--d",
        default="data",
        type=str,
        help="Data directory (results are also stored in there)",
    )
    parser.add_argument(
        "--model_name",
        "--m",
        default="llava-7b-reprod",
        type=str,
        help="Model name for result labeling",
    )
    parser.add_argument(
        "--eval_dataset",
        "--ed",
        default="SCAM",
        type=str,
        help="Name of dataset to evaluate (e.g., 'SCAM')",
    )
    parser.add_argument(
        "--model_path",
        default="/llava-checkpoints/llava-v1.5-7b-openai-clip-vit-large-patch14-336-bs16-bf16-zero3-12.8",
        type=str,
        help="Path to LLaVA model checkpoint",
    )

    args = parser.parse_args()

    return {
        "dir": args.dir,
        "eval_dataset": args.eval_dataset,
        "model_name": args.model_name,
        "model_path": args.model_path,
    }


if __name__ == "__main__":
    main()
