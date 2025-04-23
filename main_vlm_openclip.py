"""
Contains main function to generate benchmark CSVs for VLMs.

Run locally, for example, as:
`python3.10 main.py --eval_dataset SCAM --model_name ViT-B-32 --pretraining_data laion2b_s34b_b79k --device_name cpu --batch_size 4 --overwrite True`
"""

import argparse
import os
from warnings import simplefilter
from contextlib import nullcontext
import sys
from tqdm import tqdm
import open_clip
import torch
import pandas as pd
from torch.utils.data import DataLoader
import datasets
from pathlib import Path

from utils import get_dataset

simplefilter(action="ignore", category=DeprecationWarning)

TEMPLATES = [
    "a bad photo of a {}.",
    "a photo of many {}.",
    "a sculpture of a {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of the {}.",
    "a rendering of a {}.",
    "graffiti of a {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
    "a tattoo of a {}.",
    "the embroidered {}.",
    "a photo of a hard to see {}.",
    "a bright photo of a {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a dark photo of the {}.",
    "a drawing of a {}.",
    "a photo of my {}.",
    "the plastic {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a black and white photo of the {}.",
    "a painting of the {}.",
    "a painting of a {}.",
    "a pixelated photo of the {}.",
    "a sculpture of the {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a plastic {}.",
    "a photo of the dirty {}.",
    "a jpeg corrupted photo of a {}.",
    "a blurry photo of the {}.",
]


def main():
    """main function to manage the experiments"""

    print("Read arguments: ", sys.argv)
    if len(sys.argv) == 1:
        print("Using default parameters.")
        args = {
            "dir": "data",
            "eval_dataset": "SCAM",
            "model_name": "ViT-B-32",
            "pretraining_data": "laion2b_s34b_b79k",
            "device_name": "cpu",
            "batch_size": 4,
            "overwrite": True,
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

    eval_dataset = args["eval_dataset"]
    model_name = args["model_name"]
    pretraining_data = args["pretraining_data"]

    # Set up folder for saving results
    res_dir = os.path.join(args["dir"], res_dir)
    os.makedirs(res_dir, exist_ok=True)

    output_name = f"{eval_dataset}--{model_name}--{pretraining_data}.csv"
    # Replace ":" in model names by "-" (MacOS...)
    output_name = output_name.replace(":", "-")
    if not args["overwrite"] and os.path.exists(os.path.join(res_dir, output_name)):
        print(f"File {output_name} already exists. Skipping.")
        return

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretraining_data
    )
    model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
    tokenizer = open_clip.get_tokenizer(model_name)
    device_name = args["device_name"]
    model.to(device_name)

    dataset = get_dataset(dir, eval_dataset, preprocess)
    # Forget "image" column (and only keep "image_preprocessed") to be able to
    #  work with torch dataloader batches
    dataset = dataset.data.remove_columns("image")
    # Cast "image_preprocessed" into torch tensor
    dataset = dataset.with_format("torch")

    dataloader = DataLoader(
        dataset,
        batch_size=args["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=(device_name == "cuda"),
    )
    autocast_context = (
        torch.cuda.amp.autocast() if device_name == "cuda" else nullcontext()
    )

    results = []
    object_embedding_cache = {}
    attack_embedding_cache = {}

    for batch in tqdm(dataloader, total=len(dataloader), desc="Processing batches"):
        images = batch["image_preprocessed"].to(device_name)
        batch_size_current = images.size(0)
        with torch.no_grad(), autocast_context:
            img_features = model.encode_image(images)
            img_features /= img_features.norm(dim=-1, keepdim=True)
            logit_scale = model.logit_scale.exp()

        obj_embeddings = []
        atk_embeddings = []
        # Compute text embeddings for each sample in the batch, using cache.
        for i in range(batch_size_current):
            obj_label = batch["object_label"][i]
            atk_word = batch["attack_word"][i]
            if obj_label not in object_embedding_cache:
                object_prompts = [tpl.format(obj_label) for tpl in TEMPLATES]
                object_tokens = tokenizer(object_prompts).to(device_name)
                with torch.no_grad(), autocast_context:
                    obj_emb = model.encode_text(object_tokens).mean(dim=0)
                    obj_emb /= obj_emb.norm(dim=-1, keepdim=True)
                object_embedding_cache[obj_label] = obj_emb
            else:
                obj_emb = object_embedding_cache[obj_label]
            obj_embeddings.append(obj_emb)

            if atk_word not in attack_embedding_cache:
                attack_prompts = [tpl.format(atk_word) for tpl in TEMPLATES]
                attack_tokens = tokenizer(attack_prompts).to(device_name)
                with torch.no_grad(), autocast_context:
                    atk_emb = model.encode_text(attack_tokens).mean(dim=0)
                    atk_emb /= atk_emb.norm(dim=-1, keepdim=True)
                attack_embedding_cache[atk_word] = atk_emb
            else:
                atk_emb = attack_embedding_cache[atk_word]
            atk_embeddings.append(atk_emb)

        obj_feats = torch.stack(obj_embeddings, dim=0).to(device_name)
        atk_feats = torch.stack(atk_embeddings, dim=0).to(device_name)

        sim_object = logit_scale * torch.sum(
            img_features * obj_feats, dim=1, keepdim=True
        )
        sim_attack = logit_scale * torch.sum(
            img_features * atk_feats, dim=1, keepdim=True
        )
        sims = torch.cat([sim_object, sim_attack], dim=1)
        probs = torch.softmax(sims, dim=1).cpu().numpy()

        for i in range(batch_size_current):
            results.append(
                {
                    "type": batch["type"][i],
                    "id": batch["id"][i],
                    "object_label": batch["object_label"][i],
                    "attack_word": batch["attack_word"][i],
                    "postit_area_pct": batch["postit_area_pct"][i].item(),
                    "object_similarities": sim_object[i].item(),
                    "attack_similarities": sim_attack[i].item(),
                    "object_prob": probs[i][0],
                    "attack_prob": probs[i][1],
                }
            )

        # break  # Test: only do one batch

    output_csv = os.path.join(res_dir, output_name)
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")


def parse_input():
    """Parse input arguments for the script."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--d",
        "--dir",
        default="data",
        type=str,
        help="Data directory (results are also stored in there)",
    )
    parser.add_argument(
        "--e",
        "--eval_dataset",
        default="SCAM",
        type=str,
        help="Evaluation dataset",
    )
    parser.add_argument(
        "--m",
        "--model_name",
        default="",
        type=str,
        help="Model name (e.g., 'ViT-B-32')",
    )
    parser.add_argument(
        "--pd", "--pretraining_data", default="", type=str, help="Pretraining dataset"
    )
    parser.add_argument(
        "--dn",
        "--device_name",
        default="cpu",
        type=str,
        help="Device ('cpu', 'mps' or 'cuda')",
    )
    parser.add_argument(
        "--b", "--batch_size", default=4, type=int, help="Batch size for inference"
    )
    parser.add_argument(
        "--o",
        "--overwrite",
        default=False,
        type=lambda x: x.lower() == "true",
        help="Overwrite existing results",
    )
    args = parser.parse_args()
    print(args)
    assert args.dn in [
        "cpu",
        "mps",
        "cuda",
    ], "[ERROR] Device must be either 'cpu', 'mps' or 'cuda'."
    return {
        "dir": args.d,
        "eval_dataset": args.e,
        "model_name": args.m,
        "pretraining_data": args.pd,
        "device_name": args.dn,
        "batch_size": args.b,
        "overwrite": args.o,
    }


if __name__ == "__main__":
    main()
