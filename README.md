# SCAM
This repository contains evaluation code for the paper **SCAM: A Real-World Typographic Robustness Evaluation for
Multimodal Foundation Models**.

[[arXiv Preprint](https://arxiv.org/abs/2504.04893)] [[Project Page](https://bliss-e-v.github.io/SCAM-project-page/)] [[HuggingFace Dataset](https://huggingface.co/datasets/BLISS-e-V/SCAM)]

## Overview

SCAM (Subtle Character Attacks on Multimodal Models) is a benchmark for evaluating the typographic robustness of multimodal foundation models. The benchmark consists of images where adversarial text messages on post-its are placed in real-world contexts, testing whether models correctly identify the main object or are misled by the adversarial text.

Our key findings include:
1. Models of both VLM and LVLM classes experience a significant drop in accuracy when faced with real-world typographic attacks
2. Synthetic attacks closely align with real-world attacks, providing empirical support for the established practice of evaluating on synthetic datasets
3. LVLMs, such as the LLaVA family, exhibit vulnerabilities to typographic attacks, reflecting weaknesses inherited from their vision encoders
4. Employing larger LLM backbones reduces susceptibility to typographic attacks in LVLMs while also improving their textual understanding

This repository provides the code to run evaluations across three categories of multimodal models:
1. Vision-Language Models (VLMs) via OpenCLIP (`main_vlm_openclip.py`)
2. Large Vision-Language Models (LVLMs) via OpenAI's API (`main_lvlm_openai.py`)
3. Open-access LVLMs via Ollama (`main_lvlm_ollama.py`)

## Installation

### Prerequisites
- Python 3.8+
- Docker (optional, for containerized execution)

### VLM Evaluation via OpenCLIP
```bash
cp vlm_openclip/requirements.txt .
pip install -r requirements.txt
```

For Docker:
```bash
cp vlm_openclip/Dockerfile .
docker build -t scam-vlm-openclip .
```

### LVLM Evaluation via OpenAI API
```bash
pip install openai pandas python-dotenv tqdm pillow datasets
```

### LVLM Evaluation via Ollama
```bash
cp lvlm_ollama/requirements.txt .
pip install -r requirements.txt
```

For Docker:
```bash
cp lvlm_ollama/Dockerfile .
docker build -t scam-lvlm-ollama .
```

### LVLM Evaluation via LLaVA
See our [LLaVA fork](https://github.com/Bliss-e-V/LLaVA-OpenCLIP) that is adapted to work with custom OpenCLIP vision encoders and explains how we train the custom LLaVA version evaluated in our paper in appendix A.6.

For Docker:
```bash
cp lvlm_llava/Dockerfile .
docker build -t scam-lvlm-ollama .
```

Note that you need to build a Docker container with the above mentioned LLaVA fork first.

## Datasets

### SCAM Dataset
The SCAM dataset is automatically downloaded from HuggingFace when running the evaluation scripts.

### Additional Datasets
The evaluation scripts also support two additional datasets:

1. **RTA100**: Download from [Defense-Prefix Repository](https://github.com/azuma164/Defense-Prefix) and extract to `data/RTA100/`

2. **PAINT**: Download from [Patching Repository](https://github.com/mlfoundations/patching) and extract to `data/PAINT/`


## Usage Examples

### VLM Evaluation via OpenCLIP
```bash
python main_vlm_openclip.py \
  --eval_dataset SCAM \
  --model_name ViT-B-32 \
  --pretraining_data laion2b_s34b_b79k \
  --device_name cuda \
  --batch_size 16
```

### LVLM Evaluation via OpenAI API
Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_key_here
```

Run the evaluation:
```bash
python main_lvlm_openai.py
```

### LVLM Evaluation via Ollama
```bash
python main_lvlm_ollama.py \
  --model_name llava:34b \
  --eval_dataset SCAM
```

### LVLM Evaluation via LLaVA
```bash
python main_lvlm_llava.py \
  --model_name llava-7b-UCSC \
  --model_path /llava-checkpoints/llava-v1.5-7b-UCSC-VLAA-ViT-L-14-CLIPA-336-datacomp1B-bs2-bf16-zero2-11.8 \
  --eval_dataset SCAM
```

### LVLM Evalution via VLMEvalKit

see our fork of `VLMEvalKit`: https://github.com/Bliss-e-V/VLMEvalKit

## Results

The following table shows the performance of various Vision-Language Models (VLMs) and Large Vision-Language Models (LVLMs) on the SCAM datasets. The table highlights the accuracy differences between the NoSCAM dataset and the SCAM dataset, showing how much performance drops when typographic attacks are introduced.

Note that internally all LLaVA models that we evaluate utilize `ViT-L-14-336` trained by OpenAI for the image encoding. Furthermore, `ViT-bigG-14` trained on `laion2b` is used in the Kandinsky diffusion model.

| **Model**                                | **Training data** | **NoSCAM Accuracy (%)** | **SCAM Accuracy (%)** | **Accuracy Drop (↓)** |
| ---------------------------------------- | ----------------- | ----------------------- | --------------------- | --------------------- |
| **Vision-Language Models (VLMs)**        |                   |                         |                       |                       |
| `RN50`                                   | `openai`          | 97.76                   | 36.61                 | ↓61.15                |
| `ViT-B-32`                               | `laion2b`         | 98.45                   | 74.68                 | ↓23.77                |
| `ViT-B-16`                               | `laion2b`         | 98.71                   | 69.16                 | ↓29.55                |
| `ViT-B-16-SigLIP`                        | `webli`           | 99.22                   | 81.40                 | ↓17.82                |
| `ViT-L-14`                               | `commonpool_xl`   | 99.48                   | 74.68                 | ↓24.80                |
| `ViT-L-14`                               | `openai`          | 99.14                   | 40.14                 | ↓59.00                |
| `ViT-L-14-336`                           | `openai`          | 99.22                   | 33.85                 | ↓65.37                |
| `ViT-L-14-CLIPA-336`                     | `datacomp1b`      | 99.57                   | 74.76                 | ↓24.81                |
| `ViT-g-14`                               | `laion2b`         | 99.05                   | 61.93                 | ↓37.12                |
| `ViT-bigG-14`                            | `laion2b`         | 99.40                   | 70.89                 | ↓28.51                |
| **Large Vision-Language Models (LVLMs)** |                   |                         |                       |                       |
| `llava-llama3:8b`                        | -                 | 98.09                   | 39.50                 | ↓58.59                |
| `llava:7b-v1.6`                          | -                 | 97.50                   | 58.43                 | ↓39.07                |
| `llava:13b-v1.6`                         | -                 | 98.88                   | 58.00                 | ↓40.88                |
| `llava:34b-v1.6`                         | -                 | 98.97                   | 84.85                 | ↓14.11                |
| `gemma3:4b`                              | -                 | 97.24                   | 58.05                 | ↓39.19                |
| `gemma3:12b`                             | -                 | 99.14                   | 52.02                 | ↓47.12                |
| `gemma3:27b`                             | -                 | 97.42                   | 81.67                 | ↓15.75                |
| `llama3.2-vision:90b`                    | -                 | 98.88                   | 71.01                 | ↓27.87                |
| `llama4:scout`                           | -                 | 99.23                   | 88.12                 | ↓11.10                |
| `gpt-4o-mini-2024-07-18`                 | -                 | 99.40                   | 84.68                 | ↓14.72                |
| `gpt-4o-2024-08-06`                      | -                 | 99.48                   | 96.82                 | ↓2.67                 |

You can reproduce these results by running the evaluation scripts as described in the [Usage Examples](#usage-examples) section.

For the full list of all 110 models evaluated, please refer to the table in the appendix of our [paper](https://arxiv.org/abs/2504.04893).

## Citation

If you find this repository useful for your research, please consider citing our paper:

```bibtex
@misc{scambliss2025,
    title={SCAM: A Real-World Typographic Robustness Evaluation for Multimodal Foundation Models},
    author={Justus Westerhoff and Erblina Purelku and Jakob Hackstein and Leo Pinetzki and Lorenz Hufe},
    year={2025},
    eprint={2504.04893},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2504.04893},
}
```

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

We thank the authors of OpenCLIP, Ollama, and the creators of the underlying multimodal models for making their work available to the research community. We especially thank the [BLISS](https://bliss.berlin) community for making this collaborative research effort possible.
