# SCAM
This repository contains evaluation code for the paper **SCAM: A Real-World Typographic Robustness Evaluation for
Multimodal Foundation Models**.

[[arXiv Preprint](https://arxiv.org/abs/2504.04893)] [[Project Page](https://bliss-e-v.github.io/SCAM-project-page/)] [[HuggingFace Dataset](https://huggingface.co/datasets/BLISS-e-V/SCAM)]

## Overview

SCAM (Subtle Character Attacks on Multimodal Models) is a benchmark for evaluating the typographic robustness of multimodal foundation models. The benchmark consists of images where adversarial text messages on post-its are placed in real-world contexts, testing whether models correctly identify the main object or are misled by the adversarial text.

Our key findings include:
1. Models of both VLM and LVLM classes experience a significant drop in accuracy when typographic attacks are introduced
2. Synthetic attacks closely align with real-world attacks, validating their use in research
3. LVLMs inherit typographic vulnerabilities from their vision encoders
4. Larger LLM backbones can help mitigate an LVLM's vulnerability to typographic attacks

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
See our [LLaVA fork](https://github.com/Bliss-e-V/LLaVA-OpenCLIP) adapted to work with custom OpenCLIP vision encoders.

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
