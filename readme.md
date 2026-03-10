# BudhiGPD — Fine-Tuned Qwen LLM with LoRA

## Overview

BudhiGPD is a fine-tuned large language model built on top of the **Qwen 3B architecture**.

This project demonstrates an **end-to-end LLM engineering pipeline**, including:

- dataset preparation
- LoRA fine-tuning
- model weight merging
- GPU inference deployment
- interactive web interface

The model was trained using **custom JSONL datasets** and deployed using **Replicate GPU infrastructure**, with a **Gradio interface hosted on HuggingFace Spaces**.

---

## Architecture

BudhiGPD follows an end-to-end LLM development pipeline:

```
Dataset (JSONL)
       ↓
LoRA Fine-Tuning (Kaggle GPU)
       ↓
Merged Model Weights (LoRA + Base Model)
       ↓
Replicate GPU Inference
       ↓
HuggingFace Spaces (Gradio UI)
       ↓
User
```

---

## Quick Start

Run the BudhiGPD model using the Replicate API.

### Python Example

```python
import replicate

output = replicate.run(
    "budhi1997/budhi-3b",
    input={
        "prompt": "Hello!"
    }
)

print(output)
```

Make sure you set your Replicate API token:

```
export REPLICATE_API_TOKEN=your_token_here
```

---

## Technologies Used

- Python
- PyTorch
- HuggingFace Transformers
- PEFT (LoRA Fine-Tuning)
- Replicate (GPU Inference)
- HuggingFace Spaces
- Gradio

---

## Project Goal

BudhiGPD was created as a **portfolio machine learning engineering project** demonstrating practical skills including:

- dataset design and preparation
- parameter-efficient LLM fine-tuning
- model adapter merging
- GPU inference deployment
- building an interactive web interface

The goal is to showcase the **full lifecycle of a modern LLM system**.

---

## Links

**GitHub Repository**

https://github.com/Zituch/budhigpd-replicate

**Live Demo (HuggingFace Spaces)**

https://huggingface.co/spaces/Budhi1997/budhigpd
