# LoRA: Low-Rank Adaptation of Large Language Models

## Overview

This project aims to explore LoRA and compare it with other types of fine-tuning:

* **Full fine-tuning**
* **LoRA (Low-Rank Adaptation)**
* **Adapter-based fine-tuning (Houlsby adapters)**

The goal is to explore the trade-off between performance, training cost, and number of trainable parameters, following the experimental philosophy of the original paper.

All experiments are conducted in a controlled setting:

* same base model
* same task and dataset
* same training protocol (tokenization and maximum sequence length, batch sizes, number of epochs, optimizer and learning rate schedule (except for method-specific learning rate scaling))
* identical evaluation metrics


## Experimental setup

### Task and dataset

* **Task**: Binary text classification
* **Dataset**: SST-2 (GLUE benchmark)
* **Metric**: Evaluation accuracy

### Model

* **Base model**: `bert-base-uncased`
* **Classification head**: task-specific, randomly initialized

### Fine-tuning strategies

* **Full fine-tuning**: all model parameters are trainable
* **LoRA**: low-rank adapters injected into linear layers, base weights frozen

  * Tested ranks: `r ∈ {1, 2, 4, 8, 16}`
* **Adapters**: Houlsby-style adapters with frozen backbone

### Training protocol

* Same number of epochs across all methods
* Same batch sizes and optimizer settings
* Only the intended parameters are trainable in each mode
* Training time, accuracy, and parameter counts are logged automatically


## Repository structure

```
lora-tuning/
├── scripts/
│   ├── data.py                # Dataset loading and tokenization
│   ├── metrics.py             # Parameter counting utilities
│   ├── run_tuning.py          # Main experiment launcher
│   ├── run_experiments.sh     # Run all experiments (FT, LoRA, adapters)
│   └── plot_results.py        # Generate plots and LaTeX tables
├── models/
│   ├── lora.py                # LoRA implementation
│   └── bert_with_lora.py      # LoRA injection into the base model
├── outputs/
│   ├── results/               # JSON results (one per experiment)
│   ├── figures/               # Plots and LaTeX tables
│   └── runs/                  # Trainer logs
├── requirements.txt
└── README.md
```


## Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/zhukovanadezhda/lora-tuning.git
cd lora-tuning
pip install -r requirements.txt
```

## Run experiments

To run all experiments (full fine-tuning, LoRA with multiple ranks, adapters):

```bash
bash scripts/run_experiments.sh
```

This script will:

* run each configuration sequentially
* save results as JSON files
* skip experiments that were already completed

Results are stored in:

```
outputs/results/
```

## Plot and analyse

To generate comparison plots and LaTeX tables:

```bash
python scripts/plot_results.py \
  --results-dir outputs/results \
  --out-dir output/figures
```

This produces:

* training loss curves
* accuracy vs trainable parameters
* accuracy vs training time
* a LaTeX table summarizing all methods


## Key findings (summary)

* LoRA achieves **near full fine-tuning performance** with **<1% trainable parameters**
* Increasing LoRA rank does **not** guarantee monotonic accuracy gains
* Adapter-based fine-tuning is competitive but slightly less parameter-efficient than LoRA
* Full fine-tuning remains the most expensive option in both time and parameters

These observations are **consistent with the trends reported in the original paper**, despite using a smaller-scale experimental setup.

## References

Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2022
[https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)
