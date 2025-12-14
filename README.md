# LoRA: Low-Rank Adaptation of Large Language Models

## Overview

This project aims to explore and implement LoRA and compare it to the full fine-tuning as well as to adapter-based fine-tuning. The goal is to explore the trade-off between performance, training cost, and number of trainable parameters, following the experimental philosophy of the original LoRA paper.

## Experimental setup

* **Task**: Binary text classification
* **Dataset**: SST-2 (GLUE benchmark)
* **Metric**: Evaluation accuracy
* **Base model**: BERT (`bert-base-uncased`, 110M parameters)
* **Classification head**: task-specific, randomly initialized

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

## Reproduce results

Clone the repository and install dependencies:

```bash
git clone https://github.com/zhukovanadezhda/lora-tuning.git
cd lora-tuning
pip install -r requirements.txt
```

To run all experiments (full fine-tuning, LoRA with multiple ranks, adapters):

```bash
bash scripts/run_experiments.sh
```

This script will:

* run each configuration
* save results as JSON files

Results are stored in:

```
outputs/results/
```

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
* Full fine-tuning remains the most expensive option in both time and parameters

These observations are **consistent with the trends reported in the original paper**, despite using a smaller-scale experimental setup.

## References

Houlsby, Neil et al. (2019). *Parameter-Efficient Transfer Learning for NLP.*   
arXiv:1902.00751 [cs.LG].url: https://arxiv.org/abs/1902.00751.

   
Hu, Edward J. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.*   
arXiv:2106.09685[cs.CL].url: https://arxiv.org/abs/2106.09685.


Liu, Haokun et al. (2022). *Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-ContextLearning.*  
arXiv: 2205.05638 [cs.LG].url: https://arxiv.org/abs/2205.05638.


Pfeiffer, Jonas et al. (2021). *AdapterFusion: Non-Destructive Task Composition for Transfer Learning.*   
arXiv:2005.00247 [cs.CL].url: https://arxiv.org/abs/2005.00247.6

