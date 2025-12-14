import os
import json
import time
import argparse
from typing import Tuple, Dict, Any

import numpy as np
import evaluate

from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer
)

from peft import get_peft_model, AdapterConfig, TaskType

from scripts.data import load_sst2
from scripts.metrics import parameter_summary
from scripts.models.bert_with_lora import apply_lora_to_bert


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run fine-tuning experiment")

    parser.add_argument("--mode", choices=["full_ft", "lora", "adapter"], required=True)
    parser.add_argument("--model-name", default="bert-base-uncased")
    parser.add_argument("--max-length", type=int, default=128)

    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)

    parser.add_argument("--r", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)

    parser.add_argument("--output-dir", default="outputs/runs")
    parser.add_argument("--results-path", default=None)

    return parser.parse_args()


def resolve_defaults(args: argparse.Namespace) -> None:
    """Set default values for arguments based on mode."""
    if args.lr is None:
        if args.mode == "full_ft":
            args.lr = 2e-5
        elif args.mode == "lora":
            args.lr = 1e-4
        elif args.mode == "adapter":
            args.lr = 1e-4

    if args.results_path is None:
        (
            args.results_path = f"outputs/results/{args.mode}.json"
            if args.mode == "full_ft" or args.mode == "adapter"
            else f"outputs/results/lora_r{args.r}.json"
        )


def prepare_output_dirs(args: argparse.Namespace) -> None:
    """Create output directories if they do not exist."""
    os.makedirs(os.path.dirname(args.results_path), exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)


def load_data_and_tokenizer(
    model_name: str,
    max_length: int
):
    """Load SST-2 dataset and tokenizer."""
    dataset = load_sst2(
        tokenizer_name=model_name,
        max_length=max_length
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return dataset, tokenizer


def build_model(args: argparse.Namespace):
    """Build model based on the specified mode."""
    model = BertForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2
    )

    if args.mode == "lora":
        model = apply_lora_to_bert(
            model,
            r=args.r,
            alpha=args.alpha,
            dropout=args.lora_dropout
        )

    elif args.mode == "adapter":
        adapter_config = AdapterConfig(
            task_type=TaskType.SEQ_CLS,
            reduction_factor=16,
            non_linearity="relu"
        )

        model = get_peft_model(model, adapter_config)

    return model


def build_trainer(
    model,
    dataset,
    tokenizer,
    args: argparse.Namespace
) -> Trainer:
    """Build Trainer instance."""
    acc_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        """Compute accuracy metric."""
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return acc_metric.compute(
            predictions=preds,
            references=labels,
        )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        evaluation_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        save_strategy="no",
        report_to=[]
    )

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )


def run_experiment(
    trainer: Trainer,
    model,
    args: argparse.Namespace
) -> Dict[str, Any]:
    """Run training and evaluation, return results."""

    params = parameter_summary(model)
    print("[PARAMS]", params)

    start = time.time()
    trainer.train()
    metrics = trainer.evaluate()
    elapsed = time.time() - start
    log_history = trainer.state.log_history

    return {
        "mode": args.mode,
        "model": args.model_name,
        "metrics": metrics,
        "params": params,
        "elapsed_sec": elapsed,
        "config": vars(args),
        "log_history": log_history
    }


def save_results(results: Dict[str, Any], path: str) -> None:
    """Save results to a JSON file."""
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def main():
    """Main function to run the experiment."""
    args = parse_args()
    resolve_defaults(args)
    prepare_output_dirs(args)

    dataset, tokenizer = load_data_and_tokenizer(
        args.model_name,
        args.max_length
    )

    model = build_model(args)

    trainer = build_trainer(
        model=model,
        dataset=dataset,
        tokenizer=tokenizer,
        args=args
    )

    results = run_experiment(
        trainer=trainer,
        model=model,
        args=args
    )

    save_results(results, args.results_path)

    print(f"[OK] Results saved to {args.results_path}")
    print(f"[OK] Eval accuracy: {results['metrics'].get('eval_accuracy')}")


if __name__ == "__main__":
    main()
