from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Dict


def load_sst2(
    tokenizer_name: str = "bert-base-uncased",
    max_length: int = 128
    ) -> Dict:
    """
    Load and tokenize the SST-2 dataset.

    Args:
        tokenizer_name : HuggingFace tokenizer name.
        max_length     : Maximum sequence length.

    Returns:
        dict: tokenized train/validation/test datasets
    """

    # Load SST-2 from GLUE
    dataset = load_dataset("glue", "sst2")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize_fn(batch):
        return tokenizer(
            batch["sentence"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )

    # Tokenize datasets
    tokenized_dataset = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["sentence"]
    )

    # Rename label column for Transformers compatibility
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")

    # Set PyTorch format
    tokenized_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    return tokenized_dataset
