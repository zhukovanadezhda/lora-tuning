import torch
from typing import Dict


def compute_accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor
    ) -> float:
    """
    Compute classification accuracy.

    Args:
        predictions : Logits of shape (batch_size, num_classes)
        labels      : Ground-truth labels of shape (batch_size,)

    Returns:
        float: accuracy in [0, 1]
    """
    preds = torch.argmax(predictions, dim=-1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct / total


def count_trainable_parameters(model) -> int:
    """
    Count the number of trainable parameters in a model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model) -> int:
    """
    Count the total number of parameters in a model.
    """
    return sum(p.numel() for p in model.parameters())


def parameter_summary(model) -> Dict[str, int]:
    """
    Return a dictionary with total and trainable parameter counts.
    """
    total = count_total_parameters(model)
    trainable = count_trainable_parameters(model)

    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_percentage": 100.0 * trainable / total
    }
