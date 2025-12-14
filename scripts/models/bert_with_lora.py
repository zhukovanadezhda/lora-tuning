import torch.nn as nn
from transformers import BertForSequenceClassification

from scripts.models.lora import LoRALinear


def freeze_model(model: nn.Module) -> None:
    """
    Freeze all parameters in the model.
    """
    for param in model.parameters():
        param.requires_grad = False


def apply_lora_to_bert(
    model: BertForSequenceClassification,
    r: int,
    alpha: float = 1.0,
    dropout: float = 0.0
) -> BertForSequenceClassification:
    """
    Apply LoRA to BERT self-attention layers (query and value projections).

    Args:
        model  : BertForSequenceClassification
        r      : LoRA rank
        alpha  : LoRA scaling factor
        dropout: dropout on LoRA branch

    Returns:
        model with LoRA layers injected
    """

    if not isinstance(model, BertForSequenceClassification):
        raise TypeError(
            f"BertForSequenceClassification expected, got {type(model)}"
        )

    # Freeze everything first
    freeze_model(model)

    # Inject LoRA into each Transformer layer
    for layer in model.bert.encoder.layer:
        attention = layer.attention.self

        # Replace query projection
        attention.query = LoRALinear(
            base_layer=attention.query,
            r=r,
            alpha=alpha,
            dropout=dropout
        )

        # Replace value projection
        attention.value = LoRALinear(
            base_layer=attention.value,
            r=r,
            alpha=alpha,
            dropout=dropout
        )

    # Unfreeze classification head
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model
