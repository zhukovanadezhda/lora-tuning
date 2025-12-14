from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    LoRA Linear layer.

    Starts from a pretrained linear layer with parameters:
        W0 ∈ R^{out x in}, b ∈ R^{out}

    Instead of the standard Linear (y = x @ W0^T + b) LoRA makes a low-rank update:
        ΔW = (alpha / r) * (B @ A)
    with:
        A ∈ R^{r x in}
        B ∈ R^{out x r}
        r << min(in, out)

    Forward becomes:
        y = x @ W0^T + x @ (ΔW)^T + b
          = base(x) + lora(x)

    Initialization of LoRA parameters:
        A ~ N(0, sigma^2)
        B = 0
    So initially ΔW = 0 and the pretrained model behavior is preserved.

    Based on:
    "LoRA: Low-Rank Adaptation of Large Language Models"
    (https://arxiv.org/abs/2106.09685)
    """
    def __init__(
        self,
        base_layer: nn.Linear,
        r: int,
        alpha: float = 1.0,
        dropout: float = 0.0,
        init_std: float = 0.02
    ) -> None:
        super().__init__()

        self._validate_base_layer(base_layer)

        # Dimensions
        self.in_features: int = base_layer.in_features   # = in
        self.out_features: int = base_layer.out_features # = out

        # LoRA hyperparams
        self.r: int = int(r)
        self.alpha: float = float(alpha)
        self.scaling: float = (self.alpha / self.r) if self.r > 0 else 0.0

        # Frozen pretrained parameters (W0, b)
        self.weight: nn.Parameter
        self.bias: Optional[nn.Parameter]
        self._init_frozen_base(base_layer)

        # Trainable LoRA parameters (A, B) or None if r == 0
        self.lora_A: Optional[nn.Parameter] = None
        self.lora_B: Optional[nn.Parameter] = None
        self._init_lora_params(init_std)

        # Dropout applied only to LoRA branch input
        self.lora_dropout: nn.Module = (
            nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
            )

        # If merged=True, we have already added ΔW into self.weight
        self.merged: bool = False


    @staticmethod
    def _validate_base_layer(base_layer: nn.Module) -> None:
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base_layer)}")


    def _init_frozen_base(self, base_layer: nn.Linear) -> None:
        """
        Copy pretrained weights and freeze them (requires_grad=False).

        base_layer.weight: (out, in)
        base_layer.bias:   (out,) or None
        """
        self.weight = nn.Parameter(base_layer.weight.detach().clone(), requires_grad=False)

        if base_layer.bias is not None:
            self.bias = nn.Parameter(base_layer.bias.detach().clone(), requires_grad=False)
        else:
            self.bias = None


    def _init_lora_params(self, init_std: float) -> None:
        """
        Create trainable LoRA matrices A and B if r > 0.

        A: (r, in)
        B: (out, r)

        Init:
            A ~ N(0, init_std^2)
            B = 0
        so ΔW = (alpha / r) * (B@A) = 0 at initialization.
        """
        if self.r <= 0:
            return

        self.lora_A = nn.Parameter(torch.empty(self.r, self.in_features))
        nn.init.normal_(self.lora_A, mean=0.0, std=init_std)

        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.r))


    def logging(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"r={self.r}, alpha={self.alpha}, "
            f"scaling={self.scaling}, merged={self.merged}"
        )


    def _base_linear(self, x: torch.Tensor) -> torch.Tensor:
        """
        Base frozen linear:
            y0 = x @ W0^T + b

        Shapes:
            x:  (..., in)
            W0: (out, in)
            y0: (..., out)
        """
        return F.linear(x, self.weight, self.bias)

    def _lora_linear(self, x: torch.Tensor) -> torch.Tensor:
        """
        LoRA branch:
            y_lora = x @ (ΔW)^T
                   = (alpha / r) * x @ (B@A)^T
                   = (alpha / r) * (x @ A^T) @ B^T

        Shapes:
            x:    (..., in)
            A:    (r, in)  -> A^T: (in, r)
            B:    (out, r) -> B^T: (r, out)
        """
        if self.r <= 0 or self.lora_A is None or self.lora_B is None:
            # No LoRA parameters
            return torch.zeros_like(self._base_linear(x))

        x_d = self.lora_dropout(x)  # (..., in)

        # (x_d @ A^T): (..., r)
        z = x_d @ self.lora_A.t()

        # (z @ B^T): (..., out)
        y = z @ self.lora_B.t()

        return y * self.scaling

    def _delta_weight(self) -> torch.Tensor:
        """
        Compute ΔW in weight space:
            ΔW = (alpha / r) * (B@A)

        Shapes:
            B:  (out, r)
            A:  (r, in)
            ΔW: (out, in)
        """
        if self.r <= 0 or self.lora_A is None or self.lora_B is None:
            return torch.zeros_like(self.weight)

        return (self.lora_B @ self.lora_A) * self.scaling


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        If not merged:
            y = base(x) + lora(x)

        If merged:
            weight already includes ΔW, so: y = base(x)
        """
        if self.merged or self.r <= 0:
            return self._base_linear(x)

        return self._base_linear(x) + self._lora_linear(x)


    @torch.no_grad()
    def merge(self) -> None:
        """
        Merge LoRA into the frozen base weight for inference efficiency.

        After merge:
            weight := weight + ΔW
            merged = True

        Then forward uses only the base linear path.
        """
        if self.merged or self.r <= 0:
            self.merged = True
            return

        self.weight.add_(self._delta_weight())
        self.merged = True


    @torch.no_grad()
    def unmerge(self) -> None:
        """
        Undo merge:
            weight := weight - ΔW
            merged = False

        Useful to go back to training after inference.
        """
        if (not self.merged) or self.r <= 0:
            self.merged = False
            return

        self.weight.sub_(self._delta_weight())
        self.merged = False
