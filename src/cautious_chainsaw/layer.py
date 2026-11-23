import torch
import torch.nn as nn

from matmul import DEFAULT_GROUP_SIZE, int4_matmul
from packing import pack_int4
from quantization import quantize_int4


class QuantizedLinear(nn.Module):
    def __init__(
            self,
            packed_weights: torch.Tensor,
            scales: torch.Tensor,
            in_features: int,
            out_features: int,
            k_pad: int,
            group_size: int = DEFAULT_GROUP_SIZE,
            bias: torch.Tensor = None,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.k_pad = k_pad
        self.group_size = group_size

        self.register_buffer(
            "packed_weights",
            packed_weights.contiguous(),
        )

        self.register_buffer(
            "scales",
            scales.contiguous(),
        )

        if bias is not None:
            self.register_buffer(
                "bias",
                bias.detach().contiguous(),
            )
        else:
            self.register_buffer("bias", None)

    @classmethod
    def from_linear(
            cls,
            linear: nn.Linear,
            group_size: int = DEFAULT_GROUP_SIZE,
    ):
        if not isinstance(linear, nn.Linear):
            raise TypeError("linear must be nn.Linear")

        if not linear.weight.is_cuda:
            raise ValueError("linear weights must be on CUDA")

        q, scales, k, k_pad = quantize_int4(
            linear.weight.detach(),
            group_size=group_size,
        )

        packed = pack_int4(q)

        del q

        bias = (
            linear.bias.detach()
            if linear.bias is not None
            else None
        )

        return cls(
            packed_weights=packed,
            scales=scales,
            in_features=k,
            out_features=linear.out_features,
            k_pad=k_pad,
            group_size=group_size,
            bias=bias,
        )

    def forward(self, x: torch.Tensor):
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected input dimension {self.in_features}, "
                f"got {x.shape[-1]}"
            )

        if x.dtype not in (torch.float16, torch.bfloat16):
            raise TypeError("input must be FP16 or BF16")

        original_shape = x.shape

        x_2d = x.reshape(
            -1,
            self.in_features,
        ).contiguous()

        y = int4_matmul(
            x=x_2d,
            packed_weights=self.packed_weights,
            scales=self.scales,
            k=self.in_features,
            k_pad=self.k_pad,
            group_size=self.group_size,
            bias=self.bias,
        )

        return y.reshape(
            *original_shape[:-1],
            self.out_features,
        )
