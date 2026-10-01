from pathlib import Path

import torch
import torch.nn as nn


def export_embedding_onnx(
    model: nn.Module, output_path: str | Path, img_size: int = 224
) -> None:
    """Export an eval-mode embedding model to ONNX with the TorchScript exporter."""
    dummy = torch.zeros(
        1, 3, img_size, img_size, device=next(model.parameters()).device
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    # dynamo=False: use legacy TorchScript exporter (no onnxscript dep required)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        dynamo=False,
        export_params=True,
        # ViT attention lowers to scaled_dot_product_attention, which needs >= 14.
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
