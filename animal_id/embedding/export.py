from pathlib import Path

import torch
import torch.nn as nn


def export_embedding_onnx(
    model: nn.Module, output_path: str | Path, img_size: int = 224
) -> None:
    """Export an embedding model to ONNX using the legacy TorchScript exporter.

    Args:
        model: Embedding model in eval mode on the target device.
        output_path: Destination .onnx file path.
        img_size: Spatial size of the square input (default 224).
    """
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
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
