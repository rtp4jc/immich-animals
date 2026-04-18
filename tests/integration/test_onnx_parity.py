import numpy as np
import onnxruntime as ort
import torch

from animal_id.embedding.backbones import BackboneType
from animal_id.embedding.config import DATA_CONFIG, TRAINING_CONFIG
from animal_id.embedding.export import export_embedding_onnx
from animal_id.embedding.models import AnimalEmbeddingModel

EMBEDDING_DIM = TRAINING_CONFIG["EMBEDDING_DIM"]
IMG_SIZE = DATA_CONFIG["IMG_SIZE"]


def test_onnx_embedding_parity(tmp_path):
    torch.manual_seed(42)

    model = AnimalEmbeddingModel(
        backbone_type=BackboneType.MOBILENET_V3_SMALL,
        num_classes=None,  # inference mode — no ArcFace head
        embedding_dim=EMBEDDING_DIM,
        pretrained=False,
    )
    model.eval()

    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

    with torch.no_grad():
        pt_out = model(dummy).numpy()

    onnx_path = tmp_path / "embedding.onnx"
    export_embedding_onnx(model, onnx_path, img_size=IMG_SIZE)

    session = ort.InferenceSession(str(onnx_path))
    onnx_out = session.run(None, {"input": dummy.numpy()})[0]

    assert np.allclose(pt_out, onnx_out, atol=1e-5), "PyTorch and ONNX outputs diverge"
    assert onnx_out.shape == (1, EMBEDDING_DIM)
    assert abs(np.linalg.norm(onnx_out[0]) - 1.0) < 1e-5, (
        "Embedding is not L2-normalized"
    )
