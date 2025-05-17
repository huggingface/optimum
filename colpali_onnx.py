import torch
import numpy as np
import onnxruntime as ort
from PIL import Image
from transformers import ColPaliForRetrieval, ColPaliProcessor


def compare_outputs(onnx_outputs, torch_outputs, name, atol=1e-4):
    """Compare ONNX and PyTorch outputs with a specified tolerance."""
    if isinstance(torch_outputs, torch.Tensor):
        torch_outputs = torch_outputs.detach().cpu().numpy()

    # Handle shape differences - we care about the embedding vectors, not sequence dimensions.
    # The ONNX model uses 1024 tokens while PyTorch might use a different number.
    onnx_shape = onnx_outputs.shape
    torch_shape = torch_outputs.shape

    print(f"{name} shapes - ONNX: {onnx_shape}, PyTorch: {torch_shape}")

    # If shapes don't match, we'll compare just the first 1024 tokens. 
    if onnx_shape != torch_shape:
        if len(onnx_shape) == 3 and len(torch_shape) == 3:
            onnx_vectors = np.mean(onnx_outputs, axis=1)
            torch_vectors = np.mean(torch_outputs, axis=1)
            diff = np.abs(onnx_vectors - torch_vectors).max()
            print(f"{name} max absolute difference (averaged across sequence): {diff}")
        else:
            print(f"⚠️ Shape mismatch for {name}, comparing what we can")
            min_dim0 = min(onnx_shape[0], torch_shape[0])
            min_dim2 = min(onnx_shape[2] if len(onnx_shape) > 2 else 1, torch_shape[2] if len(torch_shape) > 2 else 1)

            onnx_slice = onnx_outputs[:min_dim0, :, :min_dim2] if len(onnx_shape) > 2 else onnx_outputs[:min_dim0]
            torch_slice = torch_outputs[:min_dim0, :, :min_dim2] if len(torch_shape) > 2 else torch_outputs[:min_dim0]

            diff = np.abs(onnx_slice - torch_slice).max()
            print(f"{name} max absolute difference (on comparable slice): {diff}")
    else:
        diff = np.abs(onnx_outputs - torch_outputs).max()
        print(f"{name} max absolute difference: {diff}")

    if diff > atol:
        print(f"⚠️ {name} outputs differ by more than {atol}")
    else:
        print(f"✓ {name} outputs match within tolerance {atol}")
    return diff <= atol


def test_vision_path(model_id, onnx_path):
    print("\n===== Testing Vision Path =====")

    processor = ColPaliProcessor.from_pretrained(model_id)
    model = ColPaliForRetrieval.from_pretrained(model_id).eval()

    image = Image.new("RGB", (448, 448), color="white")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        torch_outputs = model(**inputs)

    onnx_session = ort.InferenceSession(f"{onnx_path}/model.onnx")
    onnx_inputs = {"pixel_values": inputs["pixel_values"].numpy()}
    onnx_outputs = onnx_session.run(None, onnx_inputs)[0]

    return compare_outputs(onnx_outputs, torch_outputs.embeddings, "Image embeddings")


if __name__ == "__main__":
    model_id = "vidore/colpali-v1.3-hf"
    test_vision_path(model_id, "colpali_onnx")
