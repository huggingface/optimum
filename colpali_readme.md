# ColPali ONNX Conversion

This guide explains how to convert ColPali models to ONNX format using our extended version of HuggingFace Optimum.

## What is ColPali?

ColPali is a multimodal retrieval model that combines PaliGemma (vision) and Gemma (text) models for visual document retrieval. It supports:
- Text-to-embedding pathway
- Image-to-embedding pathway

## Prerequisites

```bash
git clone https://github.com/ThinamXx/optimum.git
cd optimum

pip install -e .
pip install torch transformers onnx onnxruntime
```

## Converting ColPali to ONNX

The ColPali model can be exported in two modes:
1. `visual-retrieval-text`: For converting text input to embeddings
2. `visual-retrieval-vision`: For converting image input to embeddings

### Vision Path Conversion

```bash
optimum-cli export onnx --model vidore/colpali-v1.3-hf --task visual-retrieval-vision --height 448 --width 448 --output colpali_onnx
```

## Testing the ONNX Model

After exporting the model, use the `colpali_onnx.py` script to test the ONNX model's output against the original PyTorch model. The script handles dimension mismatches that may occur due to sequence length differences between the models.

## Important Notes

1. **Sequence Length**: Our ONNX export fixes the number of image tokens to 1024, which may differ from the PyTorch model's default. This is intentional and necessary for successful ONNX export.

2. **Image Size**: We use 448×448 as the image size in both export and testing to ensure consistency.

3. **Embedding Dimensions**: While the sequence dimensions may differ between ONNX and PyTorch models (1024 vs 1030), the actual embedding dimension (128) should remain consistent.

## Running the Test

After converting the model, run the test script:

```bash
python colpali_onnx.py
```

## Troubleshooting

- **Shape Mismatch**: If you see shape mismatch errors (e.g., `(1,1024,128)` vs `(1,1030,128)`), this is expected. The test script handles this by comparing the embeddings rather than the full tensor.

- **Token Count**: If you encounter the error "Number of images does not match number of special image tokens", make sure you're using our patched version of Optimum that handles the token count correctly in the PaliGemma model.

## Implementation Details

Our implementation extends Optimum with:

1. New configurations in `model_configs.py`:
   - Added `ColPaliNormalizedConfig` and `ColPaliOnnxConfig` classes

2. Custom model patchers in `model_patcher.py`:
   - Added `ColPaliModelPatcher` to handle both text and vision paths
   - Added `PaliGemmaModelPatcher` to fix token count issues (using 1024 tokens)

3. Task registration in `tasks.py`:
   - Added `visual-retrieval-text` and `visual-retrieval-vision` tasks
   - Mapped to `ColPaliForRetrieval` class

These extensions ensure proper ONNX export of the complex multimodal ColPali architecture. 