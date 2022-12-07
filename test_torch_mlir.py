import tempfile
import time
from tempfile import NamedTemporaryFile

from torch import tensor
from torch.nn import Module
import torch_mlir
import iree_torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification


def prepare_sentence_tokens(hf_model: str, sentence: str):
    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    return tensor([tokenizer.encode(sentence)])


class OnlyLogitsHuggingFaceModel(Module):
    """Wrapper that returns only the logits from a HuggingFace model."""

    def __init__(self, model_name: str):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,  # The pretrained model name.
            # The number of output labels--2 for binary classification.
            num_labels=2,
            # Whether the model returns attentions weights.
            output_attentions=False,
            # Whether the model returns all hidden-states.
            output_hidden_states=False,
            torchscript=True,
        )
        self.model.eval()

    def forward(self, input):
        # Return only the logits.
        return self.model(input)[0]


# Suppress warnings
import warnings
warnings.simplefilter("ignore")
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"


if __name__ == '__main__':
    # The HuggingFace model name to use
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"

    # The sentence to run the model on
    sentence = "The quick brown fox jumps over the lazy dog."

    print("Parsing sentence tokens.")
    example_input = prepare_sentence_tokens(model_name, sentence)

    print("Instantiating model.")
    model = OnlyLogitsHuggingFaceModel(model_name)

    print("Compiling with Torch-MLIR")
    linalg_on_tensors_mlir = torch_mlir.compile(
        model,
        example_input,
        output_type=torch_mlir.OutputType.LINALG_ON_TENSORS,
        use_tracing=True,
        verbose=False
    )

    # print(linalg_on_tensors_mlir)
    with open(os.path.join(tempfile.gettempdir(), "minilm.mlir"), mode="w") as tmp:
        tmp.write(linalg_on_tensors_mlir.operation.get_asm(large_elements_limit=10, enable_debug_info=True))

    print("Compiling with IREE")
    # Backend options:
    #
    # llvm-cpu - cpu, native code
    # vmvx - cpu, interpreted
    # vulkan - GPU for general GPU devices
    # cuda - GPU for NVIDIA devices
    iree_backend = "llvm-cpu"
    iree_vmfb = iree_torch.compile_to_vmfb(linalg_on_tensors_mlir, iree_backend)

    print("Loading in IREE")
    invoker = iree_torch.load_vmfb(iree_vmfb, iree_backend)

    print("Running on IREE")

    for _ in range(100):
        start = time.time_ns()
        result = invoker.forward(example_input)
        end = time.time_ns()

        print(f"Forward took: {(end - start) / 1000 / 1000} ms")
    print("Done")