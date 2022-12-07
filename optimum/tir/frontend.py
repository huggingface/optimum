from enum import Enum
from typing import List, Optional, Union
from transformers import BatchEncoding, PreTrainedModel, TFPreTrainedModel
from transformers.utils.logging import get_logger
from torch_mlir import OutputType

LOGGER = get_logger("tir-compiler")


class _TirFrontEnd(Enum):
    PYTORCH = 0
    TENSORFLOW = 1
    JAX = 2


class TirTarget(Enum):
    COMPILED_CPU = "llvm-cpu"
    INTERPRETED_CPU = "vmvx"
    COMPILED_GPU = "vulkan"
    COMPILED_CUDA = "cuda"


class _TirPreTrainedModel:
    __slots__ = ("_invoker", "_original")

    def __init__(self, invoker, original):
        self._invoker = invoker
        self._original = original

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __getattr__(self, item):
        if item.startswith("__"):
            return self.__getattribute__(item)
        else:
            return getattr(self._original, item)

    def forward(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args, BatchEncoding):
            args = args.data
            return self._invoker(**args, **kwargs)
        else:
            return self._invoker(*args, **kwargs)


class TirCompiler:
    __slots__ = ("_model", "_dynamic_axes", "_wrapper", "_frontend", "_target", "_exported_ir")

    def __init__(
        self,
        model: Union[PreTrainedModel, TFPreTrainedModel],
        dynamic_axes: Optional[List[int]] = None,
        target: TirTarget = TirTarget.COMPILED_CPU
    ):
        self._wrapper = None

        self._model = model
        self._dynamic_axes = dynamic_axes
        self._target = target

        if self._model.config.return_dict:
            LOGGER.debug("Overriding config.return_dict to False to avoid compilation side-effect.")
            self._model.config.return_dict = False

        if isinstance(self._model, PreTrainedModel):
            import torch_mlir
            from optimum.tir.pytorch import convert_encodings_to_example_args, sanitize_pretrained_model_for_mlir

            self._frontend = _TirFrontEnd.PYTORCH

            # First, let's make sure we have a model in a good shape to present to the Torch MLIR exporter.
            dummy_inputs = convert_encodings_to_example_args(self._model.dummy_inputs, dynamic_axes=dynamic_axes)
            sanitized_model = sanitize_pretrained_model_for_mlir(self._model)

            # Then, we convert the model from Torch MLIR towards LinAlg dialect + Tensor
            LOGGER.info("Generating Model IR from PyTorch Module.")
            self._exported_ir = torch_mlir.compile(
                sanitized_model,
                example_args=dummy_inputs,
                output_type=OutputType.LINALG_ON_TENSORS,
                use_tracing=True,
                # ignore_traced_shapes=True,
                verbose=False
            )

    def __enter__(self):
        LOGGER.info(f"Compiling IR to {self._target}.")

        if self._frontend == _TirFrontEnd.PYTORCH:
            from iree_torch import compile_to_vmfb, load_vmfb

            LOGGER.debug("Compiling MLIR to")
            interpreted_ir = compile_to_vmfb(self._exported_ir, self._target)

            LOGGER.debug("Loading VMFB IR inside runtime.")
            invoker = load_vmfb(interpreted_ir, self._target)

            self._wrapper = _TirPreTrainedModel(invoker, self._model)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._wrapper = None



