from dataclasses import dataclass
from io import BytesIO
from logging import DEBUG
from typing import Dict, List, Union, Optional, Sequence, Callable
from inspect import signature

import torch
from iree.compiler import InputType, compile_str as iree_compile_str
from iree_torch import compile_to_vmfb, load_vmfb
from iree._runtime import VmModule, HalDevice
from torch import nn
from torch_mlir._mlir_libs._mlir.ir import Module
from transformers import BatchEncoding, PreTrainedModel
from transformers.utils.logging import get_logger
from torch_mlir import compile as torch_mlir_compile, ExampleArgs, TensorPlaceholder, OutputType

from tir import TirDispatcher, TirTarget, TirConfig

LOGGER = get_logger("tir.pytorch")


class _TirOutputWrapper(nn.Module):

    def __init__(self, model: PreTrainedModel):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        output = self.model(*args, **kwargs)

        if len(output) == 1:
            return output[0]
        else:
            return output


class TorchDispatcher(TirDispatcher):

    def __init__(self, model: Union[PreTrainedModel, nn.Module], target: TirTarget, config: TirConfig):
        super().__init__(sanitize_pretrained_model_for_mlir(model), target, config)

        if not isinstance(model, (PreTrainedModel, nn.Module)):
            raise ValueError(f"Invalid model type, awaiting PreTrainedModel or torch.nn.Module, got: {type(model)}.")

        self._parameters = self.infer_forward_signature(model)

    @staticmethod
    def infer_forward_signature(model: PreTrainedModel) -> List[str]:
        model_call_signature = signature(model.forward)
        return list(model_call_signature.parameters.keys())

    @staticmethod
    def internal_compile_to_vmfb(mlir_module, target: TirTarget, extra_args: List[str] = None):
        """
        This method is copy/pasted from torch-mlir but adds `extra-args` to give some more argument to the compiler.
        :param mlir_module:
        :param target_backend:
        :param extra_args:
        :return:
        """
        extra_args = extra_args or []

        # Here, mlir_module is typically going to be coming from the Torch-MLIR
        # MLIR CAPI assembly. We convert to bytecode to cross the border into the
        # IREE MLIR CAPI assembly.
        bytecode_stream = BytesIO()
        mlir_module.operation.write_bytecode(bytecode_stream)
        bytecode = bytecode_stream.getvalue()

        with open("mlir/bert_tiny.mlir", "w", encoding="utf-8") as f:
            f.write(str(mlir_module))

        return iree_compile_str(
            bytecode,
            target_backends=[target.value],
            input_type=InputType.TM_TENSOR,
            extra_args=extra_args
        )

    def validate_forward_inputs(self, *args, **kwargs):
        parameters = self._parameters.copy()

        # If we have a single args, let's assume it's the first one?
        if args:
            curated_inputs = list(args)
            parameters = parameters[len(args): ]  # Assume we just filled the slots for the N first parameters
        else:
            curated_inputs = []

        if kwargs:
            for input_name in parameters:
                if not kwargs:
                    break

                curated_inputs.append(kwargs.pop(input_name, None))

        return curated_inputs

    def export_model_to_mlir(
        self,
        model: Union[nn.Module, PreTrainedModel],
        target: TirTarget,
        examples: Optional = None,
        dynamic_axes: List[int] = None
    ):
        LOGGER.info(f"Exporting {type(model)} to MLIR.")
        return torch_mlir_compile(
            model,
            example_args=examples,
            output_type=OutputType.LINALG_ON_TENSORS,
            use_tracing=True,
            ignore_traced_shapes=dynamic_axes is not None,
            verbose=False
        )

    def compile_from_mlir(
        self,
        mlir_module: Module,
        target: TirTarget,
        device: Optional[HalDevice] = None,
        compiler_args: List[str] = None
    ):
        LOGGER.info(f"Compilation of MLIR module to {target} (device={device}).")
        LOGGER.debug(f"Compilation MLIR module with arguments: {compiler_args}.")

        vmfb = TorchDispatcher.internal_compile_to_vmfb(mlir_module, target, compiler_args)
        module = load_vmfb(vmfb, target.value)
        return module.forward

    def _internal_call(self, dispatch: Callable, curated_args):
        if LOGGER.isEnabledFor(DEBUG):
            LOGGER.debug(f"Calling dispatcher with {[type(arg)for arg in curated_args]} arguments.")

        return dispatch(*curated_args)


def sanitize_pretrained_model_for_mlir(model: PreTrainedModel):
    model = model.eval()

    if not model.config.torchscript:
        LOGGER.debug("Setting config.torchscript = True")
        model.config.torchscript = True

    if model.config.output_attentions:
        LOGGER.debug("Disabling output attentions.")
        model.config.output_attentions = False

    if model.config.output_hidden_states:
        LOGGER.debug("Disabling output hidden states.")
        model.config.output_hidden_states = False

    return _TirOutputWrapper(model)