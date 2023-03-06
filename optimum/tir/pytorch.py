from io import BytesIO
from logging import DEBUG
from typing import List, Union, Optional, Callable
from inspect import signature

import torch
from iree.compiler import InputType, compile_str as iree_compile_str
from iree_torch import load_vmfb
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

    def __init__(self, model: nn.Module, target: TirTarget, config: TirConfig, signatures: Optional[List[str]] = None):
        super().__init__(sanitize_pretrained_model_for_mlir(model), target, config, signatures)

        if not isinstance(model, (PreTrainedModel, nn.Module)):
            raise ValueError(f"Invalid model type, awaiting PreTrainedModel or torch.nn.Module, got: {type(model)}.")

        self._model_args = self.infer_forward_signature(model)

    @staticmethod
    def infer_forward_signature(model: nn.Module) -> List[str]:
        model_call_signature = signature(model.forward)
        return list(model_call_signature.parameters.keys())

    @staticmethod
    def internal_compile_to_vmfb(mlir_module, target: TirTarget, extra_args: List[str] = None):
        """
        This method is copy/pasted from torch-mlir but adds `extra-args` to give some more argument to the compiler.
        :param mlir_module:
        :param target:
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

        return iree_compile_str(
            bytecode,
            target_backends=[target.value],
            input_type=InputType.TM_TENSOR,
            extra_args=extra_args
        )

    def validate_forward_inputs(self, *args, **kwargs):
        model_args = self._model_args.copy()

        # If we have a single args, let's assume it's the first one?
        if args:
            curated_inputs = list(args)
            model_args = model_args[len(args): ]  # Assume we just filled the slots for the N first parameters
        else:
            curated_inputs = []

        if kwargs:
            for input_name in model_args:
                if not kwargs:
                    break

                curated_inputs.append(kwargs.pop(input_name, None))

        return curated_inputs

    def export_model_to_mlir(self, *args):
        LOGGER.info(f"Exporting {type(self._model)} to MLIR.")
        return torch_mlir_compile(
            self._model,
            example_args=args,
            output_type=OutputType.LINALG_ON_TENSORS,
            use_tracing=True,
            # ignore_traced_shapes=dynamic_axes is not None,
            ignore_traced_shapes=True,
            verbose=False
        )

    def compile_from_mlir(self, mlir_module: Module, compiler_args: List[str] = None):
        LOGGER.info(f"Compilation of MLIR module to {self._target}.")
        LOGGER.debug(f"Compilation MLIR module with arguments: {compiler_args}.")

        vmfb = TorchDispatcher.internal_compile_to_vmfb(mlir_module, self._target, compiler_args)
        module = load_vmfb(vmfb, self._target.value)
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