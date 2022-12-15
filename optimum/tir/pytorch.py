from dataclasses import dataclass
from io import BytesIO
from typing import Dict, List, Union, Optional, Sequence
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

from tir import TirDispatcher, TirTarget

LOGGER = get_logger("tir.pytorch")


@dataclass
class TorchConfig:
    """
    Data class to hold the necessary information to output a PyTorch model to MLIR
    Args:
        output: The kind of MLIR lowering we will target.
        use_tracing: Indicate if we should be using torch tracing to get the computation graph.
        examples: Dummy examples to generate the trace.
    """
    output: OutputType = OutputType.LINALG_ON_TENSORS
    use_tracing: bool = True
    examples: Optional[Union[torch.Tensor, Sequence[torch.Tensor], Dict[str, torch.Tensor], BatchEncoding]] = None


class TorchDispatcher(TirDispatcher):

    def __init__(self, model: Union[PreTrainedModel, nn.Module], target: TirTarget):
        super().__init__(sanitize_pretrained_model_for_mlir(model), target)

        if not isinstance(model, (PreTrainedModel, nn.Module)):
            raise ValueError(f"Invalid model type, awaiting PreTrainedModel or torch.nn.Module, got: {type(model)}.")

        self._parameters = self.infer_forward_signature(model)

    @staticmethod
    def infer_forward_signature(model: PreTrainedModel) -> List[str]:
        model_call_signature = signature(model.forward)
        return list(model_call_signature.parameters.keys())

    @staticmethod
    def internal_compile_to_vmfb(
        mlir_module,
        target_backend: str,
        extra_args: List[str] = None,
        cuda_llvm_target_arch: str = None
    ):
        """
        This method is copy/pasted from torch-mlir but adds `extra-args` to give some more argument to the compiler.
        :param mlir_module:
        :param target_backend:
        :param extra_args:
        :param cuda_llvm_target_arch:
        :return:
        """
        extra_args = extra_args or []
        if cuda_llvm_target_arch is not None:
            arch_flag = f"--iree-hal-cuda-llvm-target-arch={cuda_llvm_target_arch}"
            extra_args.append(arch_flag)

        # Here, mlir_module is typically going to be coming from the Torch-MLIR
        # MLIR CAPI assembly. We convert to bytecode to cross the border into the
        # IREE MLIR CAPI assembly.
        bytecode_stream = BytesIO()
        mlir_module.operation.write_bytecode(bytecode_stream)
        bytecode = bytecode_stream.getvalue()
        return iree_compile_str(
            bytecode,
            target_backends=[target_backend],
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
        return torch_mlir_compile(
            model,
            example_args=examples,
            output_type=OutputType.LINALG_ON_TENSORS,
            use_tracing=True,
            ignore_traced_shapes=dynamic_axes is not None
        )

    def compile_from_mlir(self, mlir_module: Module, target: TirTarget, device: Optional[HalDevice] = None):
        if target == TirTarget.COMPILED_CUDA:
            from torch.cuda import is_available as is_cuda_available, get_device_properties

            if is_cuda_available():
                cuda_device_props = get_device_properties(0)
                cuda_target_arch = f"sm_{cuda_device_props.major}{cuda_device_props.minor}"
                LOGGER.debug(f"Inferred CUDA Target: {cuda_target_arch}")
            else:
                # cuda_target_arch = None
                cuda_target_arch = "sm_89"
                extra_args = [
                    "--iree-hal-cuda-disable-loop-nounroll-wa",
                ]
        elif target == TirTarget.COMPILED_CPU:
            cuda_target_arch = None
            extra_args = [
                "--iree-llvm-target-cpu-features=host",
                "--iree-flow-demote-i64-to-i32",
            ]
        else:
            cuda_target_arch = extra_args = None

        vmfb = TorchDispatcher.internal_compile_to_vmfb(mlir_module, target.value, extra_args, cuda_target_arch)
        module = load_vmfb(vmfb, target.value)
        return module.forward


class TorchCompiler:

    def export(self, model: PreTrainedModel, config: TorchConfig, dynamic_axes: List[int] = None):
        LOGGER.info(f"Exporting model {model.name_or_path} to MLIR.")

        # First, let's make sure we have a model in a good shape to present to the Torch MLIR exporter.
        if config.examples:
            LOGGER.debug(f"Using provider example to generate the trace {config.examples}.")
            dummy_inputs = convert_encodings_to_example_args(config.examples, dynamic_axes=dynamic_axes)
        else:
            LOGGER.debug(f"Using model's dummy inputs to generate the trace {config.examples}.")
            dummy_inputs = convert_encodings_to_example_args(config.examples, dynamic_axes=dynamic_axes)

        sanitized_model = sanitize_pretrained_model_for_mlir(model)

        # Then, we convert the model from Torch MLIR towards LinAlg dialect + Tensor
        LOGGER.info("Generating Model IR from PyTorch Module.")
        return torch_mlir_compile(
            sanitized_model,
            example_args=dummy_inputs,
            output_type=OutputType.LINALG_ON_TENSORS,
            use_tracing=True,
            ignore_traced_shapes=dynamic_axes is not None
        )

    def compile(self, graph_ir: Union[bytes, str, Module], target: str, dialect: InputType, device: str, driver: str) -> VmModule:
        """

        Args:
            graph_ir:
            target:
            dialect:
            device:
            driver:

        Return:
        """
        print()
        # ir_bytes = into_bytes(graph_ir)
        # vmfb = compile_str(ir_bytes, target_backends=[target.value], input_type=dialect)
        #
        # config = Config(driver, device=device)
        # context = SystemContext(config=config)
        # return context, VmModule.from_flatbuffer(context.instance, vmfb)


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
            return self._invoker.forward(**args, **kwargs)
        else:
            return self._invoker.forward(*args, *tuple(kwargs.values()))


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


def create_attention_mask_from_input_ids(input_ids: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(input_ids)


def convert_encodings_to_example_args(
    encoding: Union[BatchEncoding, Dict],
    ensure_attention_mask: bool = True,
    dynamic_axes: List[int] = None
) -> ExampleArgs:

    if ensure_attention_mask and "attention_mask" not in encoding:
        LOGGER.debug("Creating attention_mask from input_ids.")
        encoding["attention_mask"] = create_attention_mask_from_input_ids(encoding["input_ids"])

    args = ExampleArgs()

    if dynamic_axes is None:
        fw_input_placeholders = list(encoding.values())
    else:
        fw_input_placeholders = list(map(
            lambda tensor: TensorPlaceholder.like(tensor, dynamic_axes=dynamic_axes),
            encoding.values()
        ))

    args.add_method("forward", fw_input_placeholders)
    # args.add_method("__call__", fw_input_placeholders)
    return args

