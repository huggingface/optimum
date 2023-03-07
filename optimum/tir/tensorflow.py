from abc import ABC
from logging import getLogger
from os import PathLike

import tensorflow as tf
from typing import Callable, List, Union, Optional, Tuple
from iree import runtime as ireert
from iree.compiler import InputType, tf as tfc
from tir import TirConfig, TirDispatcher, TirTarget, TirExportableModule, TirFrontend

LOGGER = getLogger("TensorFlowDispatcher")


class TirTensorFlowModule(TirExportableModule, ABC):

    DEFAULT_SERVING_SIGNATURE = "serving"

    def __init__(self, module_or_path_to_saved_model: Union[tf.Module, PathLike]):
        super().__init__()
        self._module = module_or_path_to_saved_model

    @property
    def framework(self) -> TirFrontend:
        return TirFrontend.TENSORFLOW

    @property
    def dialect(self) -> InputType:
        return InputType.XLA

    @property
    def signatures(self) -> List[str]:
        return [TirTensorFlowModule.DEFAULT_SERVING_SIGNATURE]


class TensorflowDispatcher(TirDispatcher):

    def __init__(self, model, target: TirTarget, config: TirConfig, signatures: Optional[List[str]]):
        if not signatures:
            signatures = ["serving"]

        if not isinstance(signatures, List):
            signatures = []

        super().__init__(model, target, config, signatures)

    def validate_forward_inputs(self, *args, **kwargs):
        return args + tuple(kwargs.values())

    def export_model_to_mlir(self, *args):
        if isinstance(self._model, str):
            return tfc.compile_saved_model(
                self._model,
                exported_names=self._signatures,
                import_only=True,
                output_mlir_debuginfo=False,
                import_extra_args=["--output-format=mlir-ir"]
            )
        else:
            return tfc.compile_module(
                self._model,
                exported_names=self._signatures,
                import_only=True,
                output_mlir_debuginfo=False,
                import_extra_args=["--output-format=mlir-ir"]
            )

    def compile_from_mlir(self, module_as_mlir: str):
        # flatbuffer_blob = compile_str(
        #     mlir,
        #     target_backends=[target.value],
        #     input_type="mhlo",
        #     strip_debug_ops=True,
        #     extra_args=compiler_args
        # )
        #
        # # Register the module with a runtime context.
        # backend = BackendInfo(target)
        # config = ireert.Config(driver)
        # ctx = ireert.SystemContext(config=config)
        # vm_module = ireert.VmModule.from_flatbuffer(ctx.instance, flatbuffer_blob)
        # ctx.add_vm_module(vm_module)
        #
        # return ctx.modules.module["forward"]
        return

    def _get_dispatching_key(self, method: str, inputs: Tuple[tf.Tensor]) -> str:
        params = ','.join([f"{'x'.join(map(str, ins.shape))}x{str(ins.dtype)[6:]}" for ins in inputs])
        return f"@{method}({params})"

    def _internal_call(self, dispatch: Callable, curated_args):
        return dispatch(curated_args)