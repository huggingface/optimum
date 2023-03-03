import tensorflow as tf
from typing import Callable, List, Optional, Union
from iree import runtime as ireert
from iree.compiler import compile_str
from iree.compiler import tf as tfc
# from transformers import TFPreTrainedModel
from tir import TirConfig, TirDispatcher, TirTarget


class TensorflowDispatcher(TirDispatcher):

    def __init__(self, model, target: TirTarget, config: TirConfig, use_tflite: bool = False):
        super().__init__(model, target, config)
        self._use_tflite = use_tflite

    def validate_forward_inputs(self, *args, **kwargs):
        # TODO: Check this
        return args + tuple(kwargs.values())

    def _internal_call(self, dispatch: Callable, curated_args):
        return dispatch(curated_args)

    def export_model_to_mlir(self, model, target: TirTarget, examples: Optional = None, dynamic_axes: List[int] = None):
        if self._use_tflite:
            from iree.compiler.tflite import compile_str as compile_tflite_str

            forward_f = model.forward.get_concrete_function()
            converter = tf.lite.TFLiteConverter.from_concrete_functions([forward_f], model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

            if target is TirTarget.COMPILED_CUDA:
                converter.target_spec.supported_types = [tf.float16]

            tflite_module = converter.convert()
            return compile_tflite_str(
                tflite_module,
                import_only=True,
                target_backends=[target.value]
            )
        elif isinstance(model, str):
            return tfc.tf.compile_saved_model(model)
        else:
            return tfc.tf.compile_module(
                model,
                import_only=True,
                output_mlir_debuginfo=False,
                import_extra_args=["--output-format=mlir-ir"],
                exported_names=["forward"],
                # saved_model_tags=["serve"],
            )

    def compile_from_mlir(
        self,
        mlir,
        target: TirTarget,
        compiler_args: List[str] = None
    ):

        # TODO: Refactor this outside of TF, can be used everywhere
        if target is TirTarget.COMPILED_CUDA:
            extra_args = [
                # TODO: Change this hard-coded value
                "--iree-hal-cuda-llvm-target-arch=sm_89",
                "--iree-hal-cuda-disable-loop-nounroll-wa",
                # "--iree-enable-fusion-with-reduction-ops"
            ]
            driver = "cuda"
        elif target is TirTarget.COMPILED_CPU:
            extra_args = [
                "--iree-hal-llvm-target-cpu-features=host",
                "--iree-mhlo-demote-i64-to-i32=false",
                "--iree-flow-demote-i64-to-i32",
                "--iree-flow-convert-linalg-matmul-to-mmt4d"
            ]
            driver = "local-task"
        else:
            extra_args = []
            driver = "local-task"

        flatbuffer_blob = compile_str(
            mlir,
            target_backends=[target.value],
            input_type="mhlo",
            strip_debug_ops=True,
            extra_args=extra_args
        )

        # Register the module with a runtime context.
        # config = ireert.Config(driver)
        # ctx = ireert.SystemContext(config=config)
        # vm_module = ireert.VmModule.from_flatbuffer(ctx.instance, flatbuffer_blob)
        # ctx.add_vm_module(vm_module)

        return ctx.modules.module["forward"]