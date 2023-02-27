from typing import Union, Optional, List

from iree._runtime import HalDevice
from tensorflow.lite import TFLiteConverter

from tir import TirDispatcher, TirTarget


class TfLiteDispatcher(TirDispatcher):
    def export_model_to_mlir(self, model, examples: Optional = None, dynamic_axes: List[int] = None):
        converter = TFLiteConverter.from_keras_model(model)

        tflite_export = converter.convert()

    def compile_from_mlir(self, mlir: Union[bytes, str], target: TirTarget, device: Optional[HalDevice] = None):
        pass

    def validate_forward_inputs(self, *args, **kwargs):
        pass