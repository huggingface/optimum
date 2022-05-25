from typing import Dict

from datasets import Dataset

from optimum.onnxruntime.configuration import AutoCalibrationConfig
from optimum.runs_base import Calibrator

from .. import ORTQuantizer
from ..configuration import QuantizationConfig


class OnnxRuntimeCalibrator(Calibrator):
    def __init__(
        self,
        calibration_dataset: Dataset,
        quantizer: ORTQuantizer,
        model_path: str,
        qconfig: QuantizationConfig,
        calibration_params: Dict,
    ):
        super().__init__(
            calibration_dataset=calibration_dataset,
            quantizer=quantizer,
            model_path=model_path,
            qconfig=qconfig,
            calibration_params=calibration_params,
        )

        # Remove the unnecessary columns of the calibration dataset before the calibration step
        self.calibration_dataset = self.quantizer.clean_calibration_dataset(calibration_dataset)

    def calibrate(self):
        # Create the calibration configuration given the selected calibration method
        if self.calibration_params["method"] == "entropy":
            calibration_config = AutoCalibrationConfig.entropy(self.calibration_dataset)
        elif self.calibration_params["method"] == "percentile":
            calibration_config = AutoCalibrationConfig.percentiles(
                self.calibration_dataset,
                percentile=self.calibration_params["calibration_histogram_percentile"],
            )
        else:
            calibration_config = AutoCalibrationConfig.minmax(
                self.calibration_dataset,
                self.calibration_params["calibration_moving_average"],
                self.calibration_params["calibration_moving_average_constant"],
            )

        # TODO estimate memory needed for entropy/percentile to autochoose number of shards
        num_calibration_shards = 4
        if not 1 <= num_calibration_shards <= len(self.calibration_dataset):
            raise ValueError(
                f"Invalid value of number of shards {num_calibration_shards} chosen to split the calibration"
                " dataset, should be higher than 0 and lower or equal to the number of samples "
                f"{len(self.calibration_dataset)}."
            )

        for i in range(num_calibration_shards):
            shard = self.calibration_dataset.shard(num_calibration_shards, i)
            self.quantizer.partial_fit(
                dataset=shard,
                calibration_config=calibration_config,
                onnx_model_path=self.model_path,
                operators_to_quantize=self.qconfig.operators_to_quantize,
                batch_size=8,
                use_external_data_format=False,
            )
        ranges = self.quantizer.compute_ranges()

        return ranges
