import os
from typing import Dict, List

import numpy as np
from datasets import Dataset

from ...runs_base import Calibrator
from .. import ORTQuantizer
from ..configuration import AutoCalibrationConfig, QuantizationConfig
from ..preprocessors import QuantizationPreprocessor
from ..preprocessors.passes import ExcludeGeLUNodes, ExcludeLayerNormNodes, ExcludeNodeAfter, ExcludeNodeFollowedBy


class OnnxRuntimeCalibrator(Calibrator):
    def __init__(
        self,
        calibration_dataset: Dataset,
        quantizer: ORTQuantizer,
        model_path: str,
        qconfig: QuantizationConfig,
        calibration_params: Dict,
        node_exclusion: List[str],
        run_dir_path: str,
    ):
        super().__init__(
            calibration_dataset=calibration_dataset,
            quantizer=quantizer,
            model_path=model_path,
            qconfig=qconfig,
            calibration_params=calibration_params,
            node_exclusion=node_exclusion,
            run_dir_path=run_dir_path,
        )

        # Remove the unnecessary columns of the calibration dataset before the calibration step
        self.calibration_dataset = self.quantizer.clean_calibration_dataset(calibration_dataset)

    def fit(self):
        # Create the calibration preprocessor excluding nodes
        quantization_preprocessor = QuantizationPreprocessor()

        if "layernorm" in self.node_exclusion:
            # Exclude the nodes constituting LayerNorm
            quantization_preprocessor.register_pass(ExcludeLayerNormNodes())
        if "gelu" in self.node_exclusion:
            # Exclude the nodes constituting GELU
            quantization_preprocessor.register_pass(ExcludeGeLUNodes())
        if "residual" in self.node_exclusion:
            # Exclude the residual connection Add nodes
            quantization_preprocessor.register_pass(ExcludeNodeAfter("Add", "Add"))
        if "gather" in self.node_exclusion:
            # Exclude the Add nodes following the Gather operator
            quantization_preprocessor.register_pass(ExcludeNodeAfter("Gather", "Add"))
        if "softmax" in self.node_exclusion:
            # Exclude the Add nodes followed by the Softmax operator
            quantization_preprocessor.register_pass(ExcludeNodeFollowedBy("Add", "Softmax"))
        if "residual-add-relu" in self.node_exclusion:
            quantization_preprocessor.register_pass(ExcludeNodeFollowedBy("Add", "Relu"))
        if "residual-relu-conv" in self.node_exclusion:
            quantization_preprocessor.register_pass(ExcludeNodeFollowedBy("Relu", "Conv"))

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
                onnx_augmented_model_name=os.path.join(self.run_dir_path, "augmented_model.onnx"),
                operators_to_quantize=self.qconfig.operators_to_quantize,
                batch_size=8,
                use_external_data_format=False,
            )
        ranges = self.quantizer.compute_ranges()

        if self.calibration_params["method"] in ["percentile", "entropy"]:
            np.save(
                os.path.join(self.run_dir_path, "calibration_histograms.npy"),
                self.quantizer._calibrator.collector.histogram_dict,
            )

        return ranges, quantization_preprocessor
