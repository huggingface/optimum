# coding=utf-8
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Base exporters config."""

import copy
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from transformers.utils import is_torch_available

from ..utils import (
    DEFAULT_DUMMY_SHAPES,
    DummyInputGenerator,
    logging,
)
from ..utils import TORCH_MINIMUM_VERSION as GLOBAL_MIN_TORCH_VERSION
from ..utils import TRANSFORMERS_MINIMUM_VERSION as GLOBAL_MIN_TRANSFORMERS_VERSION
from ..utils.doc import add_dynamic_docstring
from ..utils.import_utils import is_torch_version, is_transformers_version


if TYPE_CHECKING:
    from transformers import PretrainedConfig

logger = logging.get_logger(__name__)


GENERATE_DUMMY_DOCSTRING = r"""
        Generates the dummy inputs necessary for tracing the model. If not explicitely specified, default input shapes are used.

        Args:
            framework (`str`, defaults to `"pt"`):
                The framework for which to create the dummy inputs.
            batch_size (`int`, defaults to {batch_size}):
                The batch size to use in the dummy inputs.
            sequence_length (`int`, defaults to {sequence_length}):
                The sequence length to use in the dummy inputs.
            num_choices (`int`, defaults to {num_choices}):
                The number of candidate answers provided for multiple choice task.
            image_width (`int`, defaults to {width}):
                The width to use in the dummy inputs for vision tasks.
            image_height (`int`, defaults to {height}):
                The height to use in the dummy inputs for vision tasks.
            num_channels (`int`, defaults to {num_channels}):
                The number of channels to use in the dummpy inputs for vision tasks.
            feature_size (`int`, defaults to {feature_size}):
                The number of features to use in the dummpy inputs for audio tasks in case it is not raw audio.
                This is for example the number of STFT bins or MEL bins.
            nb_max_frames (`int`, defaults to {nb_max_frames}):
                The number of frames to use in the dummpy inputs for audio tasks in case the input is not raw audio.
            audio_sequence_length (`int`, defaults to {audio_sequence_length}):
                The number of frames to use in the dummpy inputs for audio tasks in case the input is raw audio.

        Returns:
            `Dict[str, [tf.Tensor, torch.Tensor]]`: A dictionary mapping the input names to dummy tensors in the proper framework format.
"""


class ExportConfig(ABC):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        logger.warning(
            "The `ExportConfig` class is deprecated and will be removed in a future version. "
            "Please use `ExporterConfig` instead."
        )


class ExporterConfig(ABC):
    """
    Base class describing metadata on how to export the model.

    Class attributes:

    - NORMALIZED_CONFIG_CLASS (`Type`) -- A class derived from [`~optimum.utils.NormalizedConfig`] specifying how to
    normalize the model config.
    - DUMMY_INPUT_GENERATOR_CLASSES (`Tuple[Type]`) -- A tuple of classes derived from
    [`~optimum.utils.DummyInputGenerator`] specifying how to create dummy inputs.
    - ATOL_FOR_VALIDATION (`Union[float, Dict[str, float]]`) -- A float or a dictionary mapping task names to float,
    where the float values represent the absolute tolerance value to use during model conversion validation.
    - MIN_TORCH_VERSION (`packaging.version.Version`, defaults to [`~optimum.exporters.utils.TORCH_MINIMUM_VERSION`]) -- The
    minimum torch version supporting the export of the model.
    - MIN_TRANSFORMERS_VERSION (`packaging.version.Version`, defaults to
    [`~optimum.exporters.utils.TRANSFORMERS_MINIMUM_VERSION`] -- The minimum transformers version supporting the
    export of the model. Not always up-to-date or accurate. This is more for internal use.
    - PATCHING_SPECS (`Optional[List[PatchingSpec]]`, defaults to `None`) -- Specify which operators / modules should be
    patched before performing the export, and how. This is useful when some operator is not supported for instance.

    Args:
        config (`transformers.PretrainedConfig`):
            The model configuration.
        task (`str`, defaults to `"feature-extraction"`):
            The task the model should be exported for.
        int_dtype (`str`, defaults to `"int64"`):
            The data type of integer tensors, could be ["int64", "int32", "int8"], default to "int64".
        float_dtype (`str`, defaults to `"fp32"`):
            The data type of float tensors, could be ["fp32", "fp16", "bf16"], default to "fp32".
    """

    NORMALIZED_CONFIG_CLASS = None
    DUMMY_INPUT_GENERATOR_CLASSES = ()
    ATOL_FOR_VALIDATION: Union[float, Dict[str, float]] = 1e-5
    MIN_TORCH_VERSION = GLOBAL_MIN_TORCH_VERSION
    MIN_TRANSFORMERS_VERSION = GLOBAL_MIN_TRANSFORMERS_VERSION
    _TASK_TO_COMMON_OUTPUTS = {
        "audio-classification": ["logits"],
        "audio-frame-classification": ["logits"],
        "automatic-speech-recognition": ["logits"],
        "audio-xvector": ["logits"],  # for onnx :  ["logits", "embeddings"]
        "depth-estimation": ["predicted_depth"],
        "document-question-answering": ["logits"],
        "feature-extraction": ["last_hidden_state"],  # for neuron : ["last_hidden_state", "pooler_output"]
        "fill-mask": ["logits"],
        "image-classification": ["logits"],
        "image-segmentation": ["logits"],
        "image-to-text": ["logits"],
        "image-to-image": ["reconstruction"],
        "mask-generation": ["logits"],
        "masked-im": ["reconstruction"],
        "multiple-choice": ["logits"],
        "object-detection": ["logits", "pred_boxes"],
        "question-answering": ["start_logits", "end_logits"],
        "semantic-segmentation": ["logits"],
        "text2text-generation": ["logits"],
        "text-classification": ["logits"],
        "text-generation": ["logits"],
        "time-series-forecasting": ["prediction_outputs"],
        "token-classification": ["logits"],
        "visual-question-answering": ["logits"],
        "zero-shot-image-classification": ["logits_per_image", "logits_per_text", "text_embeds", "image_embeds"],
        "zero-shot-object-detection": ["logits", "pred_boxes", "text_embeds", "image_embeds"],
    }

    def __init__(
        self,
        config: "PretrainedConfig",
        task: str,
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
    ):
        self.task = task
        self._config = config
        self._normalized_config = self.NORMALIZED_CONFIG_CLASS(self._config)
        self.int_dtype = int_dtype
        self.float_dtype = float_dtype

    def _create_dummy_input_generator_classes(self, **kwargs) -> List[DummyInputGenerator]:
        """
        Instantiates the dummy input generators from `self.DUMMY_INPUT_GENERATOR_CLASSES`.
        Each dummy input generator is independent, so this method instantiates the first generator, and
        forces the other generators to use the same batch size, meaning they will all produce inputs of the same batch
        size. Override this method for custom behavior.
        """
        return [cls_(self.task, self._normalized_config, **kwargs) for cls_ in self.DUMMY_INPUT_GENERATOR_CLASSES]

    @property
    @abstractmethod
    def inputs(self) -> Dict[str, Dict[int, str]]:
        """
        Dict containing the axis definition of the input tensors to provide to the model.

        Returns:
            `Dict[str, Dict[int, str]]`: A mapping of each input name to a mapping of axis position to the axes symbolic name.
        """
        raise NotImplementedError()

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        """
        Dict containing the axis definition of the output tensors to provide to the model.

        Returns:
            `Dict[str, Dict[int, str]]`: A mapping of each output name to a mapping of axis position to the axes symbolic name.
        """
        common_outputs = self._TASK_TO_COMMON_OUTPUTS[self.task]
        return copy.deepcopy(common_outputs)

    @property
    def values_override(self) -> Optional[Dict[str, Any]]:
        """
        Dictionary of keys to override in the model's config before exporting.

        Returns:
            `Optional[Dict[str, Any]]`: A dictionary specifying the configuration items to override.
        """
        if hasattr(self._config, "use_cache"):
            return {"use_cache": False}

        return None

    @property
    def is_transformers_support_available(self) -> bool:
        """
        Whether the installed version of Transformers allows the export.

        Returns:
            `bool`: Whether the install version of Transformers is compatible with the model.

        """
        return is_transformers_version(">=", self.MIN_TRANSFORMERS_VERSION.base_version)

    @property
    def is_torch_support_available(self) -> bool:
        """
        Whether the installed version of PyTorch allows the export.

        Returns:
            `bool`: Whether the installed version of PyTorch is compatible with the model.
        """
        if is_torch_available():
            return is_torch_version(">=", self.MIN_TORCH_VERSION.base_version)

        return False

    @add_dynamic_docstring(text=GENERATE_DUMMY_DOCSTRING, dynamic_elements=DEFAULT_DUMMY_SHAPES)
    def generate_dummy_inputs(self, framework: str = "pt", **kwargs) -> Dict:
        dummy_inputs_generators = self._create_dummy_input_generator_classes(**kwargs)
        dummy_inputs = {}
        for input_name in self.inputs:
            input_was_inserted = False
            for dummy_input_gen in dummy_inputs_generators:
                if dummy_input_gen.supports_input(input_name):
                    dummy_inputs[input_name] = dummy_input_gen.generate(
                        input_name, framework=framework, int_dtype=self.int_dtype, float_dtype=self.float_dtype
                    )
                    input_was_inserted = True
                    break
            if not input_was_inserted:
                raise RuntimeError(
                    f'Could not generate dummy input for "{input_name}". Try adding a proper dummy input generator to '
                    "the model exporters config."
                )
        return dummy_inputs
