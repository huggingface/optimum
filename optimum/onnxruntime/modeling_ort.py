#  Copyright 2022 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""ORTModelForXXX classes, allowing to run ONNX Models with ONNX Runtime using the same API as Transformers."""

import logging
import os
import re
import shutil
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForAudioClassification,
    AutoModelForAudioFrameClassification,
    AutoModelForAudioXVector,
    AutoModelForCTC,
    AutoModelForImageClassification,
    AutoModelForImageToImage,
    AutoModelForMaskedLM,
    AutoModelForMultipleChoice,
    AutoModelForQuestionAnswering,
    AutoModelForSemanticSegmentation,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    GenerationMixin,
)
from transformers.file_utils import add_end_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from transformers.modeling_outputs import (
    BaseModelOutput,
    CausalLMOutput,
    ImageClassifierOutput,
    ImageSuperResolutionOutput,
    MaskedLMOutput,
    ModelOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SemanticSegmenterOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
    XVectorOutput,
)
from transformers.utils import cached_file, is_offline_mode

import onnxruntime as ort

from ..exporters import TasksManager
from ..exporters.onnx import main_export
from ..modeling_base import FROM_PRETRAINED_START_DOCSTRING, OptimizedModel
from ..onnx.utils import _get_external_data_paths
from ..utils.file_utils import find_files_matching_pattern
from ..utils.save_utils import maybe_save_preprocessors
from .constants import ONNX_FILE_PATTERN
from .io_binding import IOBindingHelper, TypeHelper
from .utils import (
    check_io_binding,
    get_device_for_provider,
    get_provider_for_device,
    parse_device,
    validate_provider_availability,
)


if TYPE_CHECKING:
    from transformers import PretrainedConfig


logger = logging.getLogger(__name__)


_TOKENIZER_FOR_DOC = "AutoTokenizer"
_FEATURE_EXTRACTOR_FOR_DOC = "AutoFeatureExtractor"
_PROCESSOR_FOR_DOC = "AutoProcessor"

ONNX_MODEL_END_DOCSTRING = r"""
    This model inherits from [`~onnxruntime.modeling_ort.ORTModel`], check its documentation for the generic methods the
    library implements for all its model (such as downloading or saving).

    This class should be initialized using the [`onnxruntime.modeling_ort.ORTModel.from_pretrained`] method.
"""

ONNX_TEXT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`Union[torch.Tensor, np.ndarray, None]` of shape `({0})`, defaults to `None`):
            Indices of input sequence tokens in the vocabulary.
            Indices can be obtained using [`AutoTokenizer`](https://huggingface.co/docs/transformers/autoclass_tutorial#autotokenizer).
            See [`PreTrainedTokenizer.encode`](https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizerBase.encode) and
            [`PreTrainedTokenizer.__call__`](https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizerBase.__call__) for details.
            [What are input IDs?](https://huggingface.co/docs/transformers/glossary#input-ids)
        attention_mask (`Union[torch.Tensor, np.ndarray, None]` of shape `({0})`, defaults to `None`):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](https://huggingface.co/docs/transformers/glossary#attention-mask)
        token_type_ids (`Union[torch.Tensor, np.ndarray, None]` of shape `({0})`, defaults to `None`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
            - 1 for tokens that are **sentence A**,
            - 0 for tokens that are **sentence B**.
            [What are token type IDs?](https://huggingface.co/docs/transformers/glossary#token-type-ids)
"""

ONNX_IMAGE_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`Union[torch.Tensor, np.ndarray, None]` of shape `({0})`, defaults to `None`):
            Pixel values corresponding to the images in the current batch.
            Pixel values can be obtained from encoded images using [`AutoFeatureExtractor`](https://huggingface.co/docs/transformers/autoclass_tutorial#autofeatureextractor).
"""

ONNX_AUDIO_INPUTS_DOCSTRING = r"""
    Args:
        input_values (`torch.Tensor` of shape `({0})`):
            Float values of input raw speech waveform..
            Input values can be obtained from audio file loaded into an array using [`AutoFeatureExtractor`](https://huggingface.co/docs/transformers/autoclass_tutorial#autofeatureextractor).
"""


class classproperty:
    def __init__(self, getter):
        self.getter = getter

    def __get__(self, instance, owner):
        return self.getter(owner)


class ORTModel(OptimizedModel):
    """
    Base class for implementing models using ONNX Runtime.

    The ORTModel implements generic methods for interacting with the Hugging Face Hub as well as exporting vanilla
    transformers models to ONNX using `optimum.exporters.onnx` toolchain.

    Class attributes:
        - model_type (`str`, *optional*, defaults to `"onnx_model"`) -- The name of the model type to use when
        registering the ORTModel classes.
        - auto_model_class (`Type`, *optional*, defaults to `AutoModel`) -- The "AutoModel" class to represented by the
        current ORTModel class.

    Common attributes:
        - model (`ort.InferenceSession`) -- The ONNX Runtime InferenceSession that is running the model.
        - config ([`~transformers.PretrainedConfig`] -- The configuration of the model.
        - use_io_binding (`bool`, *optional*, defaults to `True`) -- Whether to use I/O bindings with **ONNX Runtime
        with the CUDAExecutionProvider**, this can significantly speedup inference depending on the task.
        - model_save_dir (`Path`) -- The directory where the model exported to ONNX is saved.
        By defaults, if the loaded model is local, the directory where the original model will be used. Otherwise, the
        cache directory is used.
        - providers (`List[str]) -- The list of execution providers available to ONNX Runtime.
    """

    model_type = "onnx_model"
    auto_model_class = AutoModel

    @classproperty
    def export_feature(cls):
        logger.warning(f"{cls.__name__}.export_feature is deprecated, and will be removed in optimum 2.0.")
        try:
            feature = TasksManager.infer_task_from_model(cls.auto_model_class)
        except ValueError:
            feature = None
        return feature

    @classmethod
    def _auto_model_to_task(cls, auto_model_class):
        """
        Get the task corresponding to a class (for example AutoModelForXXX in transformers).
        """
        return TasksManager.infer_task_from_model(auto_model_class)

    def shared_attributes_init(
        self,
        model: ort.InferenceSession,
        use_io_binding: Optional[bool] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        preprocessors: Optional[List] = None,
        **kwargs,
    ):
        """
        Initializes attributes that may be shared among several ONNX Runtime inference sesssions.
        """
        # TODO: remove at version 2.0
        if kwargs.pop("latest_model_name", None) is not None:
            logger.warning(
                f"The latest_model_name argument to create an {self.__class__.__name__} is deprecated, and not used "
                "anymore."
            )
        if kwargs:
            raise ValueError(
                f"{self.__class__.__name__} received {', '.join(kwargs.keys())}, but do not accept those arguments."
            )

        self.providers = model.get_providers()
        self._device = get_device_for_provider(
            self.providers[0], provider_options=model.get_provider_options()[self.providers[0]]
        )

        # This attribute is needed to keep one reference on the temporary directory, since garbage collecting it
        # would end-up removing the directory containing the underlying ONNX model.
        self._model_save_dir_tempdirectory_instance = None
        if model_save_dir is None:
            self.model_save_dir = Path(model._model_path).parent
        elif isinstance(model_save_dir, TemporaryDirectory):
            self._model_save_dir_tempdirectory_instance = model_save_dir
            self.model_save_dir = Path(model_save_dir.name)
        elif isinstance(model_save_dir, str):
            self.model_save_dir = Path(model_save_dir)
        else:
            self.model_save_dir = model_save_dir

        self.preprocessors = preprocessors if preprocessors is not None else []

        if self._device is None:
            logger.warning(
                f"ORTModel outputs will be sent to CPU as the device could not be inferred from the execution provider {self.providers[0]}."
                f" Use `ort_model.to()` to send the outputs to the wanted device."
            )

        self._use_io_binding = use_io_binding

        # Registers the ORTModelForXXX classes into the transformers AutoModel classes to avoid warnings when creating
        # a pipeline https://github.com/huggingface/transformers/blob/cad61b68396a1a387287a8e2e2fef78a25b79383/src/transformers/pipelines/base.py#L863
        AutoConfig.register(self.model_type, AutoConfig)
        if hasattr(self.auto_model_class, "register"):
            self.auto_model_class.register(AutoConfig, self.__class__)

        # Define the pattern here to avoid recomputing it everytime.
        self.output_shape_inference_pattern = re.compile(r"([a-zA-Z_]+)|([0-9]+)|([+-/*])|([\(\)])")

    def __init__(
        self,
        model: ort.InferenceSession,
        config: "PretrainedConfig",
        use_io_binding: Optional[bool] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        preprocessors: Optional[List] = None,
        **kwargs,
    ):
        super().__init__(model, config)

        if use_io_binding is None:
            if model.get_providers()[0] == "CUDAExecutionProvider":
                use_io_binding = True
            else:
                use_io_binding = False

        self.model_path = Path(model._model_path)
        self.model_name = self.model_path.name

        self.shared_attributes_init(
            model,
            use_io_binding,
            model_save_dir,
            preprocessors,
            **kwargs,
        )

        self.input_names = {input_key.name: idx for idx, input_key in enumerate(model.get_inputs())}
        self.input_dtypes = {input_key.name: input_key.type for input_key in model.get_inputs()}

        self.output_names = {output_key.name: idx for idx, output_key in enumerate(model.get_outputs())}
        self.output_dtypes = {output_key.name: output_key.type for output_key in model.get_outputs()}

    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the model.
        """

        for dtype in self.input_dtypes.values():
            torch_dtype = TypeHelper.ort_type_to_torch_type(dtype)
            if torch_dtype.is_floating_point:
                return torch_dtype

        for dtype in self.output_dtypes.values():
            torch_dtype = TypeHelper.ort_type_to_torch_type(dtype)
            if torch_dtype.is_floating_point:
                return torch_dtype

        return None

    @property
    def device(self) -> torch.device:
        """
        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """
        return self._device

    @device.setter
    def device(self, **kwargs):
        raise AttributeError("The device attribute is read-only, please use the `to` method to change the device.")

    @property
    def use_io_binding(self):
        return check_io_binding(self.providers, self._use_io_binding)

    @use_io_binding.setter
    def use_io_binding(self, value: bool):
        self._use_io_binding = value

    def to(self, device: Union[torch.device, str, int]):
        """
        Changes the ONNX Runtime provider according to the device.

        Args:
            device (`torch.device` or `str` or `int`):
                Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run
                the model on the associated CUDA device id. You can pass native `torch.device` or a `str` too.

        Returns:
            `ORTModel`: the model placed on the requested device.
        """

        device, provider_options = parse_device(device)

        if device.type == "cuda" and self.providers[0] == "TensorrtExecutionProvider":
            return self

        provider = get_provider_for_device(device)
        validate_provider_availability(provider)  # raise error if the provider is not available

        # IOBinding is only supported for CPU and CUDA Execution Providers.
        if device.type == "cuda" and self._use_io_binding is False and provider == "CUDAExecutionProvider":
            self.use_io_binding = True
            logger.info(
                "use_io_binding was set to False, setting it to True because it can provide a huge speedup on GPUs. "
                "It is possible to disable this feature manually by setting the use_io_binding attribute back to False."
            )

        if provider == "ROCMExecutionProvider":
            self.use_io_binding = False

        self.model.set_providers([provider], provider_options=[provider_options])
        self.providers = self.model.get_providers()
        self._device = device

        return self

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def load_model(
        path: Union[str, Path],
        provider: str = "CPUExecutionProvider",
        session_options: Optional[ort.SessionOptions] = None,
        provider_options: Optional[Dict[str, Any]] = None,
    ) -> ort.InferenceSession:
        """
        Loads an ONNX Inference session with a given provider. Default provider is `CPUExecutionProvider` to match the
        default behaviour in PyTorch/TensorFlow/JAX.

        Args:
            path (`Union[str, Path]`):
                Path of the ONNX model.
            provider (`str`, defaults to `"CPUExecutionProvider"`):
                ONNX Runtime provider to use for loading the model. See https://onnxruntime.ai/docs/execution-providers/
                for possible providers.
            session_options (`Optional[onnxruntime.SessionOptions]`, defaults to `None`):
                ONNX Runtime session options to use for loading the model.
            provider_options (`Optional[Dict[str, Any]]`, defaults to `None`):
                Provider option dictionary corresponding to the provider used. See available options
                for each provider: https://onnxruntime.ai/docs/api/c/group___global.html .
        """
        validate_provider_availability(provider)  # raise error if the provider is not available

        providers = [provider]
        if provider == "TensorrtExecutionProvider":
            # Follow advice in https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#python
            providers.append("CUDAExecutionProvider")

        if not isinstance(path, str):
            path = str(path)

        # `providers` and `provider_options` need to be of the same length
        if provider_options is not None:
            providers_options = [provider_options] + [{} for _ in range(len(providers) - 1)]
        else:
            providers_options = None

        return ort.InferenceSession(
            path,
            providers=providers,
            sess_options=session_options,
            provider_options=providers_options,
        )

    def _save_pretrained(self, save_directory: Union[str, Path]):
        """
        Saves a model and its configuration file to a directory, so that it can be re-loaded using the
        [`~optimum.onnxruntime.modeling_ort.ORTModel.from_pretrained`] class method. It will always save the
        file under model_save_dir/latest_model_name.

        Args:
            save_directory (`Union[str, Path]`):
                Directory where to save the model file.
        """
        src_paths = [self.model_path]
        dst_paths = [Path(save_directory) / self.model_path.name]

        # add external data paths in case of large models
        src_paths, dst_paths = _get_external_data_paths(src_paths, dst_paths)

        for src_path, dst_path in zip(src_paths, dst_paths):
            shutil.copyfile(src_path, dst_path)

    @staticmethod
    def _generate_regular_names_for_filename(filename: str):
        name, extension = filename.rsplit(".", maxsplit=1)
        return [filename, f"{name}_quantized.{extension}", f"{name}_optimized.{extension}"]

    @staticmethod
    def infer_onnx_filename(
        model_name_or_path: Union[str, Path],
        patterns: List[str],
        argument_name: str,
        subfolder: str = "",
        token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        fail_if_not_found: bool = True,
    ) -> str:
        onnx_files = []
        for pattern in patterns:
            onnx_files = find_files_matching_pattern(
                model_name_or_path,
                pattern,
                glob_pattern="**/*.onnx",
                subfolder=subfolder,
                token=token,
                revision=revision,
            )
            if onnx_files:
                break

        path = model_name_or_path
        if subfolder != "":
            path = f"{path}/{subfolder}"

        if len(onnx_files) == 0:
            if fail_if_not_found:
                raise FileNotFoundError(f"Could not find any ONNX model file for the regex {patterns} in {path}.")
            return None
        elif len(onnx_files) > 1:
            if argument_name is not None:
                raise RuntimeError(
                    f"Too many ONNX model files were found in {path}, specify which one to load by using the "
                    f"{argument_name} argument."
                )
        return onnx_files[0]

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: "PretrainedConfig",
        token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        file_name: Optional[str] = None,
        subfolder: str = "",
        local_files_only: bool = False,
        provider: str = "CPUExecutionProvider",
        session_options: Optional[ort.SessionOptions] = None,
        provider_options: Optional[Dict[str, Any]] = None,
        use_io_binding: Optional[bool] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        **kwargs,
    ) -> "ORTModel":
        defaut_file_name = file_name or "model.onnx"

        onnx_files = find_files_matching_pattern(
            model_id,
            ONNX_FILE_PATTERN,
            glob_pattern="**/*.onnx",
            subfolder=subfolder,
            token=token,
            revision=revision,
        )

        if len(onnx_files) == 0:
            raise FileNotFoundError(f"Could not find any ONNX model file in {model_id}")
        if len(onnx_files) == 1 and file_name and file_name != onnx_files[0].name:
            raise FileNotFoundError(f"Trying to load {file_name} but only found {onnx_files[0].name}")

        file_name = onnx_files[0].name
        subfolder = onnx_files[0].parent

        if len(onnx_files) > 1:
            for file in onnx_files:
                if file.name == defaut_file_name:
                    file_name = file.name
                    subfolder = file.parent
                    break

            logger.warning(
                f"Too many ONNX model files were found in {' ,'.join(map(str, onnx_files))}. "
                "specify which one to load by using the `file_name` and/or the `subfolder` arguments. "
                f"Loading the file {file_name} in the subfolder {subfolder}."
            )

        if os.path.isdir(model_id):
            model_id = subfolder
            subfolder = ""

        if isinstance(subfolder, Path):
            subfolder = subfolder.as_posix()

        model_cache_path = cached_file(
            model_id,
            filename=file_name,
            # hub options
            token=token,
            revision=revision,
            subfolder=subfolder,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
        )
        new_model_save_dir = Path(model_cache_path).parent

        try:
            cached_file(
                model_id,
                filename=file_name + "_data",
                # hub options
                token=token,
                revision=revision,
                subfolder=subfolder,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
            )
        except EnvironmentError:
            # If the external data file is not found, we assume that the model is not using external data.
            pass

        # model_save_dir can be provided in kwargs as a TemporaryDirectory instance, in which case we want to keep it
        # instead of the path only.
        if model_save_dir is None:
            model_save_dir = new_model_save_dir

        model = ORTModel.load_model(
            model_cache_path,
            provider=provider,
            session_options=session_options,
            provider_options=provider_options,
        )

        return cls(
            model=model,
            config=config,
            use_io_binding=use_io_binding,
            model_save_dir=model_save_dir,
        )

    @classmethod
    def _from_transformers(
        cls,
        model_id: str,
        config: "PretrainedConfig",
        token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        subfolder: str = "",
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        provider: str = "CPUExecutionProvider",
        session_options: Optional[ort.SessionOptions] = None,
        provider_options: Optional[Dict[str, Any]] = None,
        use_io_binding: Optional[bool] = None,
        task: Optional[str] = None,
    ) -> "ORTModel":
        """The method will be deprecated in future releases."""

        return cls._export(
            model_id=model_id,
            config=config,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            token=token,
            subfolder=subfolder,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            provider=provider,
            session_options=session_options,
            provider_options=provider_options,
            use_io_binding=use_io_binding,
            task=task,
        )

    @classmethod
    def _export(
        cls,
        model_id: str,
        config: "PretrainedConfig",
        token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        subfolder: str = "",
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        provider: str = "CPUExecutionProvider",
        session_options: Optional[ort.SessionOptions] = None,
        provider_options: Optional[Dict[str, Any]] = None,
        use_io_binding: Optional[bool] = None,
        task: Optional[str] = None,
    ) -> "ORTModel":
        if task is None:
            task = cls._auto_model_to_task(cls.auto_model_class)

        save_dir = TemporaryDirectory()
        save_dir_path = Path(save_dir.name)

        main_export(
            model_name_or_path=model_id,
            output=save_dir_path,
            task=task,
            do_validation=False,
            no_post_process=True,
            subfolder=subfolder,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            force_download=force_download,
            trust_remote_code=trust_remote_code,
        )
        maybe_save_preprocessors(model_id, save_dir_path, src_subfolder=subfolder)

        return cls._from_pretrained(
            save_dir_path,
            config,
            use_io_binding=use_io_binding,
            model_save_dir=save_dir,
            provider=provider,
            session_options=session_options,
            provider_options=provider_options,
        )

    @classmethod
    @add_start_docstrings(FROM_PRETRAINED_START_DOCSTRING)
    def from_pretrained(
        cls,
        model_id: Union[str, Path],
        export: bool = False,
        force_download: bool = False,
        use_auth_token: Optional[Union[bool, str]] = None,
        token: Optional[Union[bool, str]] = None,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        subfolder: str = "",
        config: Optional["PretrainedConfig"] = None,
        local_files_only: bool = False,
        revision: Optional[str] = None,
        provider: str = "CPUExecutionProvider",
        session_options: Optional[ort.SessionOptions] = None,
        provider_options: Optional[Dict[str, Any]] = None,
        use_io_binding: Optional[bool] = None,
        **kwargs,
    ):
        """
        provider (`str`, defaults to `"CPUExecutionProvider"`):
            ONNX Runtime provider to use for loading the model. See https://onnxruntime.ai/docs/execution-providers/ for
            possible providers.
        session_options (`Optional[onnxruntime.SessionOptions]`, defaults to `None`),:
            ONNX Runtime session options to use for loading the model.
        provider_options (`Optional[Dict[str, Any]]`, defaults to `None`):
            Provider option dictionaries corresponding to the provider used. See available options
            for each provider: https://onnxruntime.ai/docs/api/c/group___global.html .
        use_io_binding (`Optional[bool]`, defaults to `None`):
            Whether to use IOBinding during inference to avoid memory copy between the host and device, or between numpy/torch tensors and ONNX Runtime ORTValue. Defaults to
            `True` if the execution provider is CUDAExecutionProvider. For [~onnxruntime.ORTModelForCausalLM], defaults to `True` on CPUExecutionProvider,
            in all other cases defaults to `False`.
        kwargs (`Dict[str, Any]`):
            Will be passed to the underlying model loading methods.

        > Parameters for decoder models (ORTModelForCausalLM, ORTModelForSeq2SeqLM, ORTModelForSeq2SeqLM, ORTModelForSpeechSeq2Seq, ORTModelForVision2Seq)

        use_cache (`Optional[bool]`, defaults to `True`):
            Whether or not past key/values cache should be used. Defaults to `True`.

        > Parameters for ORTModelForCausalLM

        use_merged (`Optional[bool]`, defaults to `None`):
            whether or not to use a single ONNX that handles both the decoding without and with past key values reuse. This option defaults
            to `True` if loading from a local repository and a merged decoder is found. When exporting with `export=True`,
            defaults to `False`. This option should be set to `True` to minimize memory usage.

        Returns:
            `ORTModel`: The loaded ORTModel model.
        """

        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed soon. Please use the `token` argument instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError("You cannot use both `use_auth_token` and `token` arguments at the same time.")
            token = use_auth_token

        if isinstance(model_id, Path):
            model_id = model_id.as_posix()

        if is_offline_mode() and not local_files_only:
            logger.info("Offline mode: setting `local_files_only=True`")
            local_files_only = True

        _export = export
        try:
            if local_files_only and not os.path.isdir(model_id):
                object_id = model_id.replace("/", "--")
                cached_model_dir = os.path.join(cache_dir, f"models--{object_id}")
                refs_file = os.path.join(os.path.join(cached_model_dir, "refs"), revision or "main")
                with open(refs_file) as f:
                    _revision = f.read()
                model_id = os.path.join(cached_model_dir, "snapshots", _revision)

            onnx_files = find_files_matching_pattern(
                model_id,
                pattern=ONNX_FILE_PATTERN,
                glob_pattern="**/*.onnx",
                subfolder=subfolder,
                token=token,
                revision=revision,
            )

            _export = len(onnx_files) == 0
            if _export ^ export:
                if export:
                    logger.warning(
                        f"The model {model_id} was already converted to ONNX but got `export=True`, the model will be converted to ONNX once again. "
                        "Don't forget to save the resulting model with `.save_pretrained()`"
                    )
                    _export = True
                else:
                    logger.warning(
                        f"No ONNX files were found for {model_id}, setting `export=True` to convert the model to ONNX. "
                        "Don't forget to save the resulting model with `.save_pretrained()`"
                    )
        except Exception as exception:
            logger.warning(
                f"Could not infer whether the model was already converted or not to ONNX, keeping `export={export}`.\n{exception}"
            )

        if _export:
            file_name = kwargs.pop("file_name", None)
            if file_name is not None:
                logger.warning(
                    f"`file_name` was set to `{file_name}` but will be ignored as the model will be converted to ONNX"
                )

        return super().from_pretrained(
            model_id,
            export=_export,
            force_download=force_download,
            token=token,
            cache_dir=cache_dir,
            subfolder=subfolder,
            config=config,
            local_files_only=local_files_only,
            revision=revision,
            provider=provider,
            session_options=session_options,
            provider_options=provider_options,
            use_io_binding=use_io_binding,
            **kwargs,
        )

    def _prepare_output_buffer(self, model: ort.InferenceSession, output_shape: Tuple[int], output_name: str):
        """Prepares the buffer of output_name with a 1D tensor."""
        ort_type = TypeHelper.get_output_type(model, output_name)
        torch_type = TypeHelper.ort_type_to_torch_type(ort_type)
        if len(output_shape) > 0:
            output_buffer = torch.empty(np.prod(output_shape), dtype=torch_type, device=self.device).contiguous()
        else:
            # Case when the output is a scalar
            output_buffer = torch.tensor(0, dtype=torch_type, device=self.device).contiguous()
        return output_buffer

    def _output_shape_inference(self, axis_name: Union[str, int], dimensions: Dict[str, int]) -> Union[str, int]:
        """
        Infers the output shape of a given dynamic axis by using the `dimensions` mapping.

        For instance, for the following inputs:
            axis_name = "past_sequence_length + sequence_length"
            dimensions = {"batch_size": 2, "sequence_length": 3, "past_sequence_length": 7}

        The inferred shape is 3 + 7 = 10.
        """
        if isinstance(axis_name, int):
            return axis_name

        elif axis_name in dimensions:
            return dimensions[axis_name]

        # faster way to do the same thing, assuming the axis names are well defined (by us in the exporter config)
        tokens = axis_name.split(" ")
        for idx, token in enumerate(tokens):
            if token in dimensions:
                tokens[idx] = str(dimensions[token])

        return int(eval(" ".join(tokens)))

    # TODO: this method is bloated with state arguments (that are accesible using self) why ?
    def _prepare_io_binding(
        self,
        model: ort.InferenceSession,
        model_inputs: Dict[str, torch.Tensor],
        known_output_shapes: Optional[Dict[str, Tuple[int]]] = None,
        outputs_to_not_bind: Optional[Union[Set[str], str]] = None,
    ) -> Tuple[ort.IOBinding, Dict[str, Tuple[int]], Dict[str, torch.Tensor]]:
        """
        Prepares IO binding for ONNX Runtime.

        Args:
            model (`ort.InferenceSession`):
                The model for which we want to bind the inputs and outputs.
            model_inputs (`Dict[str, torch.Tensor]`):
                The inputs to bind to the model.
            known_output_shapes (`Optional[Dict[str, Tuple[int]]]`, defaults to `None`):
                It can be hard to infer all the output shapes from the inputs only. For instance for the past key /
                values. It is possible to explicitely pass the shape via this argument.
            outputs_to_not_bind (`Optional[Union[Set[str], str]]`, defaults to `None`):
                The names of the outputs that should not be bound.

        Returns:
            `Tuple[ort.IOBinding, Dict[str, Tuple[int]], Dict[str, torch.Tensor]`: The IOBinding object, a dictionary
            containing the shape of each output, and another one pointing to the buffers containing the outputs data.
        """
        io_binding = model.io_binding()

        input_shapes = {}
        for input_name in self.input_names.keys():
            input_shapes[input_name] = model_inputs[input_name].shape

            if not model_inputs[input_name].is_contiguous():
                model_inputs[input_name] = model_inputs[input_name].contiguous()

            tensor_dtype = model_inputs[input_name].dtype
            expected_dtype = TypeHelper.ort_type_to_torch_type(self.input_dtypes[input_name])
            if tensor_dtype != expected_dtype:
                model_inputs[input_name] = model_inputs[input_name].to(expected_dtype)

            data_ptr = model_inputs[input_name].data_ptr()
            if data_ptr == 0:
                # During first generation, sequence_length can be 0 when use_cache=True, which results in data_ptr to also be 0.
                # To keep compatibility with IO binding, we pass the data pointer of input_ids instead. This will have no impact because past_key_values will not be used during the first generation.
                data_ptr = model_inputs["input_ids"].data_ptr()

            io_binding.bind_input(
                input_name,
                self.device.type,
                IOBindingHelper.get_device_index(self.device),
                TypeHelper.ort_type_to_numpy_type(self.input_dtypes[input_name]),
                model_inputs[input_name].shape,
                data_ptr,
            )

        dimensions = {}
        for input_ in model.get_inputs():
            shape = input_.shape
            for idx, axis in enumerate(shape):
                if isinstance(axis, str):
                    dimensions[axis] = input_shapes[input_.name][idx]

        output_shapes = {}
        output_buffers = {}

        if known_output_shapes is None:
            known_output_shapes = {}

        if outputs_to_not_bind is None:
            outputs_to_not_bind = set()
        elif isinstance(outputs_to_not_bind, str):
            outputs_to_not_bind = {outputs_to_not_bind}

        for output_node in model.get_outputs():
            output_name = output_node.name
            if output_name in outputs_to_not_bind:
                continue
            if output_name in known_output_shapes:
                output_shape = known_output_shapes[output_name]
            else:
                output_shape = []
                for axis_name in output_node.shape:
                    output_shape.append(self._output_shape_inference(axis_name, dimensions))

            output_buffer = self._prepare_output_buffer(model, output_shape, output_name)

            data_ptr = output_buffer.data_ptr()

            io_binding.bind_output(
                output_name,
                self.device.type,
                IOBindingHelper.get_device_index(self.device),
                TypeHelper.ort_type_to_numpy_type(output_node.type),
                output_shape,
                data_ptr,
            )

            output_buffers[output_name] = output_buffer
            output_shapes[output_name] = output_shape

        return io_binding, output_shapes, output_buffers

    def raise_on_numpy_input_io_binding(self, use_torch: bool):
        """
        Raises an error if IO Binding is requested although the tensor used are numpy arrays.

        Args:
            use_torch (`bool`):
                Whether the tensor used during inference are of type torch.Tensor or not.
        """
        if use_torch is False and self.use_io_binding is True:
            raise ValueError(
                "IO Binding can not be used when passing numpy inputs. Please disable IO Binding"
                " with model.use_io_binding = False, or pass torch.Tensor inputs instead."
            )

    def _prepare_onnx_inputs(
        self, use_torch: bool, model_inputs: Dict[str, Union[torch.Tensor, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """
        Prepares the inputs for ONNX Runtime by converting them to numpy arrays with the expected dtype.

        Args:
            use_torch (`bool`):
                Whether the inputs are torch.Tensor or not.
            inputs (`Dict[str, Union[torch.Tensor, np.ndarray]]`):
                The inputs to prepare for ONNX Runtime.

        Returns:
            `Dict[str, np.ndarray]`: The inputs prepared for ONNX Runtime.
        """

        onnx_inputs = {}

        for input_name in self.input_names.keys():
            if model_inputs.get(input_name, None) is None:
                raise ValueError(f"Input {input_name} is required by model but not provided.")

            if use_torch:
                onnx_inputs[input_name] = model_inputs[input_name].numpy(force=True)
            else:
                onnx_inputs[input_name] = model_inputs[input_name]

            expected_dtype = TypeHelper.ort_type_to_numpy_type(self.input_dtypes[input_name])

            if onnx_inputs[input_name].dtype != expected_dtype:
                onnx_inputs[input_name] = onnx_inputs[input_name].astype(expected_dtype)

        return onnx_inputs

    def _prepare_onnx_outputs(
        self, use_torch: bool, onnx_outputs: List[np.ndarray]
    ) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        """
        Prepares the outputs from ONNX Runtime by converting them to torch.Tensor if requested.

        Args:
            use_torch (`bool`):
                Whether the outputs should be torch.Tensor or not.
            onnx_outputs (`List[np.ndarray]`):
                The outputs from ONNX Runtime.

        Returns:
            `Dict[str, Union[torch.Tensor, np.ndarray]]`: The outputs prepared for the user.
        """

        model_outputs = {}

        for output_name, idx in self.output_names.items():
            model_outputs[output_name] = onnx_outputs[idx]

            if use_torch:
                model_outputs[output_name] = torch.from_numpy(model_outputs[output_name]).to(self.device)

        return model_outputs

    def can_generate(self) -> bool:
        """
        Returns whether this model can generate sequences with `.generate()`.
        """
        return isinstance(self, GenerationMixin)

    def _warn_on_unhandled_inputs(self, kwargs: Dict[str, Any]) -> None:
        """Warn about unhandled input arguments.

        Args:
            kwargs: Dictionary of unhandled input arguments.
        """
        if kwargs:
            logger.warning(
                f"{self.__class__.__name__} received {', '.join(kwargs.keys())}, but do not handle those arguments. "
                "Please use `ORTModelForCustomTasks` if your model takes/returns arbitrary or custom tensor inputs/outputs. "
                "Or open an issue/PR in optimum repository (https://github.com/huggingface/optimum) if this argument needs to be supported in this class."
            )


FEATURE_EXTRACTION_EXAMPLE = r"""
    Example of feature extraction:

    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.onnxruntime import {model_class}
    >>> import torch

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("My name is Philipp and I live in Germany.", return_tensors="np")

    >>> outputs = model(**inputs)
    >>> last_hidden_state = outputs.last_hidden_state
    >>> list(last_hidden_state.shape)
    [1, 12, 384]
    ```

    Example using `transformers.pipeline`:

    ```python
    >>> from transformers import {processor_class}, pipeline
    >>> from optimum.onnxruntime import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> onnx_extractor = pipeline("feature-extraction", model=model, tokenizer=tokenizer)

    >>> text = "My name is Philipp and I live in Germany."
    >>> pred = onnx_extractor(text)
    ```
"""


@add_end_docstrings(ONNX_MODEL_END_DOCSTRING)
class ORTModelForFeatureExtraction(ORTModel):
    """
    ONNX Model for feature-extraction task.
    """

    auto_model_class = AutoModel

    @add_start_docstrings_to_model_forward(
        ONNX_TEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + FEATURE_EXTRACTION_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="ORTModelForFeatureExtraction",
            checkpoint="optimum/all-MiniLM-L6-v2",
        )
    )
    def forward(
        self,
        input_ids: Optional[Union[torch.Tensor, np.ndarray]] = None,
        attention_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
        token_type_ids: Optional[Union[torch.Tensor, np.ndarray]] = None,
        position_ids: Optional[Union[torch.Tensor, np.ndarray]] = None,
        pixel_values: Optional[Union[torch.Tensor, np.ndarray]] = None,
        input_features: Optional[Union[torch.Tensor, np.ndarray]] = None,
        input_values: Optional[Union[torch.Tensor, np.ndarray]] = None,
        *,
        return_dict: bool = False,
        **kwargs,
    ):
        # Warn about any unexpected kwargs using the helper method
        self._warn_on_unhandled_inputs(kwargs)

        # Determine the tensor type from any available tensor input
        tensor_inputs = [
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            pixel_values,
            input_features,
            input_values,
        ]
        first_tensor = next(filter(lambda x: x is not None, tensor_inputs))
        use_torch = isinstance(first_tensor, torch.Tensor)
        self.raise_on_numpy_input_io_binding(use_torch)

        if token_type_ids is None and "token_type_ids" in self.input_names:
            token_type_ids = torch.zeros_like(input_ids) if use_torch else np.zeros_like(input_ids)

        # Build model_inputs dictionary
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "position_ids": position_ids,
            "pixel_values": pixel_values,
            "input_features": input_features,
            "input_values": input_values,
        }

        if self.use_io_binding:
            io_binding, output_shapes, output_buffers = self._prepare_io_binding(self.model, model_inputs)

            # run inference with binding & synchronize in case of multiple CUDA streams
            if self.device.type == "cpu":
                self.model.run_with_iobinding(io_binding)
            else:
                io_binding.synchronize_inputs()
                self.model.run_with_iobinding(io_binding)
                io_binding.synchronize_outputs()

            last_hidden_state = output_buffers["last_hidden_state"].view(output_shapes["last_hidden_state"])
        else:
            onnx_inputs = self._prepare_onnx_inputs(use_torch, model_inputs)
            onnx_outputs = self.model.run(None, onnx_inputs)
            model_outputs = self._prepare_onnx_outputs(use_torch, onnx_outputs)

            if "last_hidden_state" in self.output_names:
                last_hidden_state = model_outputs["last_hidden_state"]
            else:
                # TODO: This allows to support sentence-transformers models (sentence embedding), but is not validated.
                last_hidden_state = next(iter(model_outputs.values()))

        if return_dict:
            return {"last_hidden_state": last_hidden_state}

        # converts output to namedtuple for pipelines post-processing
        return BaseModelOutput(last_hidden_state=last_hidden_state)

    @classmethod
    def _export(
        cls,
        model_id: str,
        config: "PretrainedConfig",
        token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        subfolder: str = "",
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        provider: str = "CPUExecutionProvider",
        session_options: Optional[ort.SessionOptions] = None,
        provider_options: Optional[Dict[str, Any]] = None,
        use_io_binding: Optional[bool] = None,
        task: Optional[str] = None,
    ) -> "ORTModel":
        if task is None:
            task = cls._auto_model_to_task(cls.auto_model_class)

        save_dir = TemporaryDirectory()
        save_dir_path = Path(save_dir.name)

        # ORTModelForFeatureExtraction works with Transformers type of models, thus even sentence-transformers models are loaded as such.
        main_export(
            model_name_or_path=model_id,
            output=save_dir_path,
            task=task,
            do_validation=False,
            no_post_process=True,
            subfolder=subfolder,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            force_download=force_download,
            trust_remote_code=trust_remote_code,
            library_name="transformers",
        )

        maybe_save_preprocessors(model_id, save_dir_path, src_subfolder=subfolder)

        return cls._from_pretrained(
            save_dir_path,
            config,
            use_io_binding=use_io_binding,
            model_save_dir=save_dir,
            provider=provider,
            session_options=session_options,
            provider_options=provider_options,
        )


MASKED_LM_EXAMPLE = r"""
    Example of feature extraction:

    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.onnxruntime import {model_class}
    >>> import torch

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("The capital of France is [MASK].", return_tensors="np")

    >>> outputs = model(**inputs)
    >>> logits = outputs.logits
    >>> list(logits.shape)
    [1, 8, 28996]
    ```

    Example using `transformers.pipeline`:

    ```python
    >>> from transformers import {processor_class}, pipeline
    >>> from optimum.onnxruntime import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> fill_masker = pipeline("fill-mask", model=model, tokenizer=tokenizer)

    >>> text = "The capital of France is [MASK]."
    >>> pred = fill_masker(text)
    ```
"""


@add_end_docstrings(ONNX_MODEL_END_DOCSTRING)
class ORTModelForMaskedLM(ORTModel):
    """
    ONNX Model with a MaskedLMOutput for masked language modeling tasks. This class officially supports albert, bert, camembert, convbert, data2vec_text, deberta, deberta_v2, distilbert, electra, flaubert, ibert, mobilebert, roberta, roformer, squeezebert, xlm, xlm_roberta.
    """

    auto_model_class = AutoModelForMaskedLM

    @add_start_docstrings_to_model_forward(
        ONNX_TEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + MASKED_LM_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="ORTModelForMaskedLM",
            checkpoint="optimum/bert-base-uncased-for-fill-mask",
        )
    )
    def forward(
        self,
        input_ids: Optional[Union[torch.Tensor, np.ndarray]] = None,
        attention_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
        token_type_ids: Optional[Union[torch.Tensor, np.ndarray]] = None,
        *,
        return_dict: bool = False,
        **kwargs,
    ):
        # Warn about any unexpected kwargs using the helper method
        self._warn_on_unhandled_inputs(kwargs)

        use_torch = isinstance(input_ids, torch.Tensor)
        self.raise_on_numpy_input_io_binding(use_torch)

        if "token_type_ids" in self.input_names and token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids) if use_torch else np.zeros_like(input_ids)

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

        if self.use_io_binding:
            io_binding, output_shapes, output_buffers = self._prepare_io_binding(self.model, model_inputs)

            # run inference with binding & synchronize in case of multiple CUDA streams
            if self.device.type == "cpu":
                self.model.run_with_iobinding(io_binding)
            else:
                io_binding.synchronize_inputs()
                self.model.run_with_iobinding(io_binding)
                io_binding.synchronize_outputs()

            logits = output_buffers["logits"].view(output_shapes["logits"])
        else:
            onnx_inputs = self._prepare_onnx_inputs(use_torch, model_inputs)
            onnx_outputs = self.model.run(None, onnx_inputs)
            model_outputs = self._prepare_onnx_outputs(use_torch, onnx_outputs)

            logits = model_outputs["logits"]

        if return_dict:
            return {"logits": logits}

        # converts output to namedtuple for pipelines post-processing
        return MaskedLMOutput(logits=logits)


QUESTION_ANSWERING_EXAMPLE = r"""
    Example of question answering:

    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.onnxruntime import {model_class}
    >>> import torch

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
    >>> inputs = tokenizer(question, text, return_tensors="np")
    >>> start_positions = torch.tensor([1])
    >>> end_positions = torch.tensor([3])

    >>> outputs = model(**inputs, start_positions=start_positions, end_positions=end_positions)
    >>> start_scores = outputs.start_logits
    >>> end_scores = outputs.end_logits
    ```
    Example using `transformers.pipeline`:

    ```python
    >>> from transformers import {processor_class}, pipeline
    >>> from optimum.onnxruntime import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> onnx_qa = pipeline("question-answering", model=model, tokenizer=tokenizer)

    >>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
    >>> pred = onnx_qa(question, text)
    ```
"""


@add_end_docstrings(ONNX_MODEL_END_DOCSTRING)
class ORTModelForQuestionAnswering(ORTModel):
    """
    ONNX Model with a QuestionAnsweringModelOutput for extractive question-answering tasks like SQuAD. This class officially supports albert, bart, bert, camembert, convbert, data2vec_text, deberta, deberta_v2, distilbert, electra, flaubert, gptj, ibert, mbart, mobilebert, nystromformer, roberta, roformer, squeezebert, xlm, xlm_roberta.
    """

    auto_model_class = AutoModelForQuestionAnswering

    @add_start_docstrings_to_model_forward(
        ONNX_TEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + QUESTION_ANSWERING_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="ORTModelForQuestionAnswering",
            checkpoint="optimum/roberta-base-squad2",
        )
    )
    def forward(
        self,
        input_ids: Optional[Union[torch.Tensor, np.ndarray]] = None,
        attention_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
        token_type_ids: Optional[Union[torch.Tensor, np.ndarray]] = None,
        *,
        return_dict: bool = False,
        **kwargs,
    ):
        # Warn about any unexpected kwargs using the helper method
        self._warn_on_unhandled_inputs(kwargs)

        use_torch = isinstance(input_ids, torch.Tensor)
        self.raise_on_numpy_input_io_binding(use_torch)

        if token_type_ids is None and "token_type_ids" in self.input_names:
            token_type_ids = torch.zeros_like(input_ids) if use_torch else np.zeros_like(input_ids)

        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}

        if self.use_io_binding:
            io_binding, output_shapes, output_buffers = self._prepare_io_binding(self.model, model_inputs)

            # run inference with binding & synchronize in case of multiple CUDA streams
            if self.device.type == "cpu":
                self.model.run_with_iobinding(io_binding)
            else:
                io_binding.synchronize_inputs()
                self.model.run_with_iobinding(io_binding)
                io_binding.synchronize_outputs()

            start_logits = output_buffers["start_logits"].view(output_shapes["start_logits"])
            end_logits = output_buffers["end_logits"].view(output_shapes["end_logits"])
        else:
            onnx_inputs = self._prepare_onnx_inputs(use_torch, model_inputs)
            onnx_outputs = self.model.run(None, onnx_inputs)
            model_outputs = self._prepare_onnx_outputs(use_torch, onnx_outputs)

            start_logits = model_outputs["start_logits"]
            end_logits = model_outputs["end_logits"]

        if return_dict:
            return {"start_logits": start_logits, "end_logits": end_logits}

        # converts output to namedtuple for pipelines post-processing
        return QuestionAnsweringModelOutput(start_logits=start_logits, end_logits=end_logits)


SEQUENCE_CLASSIFICATION_EXAMPLE = r"""
    Example of single-label classification:

    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.onnxruntime import {model_class}
    >>> import torch

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="np")

    >>> outputs = model(**inputs)
    >>> logits = outputs.logits
    >>> list(logits.shape)
    [1, 2]
    ```

    Example using `transformers.pipelines`:

    ```python
    >>> from transformers import {processor_class}, pipeline
    >>> from optimum.onnxruntime import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> onnx_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

    >>> text = "Hello, my dog is cute"
    >>> pred = onnx_classifier(text)
    ```

    Example using zero-shot-classification `transformers.pipelines`:

    ```python
    >>> from transformers import {processor_class}, pipeline
    >>> from optimum.onnxruntime import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("optimum/distilbert-base-uncased-mnli")
    >>> model = {model_class}.from_pretrained("optimum/distilbert-base-uncased-mnli")
    >>> onnx_z0 = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)

    >>> sequence_to_classify = "Who are you voting for in 2020?"
    >>> candidate_labels = ["Europe", "public health", "politics", "elections"]
    >>> pred = onnx_z0(sequence_to_classify, candidate_labels, multi_label=True)
    ```
"""


@add_end_docstrings(ONNX_MODEL_END_DOCSTRING)
class ORTModelForSequenceClassification(ORTModel):
    """
    ONNX Model with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks. This class officially supports albert, bart, bert, camembert, convbert, data2vec_text, deberta, deberta_v2, distilbert, electra, flaubert, ibert, mbart, mobilebert, nystromformer, roberta, roformer, squeezebert, xlm, xlm_roberta.
    """

    auto_model_class = AutoModelForSequenceClassification

    @add_start_docstrings_to_model_forward(
        ONNX_TEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + SEQUENCE_CLASSIFICATION_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="ORTModelForSequenceClassification",
            checkpoint="optimum/distilbert-base-uncased-finetuned-sst-2-english",
        )
    )
    def forward(
        self,
        input_ids: Optional[Union[torch.Tensor, np.ndarray]] = None,
        attention_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
        token_type_ids: Optional[Union[torch.Tensor, np.ndarray]] = None,
        *,
        return_dict: bool = False,
        **kwargs,
    ):
        # Warn about any unexpected kwargs using the helper method
        self._warn_on_unhandled_inputs(kwargs)

        use_torch = isinstance(input_ids, torch.Tensor)
        self.raise_on_numpy_input_io_binding(use_torch)

        if token_type_ids is None and "token_type_ids" in self.input_names:
            token_type_ids = torch.zeros_like(input_ids) if use_torch else np.zeros_like(input_ids)

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

        if self.use_io_binding:
            io_binding, output_shapes, output_buffers = self._prepare_io_binding(self.model, model_inputs)

            # run inference with binding & synchronize in case of multiple CUDA streams
            if self.device.type == "cpu":
                self.model.run_with_iobinding(io_binding)
            else:
                io_binding.synchronize_inputs()
                self.model.run_with_iobinding(io_binding)
                io_binding.synchronize_outputs()

            logits = output_buffers["logits"].view(output_shapes["logits"])
        else:
            onnx_inputs = self._prepare_onnx_inputs(use_torch, model_inputs)
            onnx_outputs = self.model.run(None, onnx_inputs)
            model_outputs = self._prepare_onnx_outputs(use_torch, onnx_outputs)

            logits = model_outputs["logits"]

        if return_dict:
            return {"logits": logits}

        # converts output to namedtuple for pipelines post-processing
        return SequenceClassifierOutput(logits=logits)


TOKEN_CLASSIFICATION_EXAMPLE = r"""
    Example of token classification:

    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.onnxruntime import {model_class}
    >>> import torch

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("My name is Philipp and I live in Germany.", return_tensors="np")

    >>> outputs = model(**inputs)
    >>> logits = outputs.logits
    >>> list(logits.shape)
    [1, 12, 9]
    ```

    Example using `transformers.pipelines`:

    ```python
    >>> from transformers import {processor_class}, pipeline
    >>> from optimum.onnxruntime import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> onnx_ner = pipeline("token-classification", model=model, tokenizer=tokenizer)

    >>> text = "My name is Philipp and I live in Germany."
    >>> pred = onnx_ner(text)
    ```
"""


@add_end_docstrings(ONNX_MODEL_END_DOCSTRING)
class ORTModelForTokenClassification(ORTModel):
    """
    ONNX Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
    for Named-Entity-Recognition (NER) tasks. This class officially supports albert, bert, bloom, camembert, convbert, data2vec_text, deberta, deberta_v2, distilbert, electra, flaubert, gpt2, ibert, mobilebert, roberta, roformer, squeezebert, xlm, xlm_roberta.

    """

    auto_model_class = AutoModelForTokenClassification

    @add_start_docstrings_to_model_forward(
        ONNX_TEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + TOKEN_CLASSIFICATION_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="ORTModelForTokenClassification",
            checkpoint="optimum/bert-base-NER",
        )
    )
    def forward(
        self,
        input_ids: Optional[Union[torch.Tensor, np.ndarray]] = None,
        attention_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
        token_type_ids: Optional[Union[torch.Tensor, np.ndarray]] = None,
        *,
        return_dict: bool = False,
        **kwargs,
    ):
        # Warn about any unexpected kwargs using the helper method
        self._warn_on_unhandled_inputs(kwargs)

        use_torch = isinstance(input_ids, torch.Tensor)
        self.raise_on_numpy_input_io_binding(use_torch)

        if token_type_ids is None and "token_type_ids" in self.input_names:
            token_type_ids = torch.zeros_like(input_ids) if use_torch else np.zeros_like(input_ids)

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

        if self.use_io_binding:
            io_binding, output_shapes, output_buffers = self._prepare_io_binding(self.model, model_inputs)

            # run inference with binding & synchronize in case of multiple CUDA streams
            if self.device.type == "cpu":
                self.model.run_with_iobinding(io_binding)
            else:
                io_binding.synchronize_inputs()
                self.model.run_with_iobinding(io_binding)
                io_binding.synchronize_outputs()

            logits = output_buffers["logits"].view(output_shapes["logits"])
        else:
            onnx_inputs = self._prepare_onnx_inputs(use_torch, model_inputs)
            onnx_outputs = self.model.run(None, onnx_inputs)
            model_outputs = self._prepare_onnx_outputs(use_torch, onnx_outputs)

            logits = model_outputs["logits"]

        if return_dict:
            return {"logits": logits}

        return TokenClassifierOutput(logits=logits)


MULTIPLE_CHOICE_EXAMPLE = r"""
    Example of mutliple choice:

    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.onnxruntime import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}", export=True)

    >>> num_choices = 4
    >>> first_sentence = ["Members of the procession walk down the street holding small horn brass instruments."] * num_choices
    >>> second_sentence = [
    ...     "A drum line passes by walking down the street playing their instruments.",
    ...     "A drum line has heard approaching them.",
    ...     "A drum line arrives and they're outside dancing and asleep.",
    ...     "A drum line turns the lead singer watches the performance."
    ... ]
    >>> inputs = tokenizer(first_sentence, second_sentence, truncation=True, padding=True)

    # Unflatten the inputs values expanding it to the shape [batch_size, num_choices, seq_length]
    >>> for k, v in inputs.items():
    ...     inputs[k] = [v[i: i + num_choices] for i in range(0, len(v), num_choices)]
    >>> inputs = dict(inputs.convert_to_tensors(tensor_type="pt"))
    >>> outputs = model(**inputs)
    >>> logits = outputs.logits
    ```
"""


@add_end_docstrings(ONNX_MODEL_END_DOCSTRING)
class ORTModelForMultipleChoice(ORTModel):
    """
    ONNX Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks. This class officially supports albert, bert, camembert, convbert, data2vec_text, deberta_v2, distilbert, electra, flaubert, ibert, mobilebert, nystromformer, roberta, roformer, squeezebert, xlm, xlm_roberta.
    """

    auto_model_class = AutoModelForMultipleChoice

    @add_start_docstrings_to_model_forward(
        ONNX_TEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + MULTIPLE_CHOICE_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="ORTModelForMultipleChoice",
            checkpoint="ehdwns1516/bert-base-uncased_SWAG",
        )
    )
    def forward(
        self,
        input_ids: Optional[Union[torch.Tensor, np.ndarray]] = None,
        attention_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
        token_type_ids: Optional[Union[torch.Tensor, np.ndarray]] = None,
        *,
        return_dict: bool = False,
        **kwargs,
    ):
        # Warn about any unexpected kwargs using the helper method
        self._warn_on_unhandled_inputs(kwargs)

        use_torch = isinstance(input_ids, torch.Tensor)
        self.raise_on_numpy_input_io_binding(use_torch)

        if token_type_ids is None and "token_type_ids" in self.input_names:
            token_type_ids = torch.zeros_like(input_ids) if use_torch else np.zeros_like(input_ids)

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

        if self.use_io_binding:
            io_binding, output_shapes, output_buffers = self._prepare_io_binding(self.model, model_inputs)

            # run inference with binding & synchronize in case of multiple CUDA streams
            if self.device.type == "cpu":
                self.model.run_with_iobinding(io_binding)
            else:
                io_binding.synchronize_inputs()
                self.model.run_with_iobinding(io_binding)
                io_binding.synchronize_outputs()

            logits = output_buffers["logits"].view(output_shapes["logits"])
        else:
            onnx_inputs = self._prepare_onnx_inputs(use_torch, model_inputs)
            onnx_outputs = self.model.run(None, onnx_inputs)
            model_outputs = self._prepare_onnx_outputs(use_torch, onnx_outputs)

            logits = model_outputs["logits"]

        if return_dict:
            return {"logits": logits}

        # converts output to namedtuple for pipelines post-processing
        return MultipleChoiceModelOutput(logits=logits)


IMAGE_CLASSIFICATION_EXAMPLE = r"""
    Example of image classification:

    ```python
    >>> import requests
    >>> from PIL import Image
    >>> from optimum.onnxruntime import {model_class}
    >>> from transformers import {processor_class}

    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> preprocessor = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = preprocessor(images=image, return_tensors="np")

    >>> outputs = model(**inputs)
    >>> logits = outputs.logits
    ```

    Example using `transformers.pipeline`:

    ```python
    >>> import requests
    >>> from PIL import Image
    >>> from transformers import {processor_class}, pipeline
    >>> from optimum.onnxruntime import {model_class}

    >>> preprocessor = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> onnx_image_classifier = pipeline("image-classification", model=model, feature_extractor=preprocessor)

    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    >>> pred = onnx_image_classifier(url)
    ```
"""


@add_end_docstrings(ONNX_MODEL_END_DOCSTRING)
class ORTModelForImageClassification(ORTModel):
    """
    ONNX Model for image-classification tasks. This class officially supports beit, convnext, convnextv2, data2vec_vision, deit, dinov2, levit, mobilenet_v1, mobilenet_v2, mobilevit, poolformer, resnet, segformer, swin, swinv2, vit.
    """

    auto_model_class = AutoModelForImageClassification

    @add_start_docstrings_to_model_forward(
        ONNX_IMAGE_INPUTS_DOCSTRING.format("batch_size, num_channels, height, width")
        + IMAGE_CLASSIFICATION_EXAMPLE.format(
            processor_class=_FEATURE_EXTRACTOR_FOR_DOC,
            model_class="ORTModelForImageClassification",
            checkpoint="optimum/vit-base-patch16-224",
        )
    )
    def forward(
        self,
        pixel_values: Union[torch.Tensor, np.ndarray],
        *,
        return_dict: bool = False,
        **kwargs,
    ):
        # Warn about any unexpected kwargs using the helper method
        self._warn_on_unhandled_inputs(kwargs)

        use_torch = isinstance(pixel_values, torch.Tensor)
        self.raise_on_numpy_input_io_binding(use_torch)

        model_inputs = {
            "pixel_values": pixel_values,
        }

        if self.use_io_binding:
            io_binding, output_shapes, output_buffers = self._prepare_io_binding(self.model, model_inputs)

            # run inference with binding & synchronize in case of multiple CUDA streams
            if self.device.type == "cpu":
                self.model.run_with_iobinding(io_binding)
            else:
                io_binding.synchronize_inputs()
                self.model.run_with_iobinding(io_binding)
                io_binding.synchronize_outputs()

            logits = output_buffers["logits"].view(output_shapes["logits"])
        else:
            onnx_inputs = self._prepare_onnx_inputs(use_torch, model_inputs)
            onnx_outputs = self.model.run(None, onnx_inputs)
            model_outputs = self._prepare_onnx_outputs(use_torch, onnx_outputs)

            logits = model_outputs["logits"]

        if return_dict:
            return {"logits": logits}

        # converts output to namedtuple for pipelines post-processing
        return ImageClassifierOutput(logits=logits)


SEMANTIC_SEGMENTATION_EXAMPLE = r"""
    Example of semantic segmentation:

    ```python
    >>> import requests
    >>> from PIL import Image
    >>> from optimum.onnxruntime import {model_class}
    >>> from transformers import {processor_class}

    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> preprocessor = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = preprocessor(images=image, return_tensors="np")

    >>> outputs = model(**inputs)
    >>> logits = outputs.logits
    ```

    Example using `transformers.pipeline`:

    ```python
    >>> import requests
    >>> from PIL import Image
    >>> from transformers import {processor_class}, pipeline
    >>> from optimum.onnxruntime import {model_class}

    >>> preprocessor = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> onnx_image_segmenter = pipeline("image-segmentation", model=model, feature_extractor=preprocessor)

    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    >>> pred = onnx_image_segmenter(url)
    ```
"""


@add_end_docstrings(ONNX_MODEL_END_DOCSTRING)
class ORTModelForSemanticSegmentation(ORTModel):
    """
    ONNX Model for semantic-segmentation, with an all-MLP decode head on top e.g. for ADE20k, CityScapes. This class officially supports maskformer, segformer.
    """

    auto_model_class = AutoModelForSemanticSegmentation

    @add_start_docstrings_to_model_forward(
        ONNX_IMAGE_INPUTS_DOCSTRING.format("batch_size, num_channels, height, width")
        + SEMANTIC_SEGMENTATION_EXAMPLE.format(
            processor_class=_FEATURE_EXTRACTOR_FOR_DOC,
            model_class="ORTModelForSemanticSegmentation",
            checkpoint="optimum/segformer-b0-finetuned-ade-512-512",
        )
    )
    def forward(
        self,
        pixel_values: Union[torch.Tensor, np.ndarray],
        *,
        return_dict: bool = False,
        **kwargs,
    ):
        # Warn about any unexpected kwargs using the helper method
        self._warn_on_unhandled_inputs(kwargs)

        use_torch = isinstance(pixel_values, torch.Tensor)
        self.raise_on_numpy_input_io_binding(use_torch)

        model_inputs = {
            "pixel_values": pixel_values,
        }

        if self.use_io_binding:
            io_binding, output_shapes, output_buffers = self._prepare_io_binding(self.model, model_inputs)

            # run inference with binding & synchronize in case of multiple CUDA streams
            if self.device.type == "cpu":
                self.model.run_with_iobinding(io_binding)
            else:
                io_binding.synchronize_inputs()
                self.model.run_with_iobinding(io_binding)
                io_binding.synchronize_outputs()

            logits = output_buffers["logits"].view(output_shapes["logits"])
        else:
            onnx_inputs = self._prepare_onnx_inputs(use_torch, model_inputs)
            onnx_outputs = self.model.run(None, onnx_inputs)
            model_outputs = self._prepare_onnx_outputs(use_torch, onnx_outputs)

            logits = model_outputs["logits"]

        if return_dict:
            return {"logits": logits}

        # converts output to namedtuple for pipelines post-processing
        return SemanticSegmenterOutput(logits=logits)


AUDIO_CLASSIFICATION_EXAMPLE = r"""
    Example of audio classification:

    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.onnxruntime import {model_class}
    >>> from datasets import load_dataset
    >>> import torch

    >>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    >>> dataset = dataset.sort("id")
    >>> sampling_rate = dataset.features["audio"].sampling_rate

    >>> feature_extractor = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> # audio file is decoded on the fly
    >>> inputs = feature_extractor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")

    >>> with torch.no_grad():
    ...     logits = model(**inputs).logits

    >>> predicted_class_ids = torch.argmax(logits, dim=-1).item()
    >>> predicted_label = model.config.id2label[predicted_class_ids]
    ```
    Example using `transformers.pipeline`:

    ```python
    >>> from transformers import {processor_class}, pipeline
    >>> from optimum.onnxruntime import {model_class}

    >>> feature_extractor = {processor_class}.from_pretrained("{checkpoint}")
    >>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    >>> dataset = dataset.sort("id")

    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> onnx_ac = pipeline("audio-classification", model=model, feature_extractor=feature_extractor)

    >>> pred = onnx_ac(dataset[0]["audio"]["array"])
    ```
"""


@add_end_docstrings(ONNX_MODEL_END_DOCSTRING)
class ORTModelForAudioClassification(ORTModel):
    """
    ONNX Model for audio-classification, with a sequence classification head on top (a linear layer over the pooled output) for tasks like
    SUPERB Keyword Spotting. This class officially supports audio_spectrogram_transformer, data2vec_audio, hubert, sew, sew_d, unispeech, unispeech_sat, wavlm, wav2vec2, wav2vec2-conformer.
    """

    auto_model_class = AutoModelForAudioClassification

    def __init__(
        self,
        model: ort.InferenceSession,
        config: "PretrainedConfig",
        use_io_binding: Optional[bool] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        preprocessors: Optional[List] = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            config=config,
            use_io_binding=use_io_binding,
            model_save_dir=model_save_dir,
            preprocessors=preprocessors,
            **kwargs,
        )

        if config.model_type == "whisper":
            self.input_name = "input_features"
        else:
            self.input_name = "input_values"

    @add_start_docstrings_to_model_forward(
        ONNX_AUDIO_INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + AUDIO_CLASSIFICATION_EXAMPLE.format(
            processor_class=_FEATURE_EXTRACTOR_FOR_DOC,
            model_class="ORTModelForAudioClassification",
            checkpoint="optimum/hubert-base-superb-ks",
        )
    )
    def forward(
        self,
        input_values: Optional[Union[torch.Tensor, np.ndarray]] = None,
        attention_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
        input_features: Optional[Union[torch.Tensor, np.ndarray]] = None,
        *,
        return_dict: bool = False,
        **kwargs,
    ):
        # Warn about any unexpected kwargs using the helper method
        self._warn_on_unhandled_inputs(kwargs)

        if self.input_name == "input_features":
            assert input_features is not None, "input_features must be provided for this model"
            model_input = input_features
        elif self.input_name == "input_values":
            assert input_values is not None, "input_values must be provided for this model"
            model_input = input_values
        else:
            raise ValueError(f"Input {self.input_name} not supported for Audio Classification")

        use_torch = isinstance(model_input, torch.Tensor)
        self.raise_on_numpy_input_io_binding(use_torch)

        model_inputs = {
            self.input_name: model_input,
            "attention_mask": attention_mask,
        }

        if self.use_io_binding:
            io_binding, output_shapes, output_buffers = self._prepare_io_binding(self.model, model_inputs)

            # run inference with binding & synchronize in case of multiple CUDA streams
            if self.device.type == "cpu":
                self.model.run_with_iobinding(io_binding)
            else:
                io_binding.synchronize_inputs()
                self.model.run_with_iobinding(io_binding)
                io_binding.synchronize_outputs()

            logits = output_buffers["logits"].view(output_shapes["logits"])
        else:
            onnx_inputs = self._prepare_onnx_inputs(use_torch, model_inputs)
            onnx_outputs = self.model.run(None, onnx_inputs)
            model_outputs = self._prepare_onnx_outputs(use_torch, onnx_outputs)

            logits = model_outputs["logits"]

        if return_dict:
            return {"logits": logits}

        # converts output to namedtuple for pipelines post-processing
        return SequenceClassifierOutput(logits=logits)


CTC_EXAMPLE = r"""
    Example of CTC:

    ```python
    >>> from transformers import {processor_class}, HubertForCTC
    >>> from optimum.onnxruntime import {model_class}
    >>> from datasets import load_dataset
    >>> import torch

    >>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    >>> dataset = dataset.sort("id")
    >>> sampling_rate = dataset.features["audio"].sampling_rate

    >>> processor = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> # audio file is decoded on the fly
    >>> inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
    >>> with torch.no_grad():
    ...     logits = model(**inputs).logits
    >>> predicted_ids = torch.argmax(logits, dim=-1)

    >>> transcription = processor.batch_decode(predicted_ids)
    ```
"""


@add_end_docstrings(ONNX_MODEL_END_DOCSTRING)
class ORTModelForCTC(ORTModel):
    """
    ONNX Model with a language modeling head on top for Connectionist Temporal Classification (CTC). This class officially supports data2vec_audio, hubert, sew, sew_d, unispeech, unispeech_sat, wavlm, wav2vec2, wav2vec2-conformer.
    """

    auto_model_class = AutoModelForCTC

    @add_start_docstrings_to_model_forward(
        ONNX_AUDIO_INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + CTC_EXAMPLE.format(
            processor_class=_PROCESSOR_FOR_DOC,
            model_class="ORTModelForCTC",
            checkpoint="optimum/hubert-large-ls960-ft",
        )
    )
    def forward(
        self,
        input_values: Optional[Union[torch.Tensor, np.ndarray]] = None,
        *,
        return_dict: bool = False,
        **kwargs,
    ):
        # Warn about any unexpected kwargs using the helper method
        self._warn_on_unhandled_inputs(kwargs)

        use_torch = isinstance(input_values, torch.Tensor)
        self.raise_on_numpy_input_io_binding(use_torch)

        model_inputs = {
            "input_values": input_values,
        }

        if self.use_io_binding:
            batch_size = input_values.shape[0]
            final_input_size = input_values.shape[-1]

            for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
                final_input_size = (final_input_size - kernel_size) // stride + 1

            known_output_shapes = {"logits": [batch_size, final_input_size, self.config.vocab_size]}

            io_binding, output_shapes, output_buffers = self._prepare_io_binding(
                self.model, model_inputs, known_output_shapes=known_output_shapes
            )

            # run inference with binding & synchronize in case of multiple CUDA streams
            if self.device.type == "cpu":
                self.model.run_with_iobinding(io_binding)
            else:
                io_binding.synchronize_inputs()
                self.model.run_with_iobinding(io_binding)
                io_binding.synchronize_outputs()

            logits = output_buffers["logits"].view(output_shapes["logits"])
        else:
            onnx_inputs = self._prepare_onnx_inputs(use_torch, model_inputs)
            onnx_outputs = self.model.run(None, onnx_inputs)
            model_outputs = self._prepare_onnx_outputs(use_torch, onnx_outputs)

            logits = model_outputs["logits"]

        if return_dict:
            return {"logits": logits}

        # converts output to namedtuple for pipelines post-processing
        return CausalLMOutput(logits=logits)


AUDIO_XVECTOR_EXAMPLE = r"""
    Example of Audio XVector:

    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.onnxruntime import {model_class}
    >>> from datasets import load_dataset
    >>> import torch

    >>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    >>> dataset = dataset.sort("id")
    >>> sampling_rate = dataset.features["audio"].sampling_rate

    >>> feature_extractor = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> # audio file is decoded on the fly
    >>> inputs = feature_extractor(
    ...     [d["array"] for d in dataset[:2]["audio"]], sampling_rate=sampling_rate, return_tensors="pt", padding=True
    ... )
    >>> with torch.no_grad():
    ...     embeddings = model(**inputs).embeddings

    >>> embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu()

    >>> cosine_sim = torch.nn.CosineSimilarity(dim=-1)
    >>> similarity = cosine_sim(embeddings[0], embeddings[1])
    >>> threshold = 0.7
    >>> if similarity < threshold:
    ...     print("Speakers are not the same!")
    >>> round(similarity.item(), 2)
    ```
"""


@add_end_docstrings(ONNX_MODEL_END_DOCSTRING)
class ORTModelForAudioXVector(ORTModel):
    """
    ONNX Model with an XVector feature extraction head on top for tasks like Speaker Verification. This class officially supports data2vec_audio, unispeech_sat, wavlm, wav2vec2, wav2vec2-conformer.
    """

    auto_model_class = AutoModelForAudioXVector

    @add_start_docstrings_to_model_forward(
        ONNX_AUDIO_INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + AUDIO_XVECTOR_EXAMPLE.format(
            processor_class=_FEATURE_EXTRACTOR_FOR_DOC,
            model_class="ORTModelForAudioXVector",
            checkpoint="optimum/wav2vec2-base-superb-sv",
        )
    )
    def forward(
        self,
        input_values: Optional[Union[torch.Tensor, np.ndarray]] = None,
        *,
        return_dict: bool = False,
        **kwargs,
    ):
        # Warn about any unexpected kwargs using the helper method
        self._warn_on_unhandled_inputs(kwargs)

        use_torch = isinstance(input_values, torch.Tensor)
        self.raise_on_numpy_input_io_binding(use_torch)

        model_inputs = {
            "input_values": input_values,
        }

        if self.use_io_binding:
            io_binding, output_shapes, output_buffers = self._prepare_io_binding(self.model, model_inputs)

            # run inference with binding & synchronize in case of multiple CUDA streams
            if self.device.type == "cpu":
                self.model.run_with_iobinding(io_binding)
            else:
                io_binding.synchronize_inputs()
                self.model.run_with_iobinding(io_binding)
                io_binding.synchronize_outputs()

            logits = output_buffers["logits"].view(output_shapes["logits"])
            embeddings = output_buffers["embeddings"].view(output_shapes["embeddings"])

        else:
            onnx_inputs = self._prepare_onnx_inputs(use_torch, model_inputs)
            onnx_outputs = self.model.run(None, onnx_inputs)
            model_outputs = self._prepare_onnx_outputs(use_torch, onnx_outputs)

            logits = model_outputs["logits"]
            embeddings = model_outputs["embeddings"]

        if return_dict:
            return {"logits": logits, "embeddings": embeddings}

        # converts output to namedtuple for pipelines post-processing
        return XVectorOutput(logits=logits, embeddings=embeddings)


AUDIO_FRAME_CLASSIFICATION_EXAMPLE = r"""
    Example of audio frame classification:

    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.onnxruntime import {model_class}
    >>> from datasets import load_dataset
    >>> import torch

    >>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    >>> dataset = dataset.sort("id")
    >>> sampling_rate = dataset.features["audio"].sampling_rate

    >>> feature_extractor = {processor_class}.from_pretrained("{checkpoint}")
    >>> model =  {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = feature_extractor(dataset[0]["audio"]["array"], return_tensors="pt", sampling_rate=sampling_rate)
    >>> with torch.no_grad():
    ...     logits = model(**inputs).logits

    >>> probabilities = torch.sigmoid(logits[0])
    >>> labels = (probabilities > 0.5).long()
    >>> labels[0].tolist()
    ```
"""


@add_end_docstrings(ONNX_MODEL_END_DOCSTRING)
class ORTModelForAudioFrameClassification(ORTModel):
    """
    ONNX Model with a frame classification head on top for tasks like Speaker Diarization. This class officially supports data2vec_audio, unispeech_sat, wavlm, wav2vec2, wav2vec2-conformer.
    """

    auto_model_class = AutoModelForAudioFrameClassification

    @add_start_docstrings_to_model_forward(
        ONNX_AUDIO_INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + AUDIO_FRAME_CLASSIFICATION_EXAMPLE.format(
            processor_class=_FEATURE_EXTRACTOR_FOR_DOC,
            model_class="ORTModelForAudioFrameClassification",
            checkpoint="optimum/wav2vec2-base-superb-sd",
        )
    )
    def forward(
        self,
        input_values: Optional[Union[torch.Tensor, np.ndarray]] = None,
        *,
        return_dict: bool = False,
        **kwargs,
    ):
        # Warn about any unexpected kwargs using the helper method
        self._warn_on_unhandled_inputs(kwargs)

        use_torch = isinstance(input_values, torch.Tensor)
        self.raise_on_numpy_input_io_binding(use_torch)

        if self.use_io_binding:
            raise NotImplementedError()
        else:
            model_inputs = {"input_values": input_values}

            onnx_inputs = self._prepare_onnx_inputs(use_torch, model_inputs)
            onnx_outputs = self.model.run(None, onnx_inputs)
            model_outputs = self._prepare_onnx_outputs(use_torch, onnx_outputs)

            logits = model_outputs["logits"]

        if return_dict:
            return {"logits": logits}

        # converts output to namedtuple for pipelines post-processing
        return TokenClassifierOutput(logits=logits)


IMAGE_TO_IMAGE_EXAMPLE = r"""
    Example of image-to-image (Super Resolution):

    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.onnxruntime import {model_class}
    >>> from PIL import Image

    >>> image = Image.open("path/to/image.jpg")

    >>> image_processor = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = image_processor(images=image, return_tensors="pt")

    >>> with torch.no_grad():
    ...     logits = model(**inputs).logits
    ```
"""


@add_end_docstrings(ONNX_MODEL_END_DOCSTRING)
class ORTModelForImageToImage(ORTModel):
    """
    ONNX Model for image-to-image tasks. This class officially supports pix2pix, cyclegan, wav2vec2, wav2vec2-conformer.
    """

    auto_model_class = AutoModelForImageToImage

    @add_start_docstrings_to_model_forward(
        ONNX_IMAGE_INPUTS_DOCSTRING.format("batch_size, num_channels, height, width")
        + IMAGE_TO_IMAGE_EXAMPLE.format(
            processor_class=_PROCESSOR_FOR_DOC,
            model_class="ORTModelForImgageToImage",
            checkpoint="caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr",
        )
    )
    def forward(
        self,
        pixel_values: Union[torch.Tensor, np.ndarray],
        *,
        return_dict: bool = False,
        **kwargs,
    ):
        # Warn about any unexpected kwargs using the helper method
        self._warn_on_unhandled_inputs(kwargs)

        use_torch = isinstance(pixel_values, torch.Tensor)
        self.raise_on_numpy_input_io_binding(use_torch)

        model_inputs = {
            "pixel_values": pixel_values,
        }

        if self.use_io_binding:
            batch_size, num_channels, height, width = pixel_values.shape
            known_output_shapes = {
                "reconstruction": [batch_size, num_channels, height * self.config.upscale, width * self.config.upscale]
            }

            io_binding, output_shapes, output_buffers = self._prepare_io_binding(
                self.model, model_inputs, known_output_shapes=known_output_shapes
            )

            if self.device.type == "cpu":
                self.model.run_with_iobinding(io_binding)
            else:
                io_binding.synchronize_inputs()
                self.model.run_with_iobinding(io_binding)
                io_binding.synchronize_outputs()

            reconstruction = output_buffers["reconstruction"].view(output_shapes["reconstruction"])
        else:
            onnx_inputs = self._prepare_onnx_inputs(use_torch, model_inputs)
            onnx_outputs = self.model.run(None, onnx_inputs)
            model_outputs = self._prepare_onnx_outputs(use_torch, onnx_outputs)
            reconstruction = model_outputs["reconstruction"]

        if return_dict:
            return {"reconstruction": reconstruction}

        return ImageSuperResolutionOutput(reconstruction=reconstruction)


CUSTOM_TASKS_EXAMPLE = r"""
    Example of custom tasks(e.g. a sentence transformers taking `pooler_output` as output):

    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.onnxruntime import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("I love burritos!", return_tensors="np")

    >>> outputs = model(**inputs)
    >>> last_hidden_state = outputs.last_hidden_state
    >>> pooler_output = outputs.pooler_output
    ```

    Example using `transformers.pipelines`(only if the task is supported):

    ```python
    >>> from transformers import {processor_class}, pipeline
    >>> from optimum.onnxruntime import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> onnx_extractor = pipeline("feature-extraction", model=model, tokenizer=tokenizer)

    >>> text = "I love burritos!"
    >>> pred = onnx_extractor(text)
    ```
"""


@add_end_docstrings(ONNX_MODEL_END_DOCSTRING)
class ORTModelForCustomTasks(ORTModel):
    """
    ONNX Model for any custom tasks. It can be used to leverage the inference acceleration for any single-file ONNX model, that may use custom inputs and outputs.
    """

    @add_start_docstrings_to_model_forward(
        CUSTOM_TASKS_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="ORTModelForCustomTasks",
            checkpoint="optimum/sbert-all-MiniLM-L6-with-pooler",
        )
    )
    def forward(self, **model_inputs: Union[torch.Tensor, np.ndarray]):
        use_torch = isinstance(next(iter(model_inputs.values())), torch.Tensor)
        self.raise_on_numpy_input_io_binding(use_torch)

        if self.use_io_binding:
            # TODO: should this be used in favor of `model.prepare_io_binding`?
            io_binding = IOBindingHelper.prepare_io_binding(self, **model_inputs)

            # run inference with binding
            if self.device.type == "cpu":
                self.model.run_with_iobinding(io_binding)
            else:
                io_binding.synchronize_inputs()
                self.model.run_with_iobinding(io_binding)
                io_binding.synchronize_outputs()

            model_outputs = {}
            for name, output in zip(self.output_names.keys(), io_binding._iobinding.get_outputs()):
                model_outputs[name] = IOBindingHelper.to_pytorch(output)

        else:
            onnx_inputs = self._prepare_onnx_inputs(use_torch, model_inputs)
            onnx_outputs = self.model.run(None, onnx_inputs)
            model_outputs = self._prepare_onnx_outputs(use_torch, onnx_outputs)

        # converts output to namedtuple for pipelines post-processing
        return ModelOutput(**model_outputs)
