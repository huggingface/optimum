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
import re
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from huggingface_hub import HfApi, HfFolder, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForAudioClassification,
    AutoModelForAudioFrameClassification,
    AutoModelForAudioXVector,
    AutoModelForCTC,
    AutoModelForImageClassification,
    AutoModelForMaskedLM,
    AutoModelForMultipleChoice,
    AutoModelForQuestionAnswering,
    AutoModelForSemanticSegmentation,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_model_forward
from transformers.modeling_outputs import (
    BaseModelOutput,
    CausalLMOutput,
    ImageClassifierOutput,
    MaskedLMOutput,
    ModelOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SemanticSegmenterOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
    XVectorOutput,
)

import onnxruntime as ort

from ..exporters import TasksManager
from ..exporters.onnx import export
from ..modeling_base import FROM_PRETRAINED_START_DOCSTRING, OptimizedModel
from ..onnx.utils import _get_external_data_paths
from ..utils.file_utils import find_files_matching_pattern
from ..utils.save_utils import maybe_load_preprocessors, maybe_save_preprocessors
from .io_binding import IOBindingHelper, TypeHelper
from .utils import (
    ONNX_WEIGHTS_NAME,
    check_io_binding,
    get_device_for_provider,
    get_ordered_input_names,
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

ONNX_MODEL_START_DOCSTRING = r"""
    This model inherits from [`~onnxruntime.modeling_ort.ORTModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving)

    Args:
        config (`transformers.PretrainedConfig`): [PretrainedConfig](https://huggingface.co/docs/transformers/main_classes/configuration#transformers.PretrainedConfig) is the Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~onnxruntime.modeling_ort.ORTModel.from_pretrained`] method to load the model weights.
        model (`onnxruntime.InferenceSession`): [onnxruntime.InferenceSession](https://onnxruntime.ai/docs/api/python/api_summary.html#inferencesession) is the main class used to run a model. Check out the [`~onnxruntime.modeling_ort.ORTModel.load_model`] method for more information.
        use_io_binding (`Optional[bool]`, defaults to `None`): Whether to use IOBinding during inference to avoid memory copy between the host and devices. Defaults to `True` if the device is CUDA, otherwise defaults to `False`.
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

    _AUTOMODELS_TO_TASKS = {cls_name: task for task, cls_name in TasksManager._TASKS_TO_AUTOMODELS.items()}
    model_type = "onnx_model"
    auto_model_class = AutoModel

    @classproperty
    def export_feature(cls):
        logger.warning(f"{cls.__name__}.export_feature is deprecated, and will be removed in optimum 2.0.")
        return cls._AUTOMODELS_TO_TASKS.get(cls.auto_model_class.__name__, None)

    @classmethod
    def _auto_model_to_task(cls, auto_model_class):
        """
        Get the task corresponding to a class (for example AutoModelForXXX in transformers).
        """
        return cls._AUTOMODELS_TO_TASKS[auto_model_class.__name__]

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
        self._device = get_device_for_provider(self.providers[0])

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

        self.model_path = Path(model._model_path)
        self.model_name = self.model_path.name

        self.shared_attributes_init(
            model,
            use_io_binding,
            model_save_dir,
            preprocessors,
            **kwargs,
        )

        self.inputs_names = {input_key.name: idx for idx, input_key in enumerate(model.get_inputs())}
        self.output_names = {output_key.name: idx for idx, output_key in enumerate(model.get_outputs())}

        self._ordered_input_names = get_ordered_input_names(self.inputs_names.keys(), func=self.forward)

    # TODO: why do we make device a property since we are only access the value, and do not do any check when setting the value?
    @property
    def device(self) -> torch.device:
        """
        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """
        return self._device

    @device.setter
    def device(self, value: torch.device):
        self._device = value

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

        if device.type == "cuda" and self._use_io_binding is False:
            self.use_io_binding = True
            logger.info(
                "use_io_binding was set to False, setting it to True because it can provide a huge speedup on GPUs. "
                "It is possible to disable this feature manually by setting the use_io_binding attribute back to False."
            )

        self.device = device
        provider = get_provider_for_device(self.device)
        validate_provider_availability(provider)  # raise error if the provider is not available

        self.model.set_providers([provider], provider_options=[provider_options])
        self.providers = self.model.get_providers()

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
        use_auth_token: Optional[Union[bool, str]] = None,
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
                use_auth_token=use_auth_token,
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
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
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
        model_path = Path(model_id)
        regular_onnx_filenames = ORTModel._generate_regular_names_for_filename(ONNX_WEIGHTS_NAME)

        if file_name is None:
            if model_path.is_dir():
                onnx_files = list(model_path.glob("*.onnx"))
            else:
                if isinstance(use_auth_token, bool):
                    token = HfFolder().get_token()
                else:
                    token = use_auth_token
                repo_files = map(Path, HfApi().list_repo_files(model_id, revision=revision, token=token))
                pattern = "*.onnx" if subfolder == "" else f"{subfolder}/*.onnx"
                onnx_files = [p for p in repo_files if p.match(pattern)]

            if len(onnx_files) == 0:
                raise FileNotFoundError(f"Could not find any ONNX model file in {model_path}")
            elif len(onnx_files) > 1:
                raise RuntimeError(
                    f"Too many ONNX model files were found in {model_path}, specify which one to load by using the "
                    "file_name argument."
                )
            else:
                file_name = onnx_files[0].name

        if file_name not in regular_onnx_filenames:
            logger.warning(
                f"The ONNX file {file_name} is not a regular name used in optimum.onnxruntime, the ORTModel might "
                "not behave as expected."
            )

        preprocessors = None
        if model_path.is_dir():
            model = ORTModel.load_model(
                model_path / file_name,
                provider=provider,
                session_options=session_options,
                provider_options=provider_options,
            )
            new_model_save_dir = model_path
            preprocessors = maybe_load_preprocessors(model_id)
        else:
            model_cache_path = hf_hub_download(
                repo_id=model_id,
                filename=file_name,
                subfolder=subfolder,
                use_auth_token=use_auth_token,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
            )

            # try download external data
            try:
                hf_hub_download(
                    repo_id=model_id,
                    subfolder=subfolder,
                    filename=file_name + "_data",
                    use_auth_token=use_auth_token,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    local_files_only=local_files_only,
                )
            except EntryNotFoundError:
                # model doesn't use external data
                pass

            model = ORTModel.load_model(
                model_cache_path, provider=provider, session_options=session_options, provider_options=provider_options
            )
            new_model_save_dir = Path(model_cache_path).parent
            preprocessors = maybe_load_preprocessors(model_id, subfolder=subfolder)

        # model_save_dir can be provided in kwargs as a TemporaryDirectory instance, in which case we want to keep it
        # instead of the path only.
        if model_save_dir is None:
            model_save_dir = new_model_save_dir

        return cls(
            model=model,
            config=config,
            use_io_binding=use_io_binding,
            model_save_dir=model_save_dir,
            preprocessors=preprocessors,
        )

    @classmethod
    def _from_transformers(
        cls,
        model_id: str,
        config: "PretrainedConfig",
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
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

        kwargs_to_get_model = {
            "subfolder": subfolder,
            "revision": revision,
            "trust_remote_code": trust_remote_code,
        }

        model = TasksManager.get_model_from_task(task, model_id, **kwargs_to_get_model)
        onnx_config_class = TasksManager.get_exporter_config_constructor(
            model=model, exporter="onnx", task=task, model_name=model_id
        )

        onnx_config = onnx_config_class(model.config)

        tmp_dir = TemporaryDirectory()
        tmp_dir_path = Path(tmp_dir.name)
        export(
            model=model,
            config=onnx_config,
            opset=onnx_config.DEFAULT_ONNX_OPSET,
            output=tmp_dir_path / ONNX_WEIGHTS_NAME,
        )
        config.save_pretrained(tmp_dir_path)
        maybe_save_preprocessors(model_id, tmp_dir_path, src_subfolder=subfolder)

        return cls._from_pretrained(
            tmp_dir_path,
            config,
            use_io_binding=use_io_binding,
            model_save_dir=tmp_dir,
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
        use_auth_token: Optional[str] = None,
        cache_dir: Optional[str] = None,
        subfolder: str = "",
        config: Optional["PretrainedConfig"] = None,
        local_files_only: bool = False,
        provider: str = "CPUExecutionProvider",
        session_options: Optional[ort.SessionOptions] = None,
        provider_options: Optional[Dict[str, Any]] = None,
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
        kwargs (`Dict[str, Any]`):
            Will be passed to the underlying model loading methods.

        > Parameters for decoder models (ORTModelForCausalLM, ORTModelForSeq2SeqLM, ORTModelForSeq2SeqLM, ORTModelForSpeechSeq2Seq, ORTModelForVision2Seq)

        use_cache (`Optional[bool]`, defaults to `True`):
            Whether or not past key/values cache should be used. Defaults to `True`.

        > Parameters for ORTModelForCausalLM

        use_merged (`Optional[bool]`, defaults to `None`):
            whether or not to use a single ONNX that handles both the decoding without and with past key values reuse. This option defaults
            to `True` if loading from a local repository and a merged decoder is found. When exporting with `from_transformers=True`,
            defaults to `False`. This option should be set to `True` to minimize memory usage.

        Returns:
            `ORTModel`: The loaded ORTModel model.
        """
        return super().from_pretrained(
            model_id,
            export=export,
            force_download=force_download,
            use_auth_token=use_auth_token,
            cache_dir=cache_dir,
            subfolder=subfolder,
            config=config,
            local_files_only=local_files_only,
            provider=provider,
            session_options=session_options,
            provider_options=provider_options,
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
        # It is actually covered below, but this is to make things faster.
        elif axis_name in dimensions:
            return dimensions[axis_name]

        # Tokens is going to be populated by iterating over every match for the self.output_shape_inference_pattern.
        # This pattern matches 4 things: axis names, integer values, operators (+, -, *, /) and parenthesis.
        tokens = []
        for idx, match_ in enumerate(re.finditer(self.output_shape_inference_pattern, axis_name)):
            groups = match_.groups()
            matched_group = None
            for idx, group in enumerate(groups):
                if group is not None:
                    matched_group = idx
                    break

            # For every match except an axis name, we simply append the content of the match to the tokens list.
            # For an axis name, we check if it is specified in the `dimensions` dictionary. If for some reason it is
            # not there, or its value not an integer, the shape inference process stops and we return the axis name as
            # is.
            if matched_group == 0:
                dim = dimensions.get(groups[0], None)
                if dim is None or not isinstance(dim, int):
                    return axis_name
                tokens.append(str(dim))
            else:
                tokens.append(groups[matched_group])

        # Here it should not be problematic to use eval since anything not matching the pattern would trigger an
        # exception.
        return int(eval(" ".join(tokens)))

    def _prepare_io_binding(
        self,
        model: ort.InferenceSession,
        *model_inputs: torch.Tensor,
        ordered_input_names: List[str],
        known_output_shapes: Optional[Dict[str, Tuple[int]]] = None,
        outputs_to_not_bind: Optional[Union[Set[str], str]] = None,
    ) -> Tuple[ort.IOBinding, Dict[str, Tuple[int]], Dict[str, torch.Tensor]]:
        """
        Prepares IO binding for ONNX Runtime.

        Args:
            model (`ort.InferenceSession`):
                The model for which we want to bind the inputs and outputs.
            *model_inputs:
                The inputs of the model.
            ordered_input_names (`List[str]`):
                Names of the inputs, that must match with the order of model_inputs.
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

        name_to_np_type = TypeHelper.get_io_numpy_type_map(model)

        input_name_to_tensor = {}
        for idx, tensor in enumerate(model_inputs):
            if tensor is None:
                continue
            name = ordered_input_names[idx]
            input_name_to_tensor[name] = tensor
            tensor = tensor.contiguous()
            io_binding.bind_input(
                name,
                tensor.device.type,
                self.device.index,
                name_to_np_type[name],
                tuple(tensor.shape),
                tensor.data_ptr(),
            )
        dimensions = {}
        for input_ in model.get_inputs():
            shape = input_.shape
            for idx, axis in enumerate(shape):
                if isinstance(axis, str):
                    dimensions[axis] = input_name_to_tensor[input_.name].shape[idx]

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
            io_binding.bind_output(
                output_name,
                output_buffer.device.type,
                self.device.index,
                name_to_np_type[output_name],
                output_shape,
                output_buffer.data_ptr(),
            )
            output_shapes[output_name] = output_shape
            output_buffers[output_name] = output_buffer

        return io_binding, output_shapes, output_buffers

    def prepare_io_binding(self, *model_inputs, ordered_input_names):
        return self._prepare_io_binding(self.model, ordered_input_names=ordered_input_names, *model_inputs)

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


@add_start_docstrings(
    """
    Onnx Model with a BaseModelOutput for feature-extraction tasks.
    """,
    ONNX_MODEL_START_DOCSTRING,
)
class ORTModelForFeatureExtraction(ORTModel):
    """
    Feature Extraction model for ONNX.
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
        **kwargs,
    ):
        use_torch = isinstance(input_ids, torch.Tensor)
        self.raise_on_numpy_input_io_binding(use_torch)

        if self.device.type == "cuda" and self.use_io_binding:
            io_binding, output_shapes, output_buffers = self.prepare_io_binding(
                input_ids,
                attention_mask,
                token_type_ids,
                ordered_input_names=self._ordered_input_names,
            )

            # run inference with binding & synchronize in case of multiple CUDA streams
            io_binding.synchronize_inputs()
            self.model.run_with_iobinding(io_binding)
            io_binding.synchronize_outputs()

            # converts output to namedtuple for pipelines post-processing
            return BaseModelOutput(
                last_hidden_state=output_buffers["last_hidden_state"].view(output_shapes["last_hidden_state"])
            )
        else:
            if use_torch:
                input_ids = input_ids.cpu().detach().numpy()
                attention_mask = attention_mask.cpu().detach().numpy()
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.cpu().detach().numpy()

            onnx_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if token_type_ids is not None:
                onnx_inputs["token_type_ids"] = token_type_ids

            outputs = self.model.run(None, onnx_inputs)

            last_hidden_state = outputs[self.output_names["last_hidden_state"]]
            if use_torch:
                last_hidden_state = torch.from_numpy(last_hidden_state).to(self.device)

            # converts output to namedtuple for pipelines post-processing
            return BaseModelOutput(last_hidden_state=last_hidden_state)


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


@add_start_docstrings(
    """
    Onnx Model with a MaskedLMOutput for masked language modeling tasks.
    """,
    ONNX_MODEL_START_DOCSTRING,
)
class ORTModelForMaskedLM(ORTModel):
    """
    Masked language model for ONNX.
    """

    auto_model_class = AutoModelForMaskedLM

    @add_start_docstrings_to_model_forward(
        ONNX_TEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + MASKED_LM_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="ORTModelForMaskedLM",
            checkpoint="optimum/bert-base-uncased-for-masked-lm",
        )
    )
    def forward(
        self,
        input_ids: Optional[Union[torch.Tensor, np.ndarray]] = None,
        attention_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
        token_type_ids: Optional[Union[torch.Tensor, np.ndarray]] = None,
        **kwargs,
    ):
        use_torch = isinstance(input_ids, torch.Tensor)
        self.raise_on_numpy_input_io_binding(use_torch)

        if self.device.type == "cuda" and self.use_io_binding:
            io_binding, output_shapes, output_buffers = self.prepare_io_binding(
                input_ids,
                attention_mask,
                token_type_ids,
                ordered_input_names=self._ordered_input_names,
            )

            # run inference with binding & synchronize in case of multiple CUDA streams
            io_binding.synchronize_inputs()
            self.model.run_with_iobinding(io_binding)
            io_binding.synchronize_outputs()

            # converts output to namedtuple for pipelines post-processing
            return MaskedLMOutput(logits=output_buffers["logits"].view(output_shapes["logits"]))
        else:
            if use_torch:
                input_ids = input_ids.cpu().detach().numpy()
                attention_mask = attention_mask.cpu().detach().numpy()
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.cpu().detach().numpy()

            # converts pytorch inputs into numpy inputs for onnx
            onnx_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if token_type_ids is not None:
                onnx_inputs["token_type_ids"] = token_type_ids

            # run inference
            outputs = self.model.run(None, onnx_inputs)
            logits = outputs[self.output_names["logits"]]

            if use_torch:
                logits = torch.from_numpy(logits).to(self.device)

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


@add_start_docstrings(
    """
    Onnx Model with a QuestionAnsweringModelOutput for extractive question-answering tasks like SQuAD.
    """,
    ONNX_MODEL_START_DOCSTRING,
)
class ORTModelForQuestionAnswering(ORTModel):
    """
    Question Answering model for ONNX.
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
        **kwargs,
    ):
        use_torch = isinstance(input_ids, torch.Tensor)
        self.raise_on_numpy_input_io_binding(use_torch)

        if self.device.type == "cuda" and self.use_io_binding:
            io_binding, output_shapes, output_buffers = self.prepare_io_binding(
                input_ids,
                attention_mask,
                token_type_ids,
                ordered_input_names=self._ordered_input_names,
            )

            # run inference with binding & synchronize in case of multiple CUDA streams
            io_binding.synchronize_inputs()
            self.model.run_with_iobinding(io_binding)
            io_binding.synchronize_outputs()

            # converts output to namedtuple for pipelines post-processing
            return QuestionAnsweringModelOutput(
                start_logits=output_buffers["start_logits"].view(output_shapes["start_logits"]),
                end_logits=output_buffers["end_logits"].view(output_shapes["end_logits"]),
            )
        else:
            if use_torch:
                input_ids = input_ids.cpu().detach().numpy()
                attention_mask = attention_mask.cpu().detach().numpy()
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.cpu().detach().numpy()

            # converts pytorch inputs into numpy inputs for onnx
            onnx_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if token_type_ids is not None:
                onnx_inputs["token_type_ids"] = token_type_ids

            # run inference
            outputs = self.model.run(None, onnx_inputs)

            start_logits = outputs[self.output_names["start_logits"]]
            end_logits = outputs[self.output_names["end_logits"]]
            if use_torch:
                start_logits = torch.from_numpy(start_logits).to(self.device)
                end_logits = torch.from_numpy(end_logits).to(self.device)

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


@add_start_docstrings(
    """
    Onnx Model with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    ONNX_MODEL_START_DOCSTRING,
)
class ORTModelForSequenceClassification(ORTModel):
    """
    Sequence Classification model for ONNX.
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
        **kwargs,
    ):
        use_torch = isinstance(input_ids, torch.Tensor)
        self.raise_on_numpy_input_io_binding(use_torch)

        if self.device.type == "cuda" and self.use_io_binding:
            io_binding, output_shapes, output_buffers = self.prepare_io_binding(
                input_ids,
                attention_mask,
                token_type_ids,
                ordered_input_names=self._ordered_input_names,
            )

            # run inference with binding & synchronize in case of multiple CUDA streams
            io_binding.synchronize_inputs()
            self.model.run_with_iobinding(io_binding)
            io_binding.synchronize_outputs()

            # converts output to namedtuple for pipelines post-processing
            return SequenceClassifierOutput(logits=output_buffers["logits"].view(output_shapes["logits"]))
        else:
            if use_torch:
                input_ids = input_ids.cpu().detach().numpy()
                attention_mask = attention_mask.cpu().detach().numpy()
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.cpu().detach().numpy()

            onnx_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if token_type_ids is not None:
                onnx_inputs["token_type_ids"] = token_type_ids

            outputs = self.model.run(None, onnx_inputs)

            logits = outputs[self.output_names["logits"]]
            if use_torch:
                logits = torch.from_numpy(logits).to(self.device)

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


@add_start_docstrings(
    """
    Onnx Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
    for Named-Entity-Recognition (NER) tasks.
    """,
    ONNX_MODEL_START_DOCSTRING,
)
class ORTModelForTokenClassification(ORTModel):
    """
    Token Classification model for ONNX.
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
        **kwargs,
    ):
        use_torch = isinstance(input_ids, torch.Tensor)
        self.raise_on_numpy_input_io_binding(use_torch)

        if self.device.type == "cuda" and self.use_io_binding:
            io_binding, output_shapes, output_buffers = self.prepare_io_binding(
                input_ids,
                attention_mask,
                token_type_ids,
                ordered_input_names=self._ordered_input_names,
            )

            # run inference with binding & synchronize in case of multiple CUDA streams
            io_binding.synchronize_inputs()
            self.model.run_with_iobinding(io_binding)
            io_binding.synchronize_outputs()

            # converts output to namedtuple for pipelines post-processing
            return TokenClassifierOutput(logits=output_buffers["logits"].view(output_shapes["logits"]))
        else:
            if use_torch:
                input_ids = input_ids.cpu().detach().numpy()
                attention_mask = attention_mask.cpu().detach().numpy()
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.cpu().detach().numpy()

            # converts pytorch inputs into numpy inputs for onnx
            onnx_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if token_type_ids is not None:
                onnx_inputs["token_type_ids"] = token_type_ids

            # run inference
            outputs = self.model.run(None, onnx_inputs)
            logits = outputs[self.output_names["logits"]]

            if use_torch:
                logits = torch.from_numpy(logits).to(self.device)

            # converts output to namedtuple for pipelines post-processing
            return TokenClassifierOutput(logits=logits)


MULTIPLE_CHOICE_EXAMPLE = r"""
    Example of mutliple choice:

    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.onnxruntime import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}", from_transformers=True)

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


@add_start_docstrings(
    """
    Onnx Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    ONNX_MODEL_START_DOCSTRING,
)
class ORTModelForMultipleChoice(ORTModel):
    """
    Multiple choice model for ONNX.
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
        **kwargs,
    ):
        use_torch = isinstance(input_ids, torch.Tensor)
        self.raise_on_numpy_input_io_binding(use_torch)

        if self.device.type == "cuda" and self.use_io_binding:
            io_binding, output_shapes, output_buffers = self.prepare_io_binding(
                input_ids,
                attention_mask,
                token_type_ids,
                ordered_input_names=self._ordered_input_names,
            )

            # run inference with binding & synchronize in case of multiple CUDA streams
            io_binding.synchronize_inputs()
            self.model.run_with_iobinding(io_binding)
            io_binding.synchronize_outputs()

            # converts output to namedtuple for pipelines post-processing
            return MultipleChoiceModelOutput(logits=output_buffers["logits"].view(output_shapes["logits"]))
        else:
            if use_torch:
                input_ids = input_ids.cpu().detach().numpy()
                attention_mask = attention_mask.cpu().detach().numpy()
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.cpu().detach().numpy()

            onnx_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if token_type_ids is not None:
                onnx_inputs["token_type_ids"] = token_type_ids

            # run inference
            outputs = self.model.run(None, onnx_inputs)
            logits = outputs[self.output_names["logits"]]

            if use_torch:
                logits = torch.from_numpy(logits).to(self.device)

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


@add_start_docstrings(
    """
    Onnx Model for image-classification tasks.
    """,
    ONNX_MODEL_START_DOCSTRING,
)
class ORTModelForImageClassification(ORTModel):
    """
    Image Classification model for ONNX.
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
        **kwargs,
    ):
        use_torch = isinstance(pixel_values, torch.Tensor)
        self.raise_on_numpy_input_io_binding(use_torch)

        if self.device.type == "cuda" and self.use_io_binding:
            io_binding, output_shapes, output_buffers = self.prepare_io_binding(
                pixel_values, ordered_input_names=self._ordered_input_names
            )

            # run inference with binding & synchronize in case of multiple CUDA streams
            io_binding.synchronize_inputs()
            self.model.run_with_iobinding(io_binding)
            io_binding.synchronize_outputs()

            # converts output to namedtuple for pipelines post-processing
            return ImageClassifierOutput(logits=output_buffers["logits"].view(output_shapes["logits"]))
        else:
            if use_torch:
                pixel_values = pixel_values.cpu().detach().numpy()

            onnx_inputs = {
                "pixel_values": pixel_values,
            }

            # run inference
            outputs = self.model.run(None, onnx_inputs)
            logits = outputs[self.output_names["logits"]]

            if use_torch:
                logits = torch.from_numpy(logits).to(self.device)

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


@add_start_docstrings(
    """
    Onnx Model with an all-MLP decode head on top e.g. for ADE20k, CityScapes.
    """,
    ONNX_MODEL_START_DOCSTRING,
)
class ORTModelForSemanticSegmentation(ORTModel):
    """
    Semantic Segmentation model for ONNX.
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
    def forward(self, **kwargs):
        use_torch = isinstance(next(iter(kwargs.values())), torch.Tensor)
        self.raise_on_numpy_input_io_binding(use_torch)

        if self.device.type == "cuda" and self.use_io_binding:
            io_binding = IOBindingHelper.prepare_io_binding(
                self,
                **kwargs,
                ordered_input_names=self._ordered_input_names,
            )

            # run inference with binding
            io_binding.synchronize_inputs()
            self.model.run_with_iobinding(io_binding)
            io_binding.synchronize_outputs()

            outputs = {}
            for name, output in zip(self.output_names.keys(), io_binding._iobinding.get_outputs()):
                outputs[name] = IOBindingHelper.to_pytorch(output)

            # converts output to namedtuple for pipelines post-processing
            return SemanticSegmenterOutput(logits=outputs["logits"])
        else:
            onnx_inputs = self._prepare_onnx_inputs(use_torch=use_torch, **kwargs)

            # run inference
            onnx_outputs = self.model.run(None, onnx_inputs)

            logits = onnx_outputs[self.output_names["logits"]]
            if use_torch:
                logits = torch.from_numpy(logits).to(self.device)

            # converts output to namedtuple for pipelines post-processing
            return SemanticSegmenterOutput(logits=logits)

    def _prepare_onnx_inputs(self, use_torch: bool, **kwargs):
        onnx_inputs = {}
        # converts pytorch inputs into numpy inputs for onnx
        for input in self.inputs_names.keys():
            onnx_inputs[input] = kwargs.pop(input)

            if use_torch:
                onnx_inputs[input] = onnx_inputs[input].cpu().detach().numpy()

        return onnx_inputs


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


@add_start_docstrings(
    """
    Onnx Model with a sequence classification head on top (a linear layer over the pooled output) for tasks like
    SUPERB Keyword Spotting.
    """,
    ONNX_MODEL_START_DOCSTRING,
)
class ORTModelForAudioClassification(ORTModel):
    """
    Audio Classification model for ONNX.
    """

    auto_model_class = AutoModelForAudioClassification

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
        input_values: Optional[torch.Tensor] = None,
        attenton_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        use_torch = isinstance(input_values, torch.Tensor)
        self.raise_on_numpy_input_io_binding(use_torch)
        if self.device.type == "cuda" and self.use_io_binding:
            io_binding, output_shapes, output_buffers = self.prepare_io_binding(
                input_values, ordered_input_names=self._ordered_input_names
            )

            # run inference with binding & synchronize in case of multiple CUDA streams
            io_binding.synchronize_inputs()
            self.model.run_with_iobinding(io_binding)
            io_binding.synchronize_outputs()

            # converts output to namedtuple for pipelines post-processing
            return SequenceClassifierOutput(logits=output_buffers["logits"].view(output_shapes["logits"]))
        else:
            if use_torch:
                # converts pytorch inputs into numpy inputs for onnx
                onnx_inputs = {
                    "input_values": input_values.cpu().detach().numpy(),
                }
            else:
                onnx_inputs = {
                    "input_values": input_values,
                }

            # run inference
            outputs = self.model.run(None, onnx_inputs)

            logits = outputs[self.output_names["logits"]]
            if use_torch:
                logits = torch.from_numpy(logits).to(self.device)

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


@add_start_docstrings(
    """
    Onnx Model with a language modeling head on top for Connectionist Temporal Classification (CTC).
    """,
    ONNX_MODEL_START_DOCSTRING,
)
class ORTModelForCTC(ORTModel):
    """
    CTC model for ONNX.
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
        input_values: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        use_torch = isinstance(input_values, torch.Tensor)
        self.raise_on_numpy_input_io_binding(use_torch)
        if self.device.type == "cuda" and self.use_io_binding:
            io_binding, output_shapes, output_buffers = self.prepare_io_binding(
                input_values, ordered_input_names=self._ordered_input_names
            )

            # run inference with binding & synchronize in case of multiple CUDA streams
            io_binding.synchronize_inputs()
            self.model.run_with_iobinding(io_binding)
            io_binding.synchronize_outputs()

            # converts output to namedtuple for pipelines post-processing
            return CausalLMOutput(logits=output_buffers["logits"].view(output_shapes["logits"]))
        else:
            if use_torch:
                # converts pytorch inputs into numpy inputs for onnx
                onnx_inputs = {
                    "input_values": input_values.cpu().detach().numpy(),
                }
            else:
                onnx_inputs = {
                    "input_values": input_values,
                }

            # run inference
            outputs = self.model.run(None, onnx_inputs)

            logits = outputs[self.output_names["logits"]]
            if use_torch:
                logits = torch.from_numpy(logits).to(self.device)
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


@add_start_docstrings(
    """
    Onnx Model with an XVector feature extraction head on top for tasks like Speaker Verification.
    """,
    ONNX_MODEL_START_DOCSTRING,
)
class ORTModelForAudioXVector(ORTModel):
    """
    Audio XVector model for ONNX.
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
        input_values: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        use_torch = isinstance(input_values, torch.Tensor)
        self.raise_on_numpy_input_io_binding(use_torch)
        if self.device.type == "cuda" and self.use_io_binding:
            io_binding, output_shapes, output_buffers = self.prepare_io_binding(
                input_values, ordered_input_names=self._ordered_input_names
            )

            # run inference with binding & synchronize in case of multiple CUDA streams
            io_binding.synchronize_inputs()
            self.model.run_with_iobinding(io_binding)
            io_binding.synchronize_outputs()

            # converts output to namedtuple for pipelines post-processing
            return XVectorOutput(
                logits=output_buffers["logits"].view(output_shapes["logits"]),
                embeddings=output_buffers["embeddings"].view(output_shapes["embeddings"]),
            )
        else:
            if use_torch:
                # converts pytorch inputs into numpy inputs for onnx
                onnx_inputs = {
                    "input_values": input_values.cpu().detach().numpy(),
                }
            else:
                onnx_inputs = {
                    "input_values": input_values,
                }

            # run inference
            outputs = self.model.run(None, onnx_inputs)

            logits = outputs[self.output_names["logits"]]
            embeddings = outputs[self.output_names["embeddings"]]
            if use_torch:
                logits = torch.from_numpy(logits).to(self.device)
                embeddings = torch.from_numpy(embeddings).to(self.device)

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


@add_start_docstrings(
    """
    Onnx Model for with a frame classification head on top for tasks like Speaker Diarization.
    """,
    ONNX_MODEL_START_DOCSTRING,
)
class ORTModelForAudioFrameClassification(ORTModel):
    """
    Audio Frame Classification model for ONNX.
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
        input_values: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        use_torch = isinstance(input_values, torch.Tensor)
        self.raise_on_numpy_input_io_binding(use_torch)

        if self.device.type == "cuda" and self.use_io_binding:
            raise NotImplementedError()
        else:
            if use_torch:
                # converts pytorch inputs into numpy inputs for onnx
                onnx_inputs = {
                    "input_values": input_values.cpu().detach().numpy(),
                }
            else:
                onnx_inputs = {
                    "input_values": input_values,
                }

            # run inference
            outputs = self.model.run(None, onnx_inputs)

            logits = outputs[self.output_names["logits"]]
            if use_torch:
                logits = torch.from_numpy(logits).to(self.device)
            # converts output to namedtuple for pipelines post-processing
            return TokenClassifierOutput(logits=logits)


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


@add_start_docstrings(
    """
    ONNX Model for any custom tasks. It can be used to leverage the inference acceleration for any single-file ONNX model.
    """,
    ONNX_MODEL_START_DOCSTRING,
)
class ORTModelForCustomTasks(ORTModel):
    """
    Model for any custom tasks if the ONNX model is stored in a single file.
    """

    @add_start_docstrings_to_model_forward(
        CUSTOM_TASKS_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="ORTModelForCustomTasks",
            checkpoint="optimum/sbert-all-MiniLM-L6-with-pooler",
        )
    )
    def forward(self, **kwargs):
        use_torch = isinstance(next(iter(kwargs.values())), torch.Tensor)
        self.raise_on_numpy_input_io_binding(use_torch)

        if self.device.type == "cuda" and self.use_io_binding:
            io_binding = IOBindingHelper.prepare_io_binding(
                self,
                **kwargs,
                ordered_input_names=self._ordered_input_names,
            )

            # run inference with binding
            io_binding.synchronize_inputs()
            self.model.run_with_iobinding(io_binding)
            io_binding.synchronize_outputs()

            outputs = {}
            for name, output in zip(self.output_names.keys(), io_binding._iobinding.get_outputs()):
                outputs[name] = IOBindingHelper.to_pytorch(output)

            # converts output to namedtuple for pipelines post-processing
            return ModelOutput(**outputs)
        else:
            # converts pytorch inputs into numpy inputs for onnx
            onnx_inputs = self._prepare_onnx_inputs(use_torch=use_torch, **kwargs)

            # run inference
            onnx_outputs = self.model.run(None, onnx_inputs)
            outputs = self._prepare_onnx_outputs(onnx_outputs, use_torch=use_torch)

            # converts output to namedtuple for pipelines post-processing
            return ModelOutput(outputs)

    def _prepare_onnx_inputs(self, use_torch: bool, **kwargs):
        onnx_inputs = {}
        # converts pytorch inputs into numpy inputs for onnx
        for input in self.inputs_names.keys():
            onnx_inputs[input] = kwargs.pop(input)

            if use_torch:
                onnx_inputs[input] = onnx_inputs[input].cpu().detach().numpy()

        return onnx_inputs

    def _prepare_onnx_outputs(self, onnx_outputs, use_torch: bool):
        outputs = {}
        # converts onnxruntime outputs into tensor for standard outputs
        for output, idx in self.output_names.items():
            outputs[output] = onnx_outputs[idx]

            if use_torch:
                outputs[output] = torch.from_numpy(outputs[output]).to(self.device)

        return outputs
