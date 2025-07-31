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
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union

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
from typing_extensions import Self

from onnxruntime import InferenceSession, SessionOptions

from ..exporters import TasksManager
from ..exporters.onnx import main_export
from ..modeling_base import FROM_PRETRAINED_START_DOCSTRING, OptimizedModel
from ..utils.file_utils import find_files_matching_pattern
from ..utils.save_utils import maybe_save_preprocessors
from .base import ORTSessionMixin
from .constants import ONNX_FILE_PATTERN
from .utils import prepare_providers_and_provider_options


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


# TODO: remove OptimizedModel and use a HubMixin to be able to combine it freely with other mixins
class ORTModel(ORTSessionMixin, OptimizedModel):
    """
    Base class for implementing models using ONNX Runtime.

    The ORTModel implements generic methods for interacting with the Hugging Face Hub as well as exporting vanilla
    transformers models to ONNX using `optimum.exporters.onnx` toolchain.

    Class attributes:
        - model_type (`str`, *optional*, defaults to `"onnx_model"`) -- The name of the model type to use when
        registering the ORTModel classes.
        - auto_model_class (`Type`, *optional*, defaults to `AutoModel`) -- The "AutoModel" class to represented by the
        current ORTModel class.

    Args:
        - config ([`~transformers.PretrainedConfig`] -- The configuration of the model.
        - session (`~onnxruntime.InferenceSession`) -- The ONNX Runtime InferenceSession that is running the model.
        - use_io_binding (`bool`, *optional*, defaults to `True`) -- Whether to use I/O bindings with **ONNX Runtime
        with the CUDAExecutionProvider**, this can significantly speedup inference depending on the task.
        - model_save_dir (`Path`) -- The directory where the model exported to ONNX is saved.
        By defaults, if the loaded model is local, the directory where the original model will be used. Otherwise, the
        cache directory is used.
    """

    model_type = "onnx_model"
    auto_model_class = AutoModel
    _library_name: Optional[str] = None

    def __init__(
        self,
        *args,
        config: "PretrainedConfig" = None,
        session: "InferenceSession" = None,
        use_io_binding: Optional[bool] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        **kwargs,
    ):
        # DEPRECATED BEHAVIOR
        if args:
            logger.warning(
                "Instantiating an ORTModel with positional arguments is deprecated and will be removed in the next version. "
                "Please use the keyword arguments {config, session, use_io_binding, model_save_dir} instead."
            )
            # old signature is ORTModel(model, config, use_io_binding, model_save_dir, preprocessors)
            session = args[0]
            if len(args) > 1:
                config = args[1]
            if len(args) > 2:
                use_io_binding = args[2]
            if len(args) > 3:
                model_save_dir = args[3]
            if len(args) > 4:
                _ = args[4]

        if kwargs.get("model", None) is not None:
            logger.warning(
                "Passing the inference session as `model` argument to an ORTModel is deprecated. "
                "Please use `session` instead."
            )
            session = kwargs.pop("model")
        if kwargs:
            logger.warning(
                f"Some keyword arguments were passed to the ORTModel constructor that are not part of its signature: {', '.join(kwargs.keys())}. "
                "These arguments will be ignored in the current version and will raise an error in the next version."
            )

        if config is None:
            raise ValueError(
                "The parameter config is required. Please pass a config or use the from_pretrained method."
            )
        if session is None:
            raise ValueError(
                "The parameter session is required. Please pass a session or use the from_pretrained method."
            )
        ## END OF DEPRECATED BEHAVIOR

        super().__init__(model=session, config=config)
        self.initialize_ort_attributes(session=session, use_io_binding=use_io_binding)

        # This attribute is needed to keep one reference on the temporary directory, since garbage collecting it
        # would end-up removing the directory containing the underlying ONNX model.
        self._model_save_dir_tempdirectory_instance = None
        if model_save_dir is None:
            self.model_save_dir = Path(session._model_path).parent
        elif isinstance(model_save_dir, TemporaryDirectory):
            self._model_save_dir_tempdirectory_instance = model_save_dir
            self.model_save_dir = Path(model_save_dir.name)
        elif isinstance(model_save_dir, str):
            self.model_save_dir = Path(model_save_dir)
        else:
            self.model_save_dir = model_save_dir

        # Registers the ORTModelForXXX classes into the transformers AutoModel classes to avoid warnings when creating
        # a pipeline https://github.com/huggingface/transformers/blob/cad61b68396a1a387287a8e2e2fef78a25b79383/src/transformers/pipelines/base.py#L863
        AutoConfig.register(self.model_type, AutoConfig)
        if hasattr(self.auto_model_class, "register"):
            self.auto_model_class.register(AutoConfig, self.__class__)

    def _save_pretrained(self, save_directory: Union[str, Path]):
        """
        Saves the underlying ONNX model and its external data files (if any) to the specified directory.
        This method is called by the `save_pretrained` method of the `OptimizedModel` class.
        The model is copied from `self.session._model_path` to `save_directory`.

        Args:
            save_directory (`Union[str, Path]`):
                Directory where to save the model file.
        """
        self.save_session(save_directory)

    @staticmethod
    def _generate_regular_names_for_filename(filename: str):
        name, extension = filename.rsplit(".", maxsplit=1)
        return [filename, f"{name}_quantized.{extension}", f"{name}_optimized.{extension}"]

    @staticmethod
    def _infer_onnx_filename(
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
        # hub options
        subfolder: str = "",
        revision: str = "main",
        force_download: bool = False,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        token: Optional[Union[bool, str]] = None,
        # file options
        file_name: Optional[str] = None,
        # session options
        provider: str = "CPUExecutionProvider",
        providers: Optional[Sequence[str]] = None,
        provider_options: Optional[Union[Sequence[Dict[str, Any]], Dict[str, Any]]] = None,
        session_options: Optional[SessionOptions] = None,
        # inference options
        use_io_binding: Optional[bool] = None,
        # other arguments
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
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

        # model_save_dir can be provided in kwargs as a TemporaryDirectory instance,
        # in which case we want to keep it instead of the path only.
        if model_save_dir is None:
            model_save_dir = new_model_save_dir

        providers, provider_options = prepare_providers_and_provider_options(
            provider=provider, providers=providers, provider_options=provider_options
        )
        session = InferenceSession(
            model_cache_path,
            providers=providers,
            provider_options=provider_options,
            sess_options=session_options,
        )

        return cls(
            config=config,
            session=session,
            use_io_binding=use_io_binding,
            model_save_dir=model_save_dir,
        )

    @classmethod
    def _export(
        cls,
        model_id: Union[str, Path],
        config: "PretrainedConfig",
        # hub options
        subfolder: str = "",
        revision: str = "main",
        force_download: bool = False,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        token: Optional[Union[bool, str]] = None,
        # other arguments
        **kwargs,
    ) -> "ORTModel":
        # this is garanteed to work since we it uses a mapping from model classes to task names
        # instead of relying on the hub metadata or the model configuration
        task = TasksManager._infer_task_from_model_or_model_class(model_class=cls.auto_model_class)

        if kwargs.get("task", None) is not None:
            raise ValueError(
                f"The `task` argument is not needed when exporting a model with `{cls.__name__}`. "
                f"The `task` is automatically inferred from the class as `{task}`."
            )

        model_save_dir = TemporaryDirectory()
        model_save_path = Path(model_save_dir.name)

        main_export(
            model_name_or_path=model_id,
            output=model_save_path,
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
            library_name=cls._library_name,
        )
        maybe_save_preprocessors(model_id, model_save_path, src_subfolder=subfolder)

        return cls._from_pretrained(model_save_path, config, model_save_dir=model_save_dir, **kwargs)

    @classmethod
    @add_start_docstrings(FROM_PRETRAINED_START_DOCSTRING)
    def from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: Optional["PretrainedConfig"] = None,
        # export options
        export: bool = False,
        # hub options
        subfolder: str = "",
        revision: str = "main",
        force_download: bool = False,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        token: Optional[Union[bool, str]] = None,
        # session options
        provider: str = "CPUExecutionProvider",
        providers: Optional[Sequence[str]] = None,
        provider_options: Optional[Union[Sequence[Dict[str, Any]], Dict[str, Any]]] = None,
        session_options: Optional[SessionOptions] = None,
        # inference options
        use_io_binding: Optional[bool] = None,
        **kwargs,
    ) -> Self:
        """
        provider (`str`, defaults to `"CPUExecutionProvider"`):
            ONNX Runtime provider to use for loading the model.
            See https://onnxruntime.ai/docs/execution-providers/ for possible providers.
        providers (`Optional[Sequence[str]]`, defaults to `None`):
            List of execution providers to use for loading the model.
            This argument takes precedence over the `provider` argument.
        provider_options (`Optional[Dict[str, Any]]`, defaults to `None`):
            Provider option dictionaries corresponding to the provider used. See available options
            for each provider: https://onnxruntime.ai/docs/api/c/group___global.html .
        session_options (`Optional[onnxruntime.SessionOptions]`, defaults to `None`),:
            ONNX Runtime session options to use for loading the model.
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
            config=config,
            export=_export,
            force_download=force_download,
            token=token,
            cache_dir=cache_dir,
            subfolder=subfolder,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            revision=revision,
            provider=provider,
            providers=providers,
            provider_options=provider_options,
            session_options=session_options,
            use_io_binding=use_io_binding,
            **kwargs,
        )

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
    _library_name: Optional[str] = "transformers"

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
        return_dict: bool = True,
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
            output_shapes, output_buffers = self._prepare_io_binding(model_inputs)

            # run inference with binding & synchronize in case of multiple CUDA streams
            if self.device.type == "cpu":
                self.session.run_with_iobinding(self._io_binding)
            else:
                self._io_binding.synchronize_inputs()
                self.session.run_with_iobinding(self._io_binding)
                self._io_binding.synchronize_outputs()

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

        if not return_dict:
            return (last_hidden_state,)

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


@add_end_docstrings(ONNX_MODEL_END_DOCSTRING)
class ORTModelForMaskedLM(ORTModel):
    """
    ONNX Model with a MaskedLMOutput for masked language modeling tasks. This class officially supports albert, bert, camembert, convbert, data2vec-text, deberta, deberta_v2, distilbert, electra, flaubert, ibert, mobilebert, roberta, roformer, squeezebert, xlm, xlm_roberta.
    """

    auto_model_class = AutoModelForMaskedLM
    _library_name: Optional[str] = "transformers"

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
        return_dict: bool = True,
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
            output_shapes, output_buffers = self._prepare_io_binding(model_inputs)

            # run inference with binding & synchronize in case of multiple CUDA streams
            if self.device.type == "cpu":
                self.session.run_with_iobinding(self._io_binding)
            else:
                self._io_binding.synchronize_inputs()
                self.session.run_with_iobinding(self._io_binding)
                self._io_binding.synchronize_outputs()

            logits = output_buffers["logits"].view(output_shapes["logits"])
        else:
            onnx_inputs = self._prepare_onnx_inputs(use_torch, model_inputs)
            onnx_outputs = self.model.run(None, onnx_inputs)
            model_outputs = self._prepare_onnx_outputs(use_torch, onnx_outputs)

            logits = model_outputs["logits"]

        if not return_dict:
            return (logits,)

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
    ONNX Model with a QuestionAnsweringModelOutput for extractive question-answering tasks like SQuAD. This class officially supports albert, bart, bert, camembert, convbert, data2vec-text, deberta, deberta_v2, distilbert, electra, flaubert, gptj, ibert, mbart, mobilebert, nystromformer, roberta, roformer, squeezebert, xlm, xlm_roberta.
    """

    auto_model_class = AutoModelForQuestionAnswering
    _library_name: Optional[str] = "transformers"

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
        return_dict: bool = True,
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
            output_shapes, output_buffers = self._prepare_io_binding(model_inputs)

            # run inference with binding & synchronize in case of multiple CUDA streams
            if self.device.type == "cpu":
                self.session.run_with_iobinding(self._io_binding)
            else:
                self._io_binding.synchronize_inputs()
                self.session.run_with_iobinding(self._io_binding)
                self._io_binding.synchronize_outputs()

            start_logits = output_buffers["start_logits"].view(output_shapes["start_logits"])
            end_logits = output_buffers["end_logits"].view(output_shapes["end_logits"])
        else:
            onnx_inputs = self._prepare_onnx_inputs(use_torch, model_inputs)
            onnx_outputs = self.model.run(None, onnx_inputs)
            model_outputs = self._prepare_onnx_outputs(use_torch, onnx_outputs)

            start_logits = model_outputs["start_logits"]
            end_logits = model_outputs["end_logits"]

        if not return_dict:
            return (start_logits, end_logits)

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
    pooled output) e.g. for GLUE tasks. This class officially supports albert, bart, bert, camembert, convbert, data2vec-text, deberta, deberta_v2, distilbert, electra, flaubert, ibert, mbart, mobilebert, nystromformer, roberta, roformer, squeezebert, xlm, xlm_roberta.
    """

    auto_model_class = AutoModelForSequenceClassification
    _library_name: Optional[str] = "transformers"

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
        return_dict: bool = True,
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
            output_shapes, output_buffers = self._prepare_io_binding(model_inputs)

            # run inference with binding & synchronize in case of multiple CUDA streams
            if self.device.type == "cpu":
                self.session.run_with_iobinding(self._io_binding)
            else:
                self._io_binding.synchronize_inputs()
                self.session.run_with_iobinding(self._io_binding)
                self._io_binding.synchronize_outputs()

            logits = output_buffers["logits"].view(output_shapes["logits"])
        else:
            onnx_inputs = self._prepare_onnx_inputs(use_torch, model_inputs)
            onnx_outputs = self.model.run(None, onnx_inputs)
            model_outputs = self._prepare_onnx_outputs(use_torch, onnx_outputs)

            logits = model_outputs["logits"]

        if not return_dict:
            return (logits,)

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
    for Named-Entity-Recognition (NER) tasks. This class officially supports albert, bert, bloom, camembert, convbert, data2vec-text, deberta, deberta_v2, distilbert, electra, flaubert, gpt2, ibert, mobilebert, roberta, roformer, squeezebert, xlm, xlm_roberta.

    """

    auto_model_class = AutoModelForTokenClassification
    _library_name: Optional[str] = "transformers"

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
        return_dict: bool = True,
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
            output_shapes, output_buffers = self._prepare_io_binding(model_inputs)

            # run inference with binding & synchronize in case of multiple CUDA streams
            if self.device.type == "cpu":
                self.session.run_with_iobinding(self._io_binding)
            else:
                self._io_binding.synchronize_inputs()
                self.session.run_with_iobinding(self._io_binding)
                self._io_binding.synchronize_outputs()

            logits = output_buffers["logits"].view(output_shapes["logits"])
        else:
            onnx_inputs = self._prepare_onnx_inputs(use_torch, model_inputs)
            onnx_outputs = self.model.run(None, onnx_inputs)
            model_outputs = self._prepare_onnx_outputs(use_torch, onnx_outputs)

            logits = model_outputs["logits"]

        if not return_dict:
            return (logits,)

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
    softmax) e.g. for RocStories/SWAG tasks. This class officially supports albert, bert, camembert, convbert, data2vec-text, deberta_v2, distilbert, electra, flaubert, ibert, mobilebert, nystromformer, roberta, roformer, squeezebert, xlm, xlm_roberta.
    """

    auto_model_class = AutoModelForMultipleChoice
    _library_name: Optional[str] = "transformers"

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
        return_dict: bool = True,
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
            output_shapes, output_buffers = self._prepare_io_binding(model_inputs)

            # run inference with binding & synchronize in case of multiple CUDA streams
            if self.device.type == "cpu":
                self.session.run_with_iobinding(self._io_binding)
            else:
                self._io_binding.synchronize_inputs()
                self.session.run_with_iobinding(self._io_binding)
                self._io_binding.synchronize_outputs()

            logits = output_buffers["logits"].view(output_shapes["logits"])
        else:
            onnx_inputs = self._prepare_onnx_inputs(use_torch, model_inputs)
            onnx_outputs = self.model.run(None, onnx_inputs)
            model_outputs = self._prepare_onnx_outputs(use_torch, onnx_outputs)

            logits = model_outputs["logits"]

        if not return_dict:
            return (logits,)

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
    ONNX Model for image-classification tasks. This class officially supports beit, convnext, convnextv2, data2vec-vision, deit, dinov2, levit, mobilenet_v1, mobilenet_v2, mobilevit, poolformer, resnet, segformer, swin, swinv2, vit.
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
        return_dict: bool = True,
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
            output_shapes, output_buffers = self._prepare_io_binding(model_inputs)

            # run inference with binding & synchronize in case of multiple CUDA streams
            if self.device.type == "cpu":
                self.session.run_with_iobinding(self._io_binding)
            else:
                self._io_binding.synchronize_inputs()
                self.session.run_with_iobinding(self._io_binding)
                self._io_binding.synchronize_outputs()

            logits = output_buffers["logits"].view(output_shapes["logits"])
        else:
            onnx_inputs = self._prepare_onnx_inputs(use_torch, model_inputs)
            onnx_outputs = self.model.run(None, onnx_inputs)
            model_outputs = self._prepare_onnx_outputs(use_torch, onnx_outputs)

            logits = model_outputs["logits"]

        if not return_dict:
            return (logits,)

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
    _library_name: Optional[str] = "transformers"

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
        return_dict: bool = True,
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
            output_shapes, output_buffers = self._prepare_io_binding(model_inputs)

            # run inference with binding & synchronize in case of multiple CUDA streams
            if self.device.type == "cpu":
                self.session.run_with_iobinding(self._io_binding)
            else:
                self._io_binding.synchronize_inputs()
                self.session.run_with_iobinding(self._io_binding)
                self._io_binding.synchronize_outputs()

            logits = output_buffers["logits"].view(output_shapes["logits"])
        else:
            onnx_inputs = self._prepare_onnx_inputs(use_torch, model_inputs)
            onnx_outputs = self.model.run(None, onnx_inputs)
            model_outputs = self._prepare_onnx_outputs(use_torch, onnx_outputs)

            logits = model_outputs["logits"]

        if not return_dict:
            return (logits,)

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
    SUPERB Keyword Spotting. This class officially supports audio_spectrogram_transformer, data2vec-audio, hubert, sew, sew-d, unispeech, unispeech_sat, wavlm, wav2vec2, wav2vec2-conformer.
    """

    auto_model_class = AutoModelForAudioClassification
    _library_name: Optional[str] = "transformers"

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
        return_dict: bool = True,
        **kwargs,
    ):
        # Warn about any unexpected kwargs using the helper method
        self._warn_on_unhandled_inputs(kwargs)

        if self.config.model_type == "whisper":
            assert input_features is not None, "input_features must be provided for this model"
            input_name = "input_features"
            model_input = input_features
        else:
            assert input_values is not None, "input_values must be provided for this model"
            input_name = "input_values"
            model_input = input_values

        use_torch = isinstance(model_input, torch.Tensor)
        self.raise_on_numpy_input_io_binding(use_torch)

        model_inputs = {
            input_name: model_input,
            "attention_mask": attention_mask,
        }

        if self.use_io_binding:
            output_shapes, output_buffers = self._prepare_io_binding(model_inputs)

            # run inference with binding & synchronize in case of multiple CUDA streams
            if self.device.type == "cpu":
                self.session.run_with_iobinding(self._io_binding)
            else:
                self._io_binding.synchronize_inputs()
                self.session.run_with_iobinding(self._io_binding)
                self._io_binding.synchronize_outputs()

            logits = output_buffers["logits"].view(output_shapes["logits"])
        else:
            onnx_inputs = self._prepare_onnx_inputs(use_torch, model_inputs)
            onnx_outputs = self.model.run(None, onnx_inputs)
            model_outputs = self._prepare_onnx_outputs(use_torch, onnx_outputs)

            logits = model_outputs["logits"]

        if not return_dict:
            return (logits,)

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
    ONNX Model with a language modeling head on top for Connectionist Temporal Classification (CTC). This class officially supports data2vec-audio, hubert, sew, sew-d, unispeech, unispeech_sat, wavlm, wav2vec2, wav2vec2-conformer.
    """

    auto_model_class = AutoModelForCTC
    _library_name: Optional[str] = "transformers"

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
        return_dict: bool = True,
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
            sequence_length = input_values.shape[-1]

            for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
                sequence_length = (sequence_length - kernel_size) // stride + 1

            known_output_shapes = {"logits": [batch_size, sequence_length, self.config.vocab_size]}

            output_shapes, output_buffers = self._prepare_io_binding(
                model_inputs, known_output_shapes=known_output_shapes
            )

            # run inference with binding & synchronize in case of multiple CUDA streams
            if self.device.type == "cpu":
                self.session.run_with_iobinding(self._io_binding)
            else:
                self._io_binding.synchronize_inputs()
                self.session.run_with_iobinding(self._io_binding)
                self._io_binding.synchronize_outputs()

            logits = output_buffers["logits"].view(output_shapes["logits"])
        else:
            onnx_inputs = self._prepare_onnx_inputs(use_torch, model_inputs)
            onnx_outputs = self.model.run(None, onnx_inputs)
            model_outputs = self._prepare_onnx_outputs(use_torch, onnx_outputs)

            logits = model_outputs["logits"]

        if not return_dict:
            return (logits,)

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
    ONNX Model with an XVector feature extraction head on top for tasks like Speaker Verification. This class officially supports data2vec-audio, unispeech_sat, wavlm, wav2vec2, wav2vec2-conformer.
    """

    auto_model_class = AutoModelForAudioXVector
    _library_name: Optional[str] = "transformers"

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
        return_dict: bool = True,
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
            output_shapes, output_buffers = self._prepare_io_binding(model_inputs)

            # run inference with binding & synchronize in case of multiple CUDA streams
            if self.device.type == "cpu":
                self.session.run_with_iobinding(self._io_binding)
            else:
                self._io_binding.synchronize_inputs()
                self.session.run_with_iobinding(self._io_binding)
                self._io_binding.synchronize_outputs()

            logits = output_buffers["logits"].view(output_shapes["logits"])
            embeddings = output_buffers["embeddings"].view(output_shapes["embeddings"])

        else:
            onnx_inputs = self._prepare_onnx_inputs(use_torch, model_inputs)
            onnx_outputs = self.model.run(None, onnx_inputs)
            model_outputs = self._prepare_onnx_outputs(use_torch, onnx_outputs)

            logits = model_outputs["logits"]
            embeddings = model_outputs["embeddings"]

        if not return_dict:
            return (logits, embeddings)

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
    ONNX Model with a frame classification head on top for tasks like Speaker Diarization. This class officially supports data2vec-audio, unispeech_sat, wavlm, wav2vec2, wav2vec2-conformer.
    """

    auto_model_class = AutoModelForAudioFrameClassification
    _library_name: Optional[str] = "transformers"

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
        return_dict: bool = True,
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
            output_shapes, output_buffers = self._prepare_io_binding(model_inputs)

            # run inference with binding & synchronize in case of multiple CUDA streams
            if self.device.type == "cpu":
                self.session.run_with_iobinding(self._io_binding)
            else:
                self._io_binding.synchronize_inputs()
                self.session.run_with_iobinding(self._io_binding)
                self._io_binding.synchronize_outputs()

            logits = output_buffers["logits"].view(output_shapes["logits"])
        else:
            onnx_inputs = self._prepare_onnx_inputs(use_torch, model_inputs)
            onnx_outputs = self.model.run(None, onnx_inputs)
            model_outputs = self._prepare_onnx_outputs(use_torch, onnx_outputs)

            logits = model_outputs["logits"]

        if not return_dict:
            return (logits,)

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
    _library_name: Optional[str] = "transformers"

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
        return_dict: bool = True,
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

            output_shapes, output_buffers = self._prepare_io_binding(
                model_inputs,
                known_output_shapes={
                    "reconstruction": [
                        batch_size,
                        num_channels,
                        height * self.config.upscale,
                        width * self.config.upscale,
                    ]
                },
            )

            # run inference with binding & synchronize in case of multiple CUDA streams
            if self.device.type == "cpu":
                self.session.run_with_iobinding(self._io_binding)
            else:
                self._io_binding.synchronize_inputs()
                self.session.run_with_iobinding(self._io_binding)
                self._io_binding.synchronize_outputs()

            reconstruction = output_buffers["reconstruction"].view(output_shapes["reconstruction"])
        else:
            onnx_inputs = self._prepare_onnx_inputs(use_torch, model_inputs)
            onnx_outputs = self.model.run(None, onnx_inputs)
            model_outputs = self._prepare_onnx_outputs(use_torch, onnx_outputs)

            reconstruction = model_outputs["reconstruction"]

        if not return_dict:
            return (reconstruction,)

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
            output_shapes, output_buffers = self._prepare_io_binding(model_inputs)

            # run inference with binding & synchronize in case of multiple CUDA streams
            if self.device.type == "cpu":
                self.session.run_with_iobinding(self._io_binding)
            else:
                self._io_binding.synchronize_inputs()
                self.session.run_with_iobinding(self._io_binding)
                self._io_binding.synchronize_outputs()

            model_outputs = {name: output_buffers[name].view(shape) for name, shape in output_shapes.items()}
        else:
            onnx_inputs = self._prepare_onnx_inputs(use_torch, model_inputs)
            onnx_outputs = self.model.run(None, onnx_inputs)
            model_outputs = self._prepare_onnx_outputs(use_torch, onnx_outputs)

        # converts output to namedtuple for pipelines post-processing
        return ModelOutput(**model_outputs)
