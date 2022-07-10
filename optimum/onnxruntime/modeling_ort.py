import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageClassification,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    PretrainedConfig,
)
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_model_forward, default_cache_path
from transformers.generation_utils import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutput,
    CausalLMOutputWithCrossAttentions,
    ImageClassifierOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.onnx import FeaturesManager, export
from transformers.onnx.utils import get_preprocessor

import onnxruntime as ort
from huggingface_hub import HfApi, hf_hub_download

from ..modeling_base import OptimizedModel
from .utils import ONNX_WEIGHTS_NAME, get_device_for_provider, get_provider_for_device, _is_gpu_available

logger = logging.getLogger(__name__)


_TOKENIZER_FOR_DOC = "AutoTokenizer"
_FEATURE_EXTRACTOR_FOR_DOC = "AutoFeatureExtractor"

ONNX_MODEL_START_DOCSTRING = r"""
    This model inherits from [~`onnxruntime.modeling_ort.ORTModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving)
    Parameters:
        config (`transformers.PretrainedConfig`): [PretrainedConfig](https://huggingface.co/docs/transformers/main_classes/configuration#transformers.PretrainedConfig) is the Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~onnxruntime.modeling_ort.ORTModel.from_pretrained`] method to load the model weights.
        model (`onnxruntime.InferenceSession`): [onnxruntime.InferenceSession](https://onnxruntime.ai/docs/api/python/api_summary.html#inferencesession) is the main class used to run a model. Check out the [`~onnxruntime.modeling_ort.ORTModel.load_model`] method for more information.
"""

ONNX_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.Tensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.
            Indices can be obtained using [`AutoTokenizer`](https://huggingface.co/docs/transformers/autoclass_tutorial#autotokenizer).
            See [`PreTrainedTokenizer.encode`](https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizerBase.encode) and
            [`PreTrainedTokenizer.__call__`](https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizerBase.__call__) for details.
            [What are input IDs?](https://huggingface.co/docs/transformers/glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](https://huggingface.co/docs/transformers/glossary#attention-mask)
        token_type_ids (`torch.Tensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
            - 1 for tokens that are **sentence A**,
            - 0 for tokens that are **sentence B**.
            [What are token type IDs?](https://huggingface.co/docs/transformers/glossary#token-type-ids)
"""


@add_start_docstrings(
    """
    Base ORTModel class for implementing models using ONNX Runtime. The ORTModel implements generic methods for interacting
    with the Hugging Face Hub as well as exporting vanilla transformers models to ONNX using `transformers.onnx` toolchain.
    The ORTModel implements additionally generic methods for optimizing and quantizing Onnx models.
    """,
)
class ORTModel(OptimizedModel):
    base_model_prefix = "onnx_model"
    auto_model_class = AutoModel

    def __init__(self, model=None, config=None, **kwargs):
        self.model = model
        self.config = config
        self.model_save_dir = kwargs.get("model_save_dir", None)
        self.latest_model_name = kwargs.get("latest_model_name", "model.onnx")
        self._device = get_device_for_provider(self.model.get_providers()[0])

        # registers the ORTModelForXXX classes into the transformers AutoModel classes
        # to avoid warnings when create a pipeline https://github.com/huggingface/transformers/blob/cad61b68396a1a387287a8e2e2fef78a25b79383/src/transformers/pipelines/base.py#L863
        AutoConfig.register(self.base_model_prefix, AutoConfig)
        self.auto_model_class.register(AutoConfig, self.__class__)

    @property
    def device(self) -> torch.device:
        """
        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """
        return self._device

    @device.setter
    def device(self, value):
        self._device = value

    def to(self, device):
        """
        Changes the ONNX Runtime provider according to the device.
        """
        self.device = device
        provider = get_provider_for_device(self.device)
        self.model.set_providers([provider])
        return self

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def load_model(path: Union[str, Path], provider=None):
        """
        Loads an ONNX Inference session with a given provider. Default provider is `CPUExecutionProvider` to match the default behaviour in PyTorch/TensorFlow/JAX.

        Arguments:
            path (`str` or `Path`):
                Directory from which to load the model.
            provider(`str`, *optional*):
                ONNX Runtime provider to use for loading the model. Defaults to `CPUExecutionProvider`.
        """
        if provider is None:
            provider = "CUDAExecutionProvider" if _is_gpu_available() else "CPUExecutionProvider"

        return ort.InferenceSession(path, providers=[provider])

    def _save_pretrained(self, save_directory: Union[str, Path], file_name: Optional[str] = None, **kwargs):
        """
        Saves a model and its configuration file to a directory, so that it can be re-loaded using the
        [`~optimum.onnxruntime.modeling_ort.ORTModel.from_pretrained`] class method. It will always save the latest_model_name.
        Arguments:
            save_directory (`str` or `Path`):
                Directory where to save the model file.
            file_name(`str`, *optional*):
                Overwrites the default model file name from `"model.onnx"` to `file_name`. This allows you to save the model with
                a different name.
        """
        model_file_name = file_name if file_name is not None else ONNX_WEIGHTS_NAME

        src_path = self.model_save_dir.joinpath(self.latest_model_name)
        dst_path = Path(save_directory).joinpath(model_file_name)
        shutil.copyfile(src_path, dst_path)

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        use_auth_token: Optional[Union[bool, str, None]] = None,
        revision: Optional[Union[str, None]] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        file_name: Optional[str] = None,
        **kwargs,
    ):
        """
        Loads a model and its configuration file from a directory or the HF Hub.
        Implements: https://github.com/huggingface/huggingface_hub/blob/e67de48368bc1843e40afc1cc9d236402b9609ee/src/huggingface_hub/hub_mixin.py#L73
        Arguments:
            model_id (`str` or `Path`):
                Directory from which to load
            use_auth_token (`str` or `bool`):
                Is needed to load models from a private repository
            revision (`str`):
                Revision is the specific model version to use. It can be a branch name, a tag name, or a commit id
            cache_dir (`Union[str, Path]`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            file_name(`str`):
                Overwrites the default model file name from `"model.onnx"` to `file_name`. This allows you to load different model files from the same
                repository or directory.
            kwargs (`Dict`, *optional*):
                kwargs will be passed to the model during initialization
        """
        config_dict = kwargs.pop("config", {})
        model_file_name = file_name if file_name is not None else ONNX_WEIGHTS_NAME
        # load model from local directory
        if os.path.isdir(model_id):
            config = PretrainedConfig.from_dict(config_dict)
            model = ORTModel.load_model(os.path.join(model_id, model_file_name))
            kwargs["model_save_dir"] = Path(model_id)
            kwargs["latest_model_name"] = model_file_name
        # load model from hub
        else:
            # download model
            model_cache_path = hf_hub_download(
                repo_id=model_id,
                filename=model_file_name,
                use_auth_token=use_auth_token,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
            )
            kwargs["model_save_dir"] = Path(model_cache_path).parent
            kwargs["latest_model_name"] = Path(model_cache_path).name
            model = ORTModel.load_model(model_cache_path)
            config = PretrainedConfig.from_dict(config_dict)
        return cls(model=model, config=config, **kwargs)

    @classmethod
    def _from_transformers(
        cls,
        model_id: str,
        save_dir: Union[str, Path] = default_cache_path,
        use_auth_token: Optional[Union[bool, str, None]] = None,
        revision: Optional[Union[str, None]] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        **kwargs,
    ):
        """
        Converts a vanilla Transformers model into an optimized model using `transformers.onnx.export_onnx`.
        Arguments:
            model_id (`str` or `Path`):
                Directory from which to load
            save_dir (`str` or `Path`):
                Directory where the onnx model should be saved, default to `transformers.file_utils.default_cache_path`, which is the cache dir for
                transformers.
            use_auth_token (`str` or `bool`):
                Is needed to load models from a private repository
            revision (`str`):
                Revision is the specific model version to use. It can be a branch name, a tag name, or a commit id
            cache_dir (`Union[str, Path]`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            kwargs (`Dict`, *optional*):
                kwargs will be passed to the model during initialization
        """

        # create local save dir in cache dir
        save_dir = Path(save_dir).joinpath(model_id)
        save_dir.mkdir(parents=True, exist_ok=True)
        kwargs["model_save_dir"] = save_dir

        # reads pipeline task from ORTModelForXXX class if available else tries to extract from hub
        if cls.pipeline_task is not None:
            task = cls.pipeline_task
        else:
            task = HfApi().model_info(model_id, revision=revision).pipeline_tag
            if task in ["sentiment-analysis", "text-classification", "zero-shot-classification"]:
                task = "sequence-classification"
            elif task in ["feature-extraction", "fill-mask"]:
                task = "default"
        # 2. convert to temp dir
        # FIXME: transformers.onnx conversion doesn't support private models
        preprocessor = get_preprocessor(model_id)
        model = FeaturesManager.get_model_from_feature(task, model_id)
        _, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature=task)
        onnx_config = model_onnx_config(model.config)

        # export model
        export(
            preprocessor=preprocessor,
            model=model,
            config=onnx_config,
            opset=onnx_config.default_onnx_opset,
            output=save_dir.joinpath(ONNX_WEIGHTS_NAME),
        )
        kwargs["config"] = model.config.__dict__
        # 3. load normal model
        return cls._from_pretrained(save_dir.as_posix(), **kwargs)


FEAUTRE_EXTRACTION_EXAMPLE = r"""
    Example of feature extraction:

    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.onnxruntime import {model_class}
    >>> import torch

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("My name is Philipp and I live in Germany.", return_tensors="pt")

    >>> outputs = model(**inputs)
    >>> logits = outputs.logits
    >>> list(logits.shape)
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
    Onnx Model with a MaskedLMOutput for feature-extraction tasks.
    """,
    ONNX_MODEL_START_DOCSTRING,
)
class ORTModelForFeatureExtraction(ORTModel):
    """
    Feature Extraction model for ONNX.
    """

    # used in from_transformers to export model to onnx
    pipeline_task = "default"
    auto_model_class = AutoModel

    def __init__(self, model=None, config=None, **kwargs):
        super().__init__(model, config, **kwargs)
        # create {name:idx} dict for model outputs
        self.model_outputs = {output_key.name: idx for idx, output_key in enumerate(self.model.get_outputs())}

    @add_start_docstrings_to_model_forward(
        ONNX_INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + FEAUTRE_EXTRACTION_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="ORTModelForFeatureExtraction",
            checkpoint="optimum/all-MiniLM-L6-v2",
        )
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # converts pytorch inputs into numpy inputs for onnx
        onnx_inputs = {
            "input_ids": input_ids.cpu().detach().numpy(),
            "attention_mask": attention_mask.cpu().detach().numpy(),
        }
        if token_type_ids is not None:
            onnx_inputs["token_type_ids"] = token_type_ids.cpu().detach().numpy()
        # run inference
        outputs = self.model.run(None, onnx_inputs)
        last_hidden_state = torch.from_numpy(outputs[self.model_outputs["last_hidden_state"]]).to(self.device)
        # converts output to namedtuple for pipelines post-processing
        return BaseModelOutput(last_hidden_state=last_hidden_state)


QUESTION_ANSWERING_EXAMPLE = r"""
    Example of question answering:

    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.onnxruntime import {model_class}
    >>> import torch

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
    >>> inputs = tokenizer(question, text, return_tensors="pt")
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

    # used in from_transformers to export model to onnx
    pipeline_task = "question-answering"
    auto_model_class = AutoModelForQuestionAnswering

    def __init__(self, model=None, config=None, **kwargs):
        super().__init__(model, config, **kwargs)
        # create {name:idx} dict for model outputs
        self.model_outputs = {output_key.name: idx for idx, output_key in enumerate(self.model.get_outputs())}

    @add_start_docstrings_to_model_forward(
        ONNX_INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + QUESTION_ANSWERING_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="ORTModelForQuestionAnswering",
            checkpoint="optimum/roberta-base-squad2",
        )
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # converts pytorch inputs into numpy inputs for onnx
        onnx_inputs = {
            "input_ids": input_ids.cpu().detach().numpy(),
            "attention_mask": attention_mask.cpu().detach().numpy(),
        }
        if token_type_ids is not None:
            onnx_inputs["token_type_ids"] = token_type_ids.cpu().detach().numpy()
        # run inference
        outputs = self.model.run(None, onnx_inputs)
        start_logits = torch.from_numpy(outputs[self.model_outputs["start_logits"]]).to(self.device)
        end_logits = torch.from_numpy(outputs[self.model_outputs["end_logits"]]).to(self.device)
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

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

    >>> outputs = model(**inputs)
    >>> logits = outputs.logits
    >>> list(logits.shape)
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
    >>> pred = onnx_z0(sequence_to_classify, candidate_labels, multi_class=True)
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

    # used in from_transformers to export model to onnx
    pipeline_task = "sequence-classification"
    auto_model_class = AutoModelForSequenceClassification

    def __init__(self, model=None, config=None, **kwargs):
        super().__init__(model, config, **kwargs)
        # create {name:idx} dict for model outputs
        self.model_outputs = {output_key.name: idx for idx, output_key in enumerate(self.model.get_outputs())}
        self.model_inputs = {output_key.name: idx for idx, output_key in enumerate(self.model.get_inputs())}

    @add_start_docstrings_to_model_forward(
        ONNX_INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + SEQUENCE_CLASSIFICATION_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="ORTModelForSequenceClassification",
            checkpoint="optimum/distilbert-base-uncased-finetuned-sst-2-english",
        )
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # converts pytorch inputs into numpy inputs for onnx
        onnx_inputs = {
            "input_ids": input_ids.cpu().detach().numpy(),
            "attention_mask": attention_mask.cpu().detach().numpy(),
        }

        if token_type_ids is not None:
            onnx_inputs["token_type_ids"] = token_type_ids.cpu().detach().numpy()
        # run inference
        outputs = self.model.run(None, onnx_inputs)
        logits = torch.from_numpy(outputs[self.model_outputs["logits"]]).to(self.device)
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

    >>> inputs = tokenizer("My name is Philipp and I live in Germany.", return_tensors="pt")

    >>> outputs = model(**inputs)
    >>> logits = outputs.logits
    >>> list(logits.shape)
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

    # used in from_transformers to export model to onnx
    pipeline_task = "token-classification"
    auto_model_class = AutoModelForTokenClassification

    def __init__(self, model=None, config=None, **kwargs):
        super().__init__(model, config, **kwargs)
        # create {name:idx} dict for model outputs
        self.model_outputs = {output_key.name: idx for idx, output_key in enumerate(self.model.get_outputs())}

    @add_start_docstrings_to_model_forward(
        ONNX_INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + TOKEN_CLASSIFICATION_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="ORTModelForTokenClassification",
            checkpoint="optimum/bert-base-NER",
        )
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # converts pytorch inputs into numpy inputs for onnx
        onnx_inputs = {
            "input_ids": input_ids.cpu().detach().numpy(),
            "attention_mask": attention_mask.cpu().detach().numpy(),
        }
        if token_type_ids is not None:
            onnx_inputs["token_type_ids"] = token_type_ids.cpu().detach().numpy()
        # run inference
        outputs = self.model.run(None, onnx_inputs)
        logits = torch.from_numpy(outputs[self.model_outputs["logits"]]).to(self.device)
        # converts output to namedtuple for pipelines post-processing
        return TokenClassifierOutput(logits=logits)


TEXT_GENERATION_EXAMPLE = r"""
    Example of text generation:

    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.onnxruntime import {model_class}
    >>> import torch

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("My name is Philipp and I live in Germany.", return_tensors="pt")

    >>> gen_tokens = model.generate(**inputs,do_sample=True,temperature=0.9, min_length=20,max_length=20)
    >>> tokenizer.batch_decode(gen_tokens)
    ```

    Example using `transformers.pipelines`:

    ```python
    >>> from transformers import {processor_class}, pipeline
    >>> from optimum.onnxruntime import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> onnx_gen = pipeline("text-generation", model=model, tokenizer=tokenizer)

    >>> text = "My name is Philipp and I live in Germany."
    >>> gen = onnx_gen(text)
    ```
"""


@add_start_docstrings(
    """
    Onnx Model with a causal language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    ONNX_MODEL_START_DOCSTRING,
)
class ORTModelForCausalLM(ORTModel, GenerationMixin):
    """
    Causal LM model for ONNX.
    """

    # used in from_transformers to export model to onnx
    pipeline_task = "causal-lm"
    auto_model_class = AutoModelForCausalLM

    def __init__(self, model=None, config=None, **kwargs):
        super().__init__(model, config, **kwargs)
        # create {name:idx} dict for model outputs
        self.main_input_name = "input_ids"
        self.model_outputs = {output_key.name: idx for idx, output_key in enumerate(self.model.get_outputs())}

    def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, **kwargs) -> Dict[str, Any]:
        """
        Implement in subclasses of [`PreTrainedModel`] for custom behavior to prepare inputs in the generate method.
        """
        inputs = {"input_ids": input_ids}
        if kwargs.get("attention_mask", None) is not None:
            inputs["attention_mask"] = kwargs["attention_mask"]
        return inputs

    @add_start_docstrings_to_model_forward(
        ONNX_INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + TEXT_GENERATION_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="ORTModelForCausalLM",
            checkpoint="optimum/gpt2",
        )
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # converts pytorch inputs into numpy inputs for onnx
        onnx_inputs = {
            "input_ids": input_ids.cpu().detach().numpy(),
            "attention_mask": attention_mask.cpu().detach().numpy(),
        }
        # run inference
        outputs = self.model.run(None, onnx_inputs)
        logits = torch.from_numpy(outputs[self.model_outputs["logits"]]).to(self.device)
        # converts output to namedtuple for pipelines post-processing
        return CausalLMOutputWithCrossAttentions(logits=logits)

    # Adapted from https://github.com/huggingface/transformers/blob/99289c08a1b16a805dd4ee46de029e9fd23cba3d/src/transformers/generation_utils.py#L490
    def _prepare_attention_mask_for_generation(
        self,
        inputs: torch.Tensor,
        pad_token_id: int,
        eos_token_id: int,
    ) -> torch.LongTensor:
        """
        Overrides the base method of `GenerationMixin` to ensure input IDs and
        attention mask are on the same device.
        """
        is_input_ids = len(inputs.shape) == 2 and inputs.dtype in [torch.int, torch.long]
        is_pad_token_in_inputs = (pad_token_id is not None) and (pad_token_id in inputs)
        is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or (
            (eos_token_id is not None) and (pad_token_id != eos_token_id)
        )
        # Check if input is input_ids and padded -> only then is attention_mask defined
        if is_input_ids and is_pad_token_in_inputs and is_pad_token_not_equal_to_eos_token_id:
            return inputs.ne(pad_token_id).long()
        else:
            # Ensure attention mask is on the same device as the input IDs
            return torch.ones(inputs.shape[:2], dtype=torch.long, device=inputs.device)


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

    >>> inputs = preprocessor(images=image, return_tensors="pt")

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

    # used in from_transformers to export model to onnx
    pipeline_task = "image-classification"
    auto_model_class = AutoModelForImageClassification

    def __init__(self, model=None, config=None, **kwargs):
        super().__init__(model, config, **kwargs)
        # create {name:idx} dict for model outputs
        self.model_outputs = {output_key.name: idx for idx, output_key in enumerate(self.model.get_outputs())}

    @add_start_docstrings_to_model_forward(
        ONNX_INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + FEAUTRE_EXTRACTION_EXAMPLE.format(
            processor_class=_FEATURE_EXTRACTOR_FOR_DOC,
            model_class="ORTModelForImageClassification",
            checkpoint="optimum/vit-base-patch16-224",
        )
    )
    def forward(
        self,
        pixel_values: torch.Tensor,
        **kwargs,
    ):
        # converts pytorch inputs into numpy inputs for onnx
        onnx_inputs = {
            "pixel_values": pixel_values.cpu().detach().numpy(),
        }
        # run inference
        outputs = self.model.run(None, onnx_inputs)
        # converts output to namedtuple for pipelines post-processing
        return ImageClassifierOutput(
            logits=torch.from_numpy(outputs[self.model_outputs["logits"]]),
        )
