import logging
import os
import shutil
from pathlib import Path
from typing import Optional, Union

import torch
from transformers import AutoTokenizer, PretrainedConfig
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_model_forward, default_cache_path
from transformers.modeling_outputs import (
    BaseModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.onnx import FeaturesManager, export

import onnxruntime as ort
from huggingface_hub import HfApi, hf_hub_download
from onnxruntime.transformers.fusion_options import FusionOptions
from onnxruntime.transformers.optimizer import optimize_model
from onnxruntime.transformers.quantize_helper import QuantizeHelper

from ..modeling_base import OptimizedModel
from .utils import ONNX_WEIGHTS_NAME, _is_gpu_available


logger = logging.getLogger(__name__)


_TOKENIZER_FOR_DOC = "AutoTokenizer"

ONNX_MODEL_START_DOCSTRING = r"""
    This model inherits from [`ORTModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving)
    Parameters:
        config ([`PretrainedConfig`](https://huggingface.co/docs/transformers/main_classes/configuration#transformers.PretrainedConfig)): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~ORTModel.from_pretrained`] method to load the model weights.
        model ([`onnxruntime.InferenceSession`](https://onnxruntime.ai/docs/api/python/api_summary.html#inferencesession)): This is the main class used to run a model. Check out the [`~ORTModel.load_model`]
        for more information.
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

    def __init__(self, model=None, config=None, **kwargs):
        self.model = model
        self.config = config
        self.model_save_dir = kwargs.get("model_save_dir", None)
        self.latest_model_name = kwargs.get("latest_model_name", "model.onnx")

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def optimize(
        self,
        input_path: Union[str, Path] = None,
        model_type: str = "bert",
        output_path: Union[str, Path] = default_cache_path,
        opt_level: int = 1,
        **kwargs,
    ):
        """
        optimizes the mode using onnxruntime.tools.transformers.optimize
        Arguments:
            input_path (:obj:`str` or :obj:`Path`):
                Directory from which to load
            output_path(:obj:`str`):
                Directory where to save the quantized model should be saved, default to
                `transformers.file_utils.default_cache_path`, which is the cache dir for transformers.
        """
        _input_path = None
        output_path = Path(output_path).joinpath("optimized_model.onnx")

        if input_path is None and self.model_save_dir is None:
            raise ValueError("Need to provide input_path wer model is stored. Couldn't find cached model file")
        elif input_path is None and self.model_save_dir is not None:
            _input_path = self.model_save_dir.joinpath(self.latest_model_name)
        else:
            _input_path = Path(input_path)

        # TODO: Reenable when fixed: https://github.com/microsoft/onnxruntime/issues/9573
        optimization_options = FusionOptions(model_type)
        optimization_options.enable_embed_layer_norm = False

        # optimize onnx model using onnxruntime.tools.transformers.optimize
        optimized_model = optimize_model(
            input=_input_path.as_posix(),
            model_type=model_type,
            num_heads=self.config.n_heads,
            hidden_size=self.config.dim,
            opt_level=opt_level,
            optimization_options=optimization_options,
        )
        # converts float32 to float16 for GPU models
        if _is_gpu_available():
            optimized_model.convert_float_to_float16()

        # save optimized model
        optimized_model.save_model_to_file(output_path.as_posix())
        # load optimized model as class instance and updates current instance
        self.model = ORTModel.load_model(output_path.as_posix())
        self.model_save_dir = output_path.parent
        self.latest_model_name = output_path.name

        return self

    def quantize(
        self,
        input_path: Union[str, Path] = None,
        output_path: Union[str, Path] = default_cache_path,
    ):
        """
        quantizes the mode using onnxruntime.tools.transformers.optimize
        Arguments:
            input_path (:obj:`str` or :obj:`Path`):
                Directory from which to load
            output_path(:obj:`str`):
                Directory where to save the quantized model should be saved, default to
                `transformers.file_utils.default_cache_path`, which is the cache dir for transformers.
        """
        if _is_gpu_available():
            raise ValueError("GPU is not supported for quantization")

        _input_path = None
        output_path = Path(output_path).joinpath("q8_model.onnx")

        if input_path is None and self.model_save_dir is None:
            raise ValueError("Need to provide input_path wer model is stored. Couldn't find cached model file")
        elif input_path is None and self.model_save_dir is not None:
            _input_path = self.model_save_dir.joinpath(self.latest_model_name)
        else:
            _input_path = Path(input_path)

        QuantizeHelper.quantize_onnx_model(
            _input_path.as_posix(),
            output_path.as_posix(),
        )
        self.model = ORTModel.load_model(output_path.as_posix())
        self.model_save_dir = output_path.parent
        self.latest_model_name = output_path.name

        return self

    @staticmethod
    def load_model(path: Union[str, Path], provider=None):
        """
        loads ONNX Inference session with Provider. Default Provider is if GPU available else `CPUExecutionProvider`
        Arguments:
            path (:obj:`str` or :obj:`Path`):
                Directory from which to load
            provider(:obj:`str`):
                Onnxruntime provider to use for loading the model, defaults to `CUDAExecutionProvider` if GPU is
                available else `CPUExecutionProvider`
        """
        if provider is None:
            provider = "CUDAExecutionProvider" if _is_gpu_available() else "CPUExecutionProvider"

        return ort.InferenceSession(path, providers=[provider])

    def _save_pretrained(self, save_directory: Union[str, Path], file_name: Optional[str] = None, **kwargs):
        """
        Save a model and its configuration file to a directory, so that it can be re-loaded using the
        `:func:`~optimum.onnxruntime.modeling_ort.ORTModel.from_pretrained`` class method. It will always save the latest_model_name.
        Arguments:
            save_directory (:obj:`str` or :obj:`Path`):
                Directory where to save the model file.
            file_name(:obj:`str`):
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
        force_download: bool = True,
        cache_dir: Optional[str] = None,
        file_name: Optional[str] = None,
        **kwargs,
    ):
        """
        load a model and its configuration file from a directory or the HF Hub.
        Implements: https://github.com/huggingface/huggingface_hub/blob/e67de48368bc1843e40afc1cc9d236402b9609ee/src/huggingface_hub/hub_mixin.py#L73
        Arguments:
            model_id (:obj:`str` or :obj:`Path`):
                Directory from which to load
            use_auth_token (:obj:`str` or :obj:`bool`):
                Is needed to load models from a private repository
            revision (:obj:`str`):
                Revision is the specific model version to use. It can be a branch name, a tag name, or a commit id
            cache_dir (:obj:`Union[str, Path]`, `optional`):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            file_name(:obj:`str`):
                Overwrites the default model file name from `"model.onnx"` to `file_name`. This allows you to load different model files from the same
                repository or directory.
            kwargs (:obj:`Dict`, `optional`)::
                kwargs will be passed to the model during initialization
        """
        config_dict = kwargs.pop("config", {})
        model_file_name = file_name if file_name is not None else ONNX_WEIGHTS_NAME
        # load model from local directory
        if os.path.isdir(model_id):
            config = PretrainedConfig.from_dict(config_dict)
            model = ORTModel.load_model(os.path.join(model_id, model_file_name))
            kwargs["model_save_dir"] = Path(model_id)
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
        force_download: bool = True,
        cache_dir: Optional[str] = None,
        **kwargs,
    ):
        """
        Converts a vanilla Transformers model into an optimized model using `transformers.onnx.export_onnx`.
        Arguments:
            model_id (:obj:`str` or :obj:`Path`):
                Directory from which to load
            save_dir (:obj:`str` or :obj:`Path`):
                Directory where the onnx model should be saved, default to `transformers.file_utils.default_cache_path`, which is the cache dir for
                transformers.
            use_auth_token (:obj:`str` or :obj:`bool`):
                Is needed to load models from a private repository
            revision (:obj:`str`):
                Revision is the specific model version to use. It can be a branch name, a tag name, or a commit id
            cache_dir (:obj:`Union[str, Path]`, `optional`):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            kwargs (:obj:`Dict`, `optional`)::
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
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = FeaturesManager.get_model_from_feature(task, model_id)
        _, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature=task)
        onnx_config = model_onnx_config(model.config)

        # export model
        export(
            preprocessor=tokenizer,
            model=model,
            config=onnx_config,
            opset=onnx_config.default_onnx_opset,
            output=save_dir.joinpath(ONNX_WEIGHTS_NAME),
        )
        kwargs["config"] = model.config.__dict__
        # 3. load normal model
        return cls._from_pretrained(save_dir.as_posix(), **kwargs)


FEAUTRE_EXTRACTION_SAMPLE = r"""
    Example of feature extraction:

    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.onnxruntime import {model_class}
    >>> import torch

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("My Name is Philipp and i live in Germany.", return_tensors="pt")

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
    >>> onnx_ner = pipeline("feature-extraction", model=model, tokenizer=tokenizer)

    >>> text = "My Name is Philipp and i live in Germany."
    >>> pred = onnx_ner(text)
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # create {name:idx} dict for model outputs
        self.model_outputs = {output_key.name: idx for idx, output_key in enumerate(self.model.get_outputs())}

    @add_start_docstrings_to_model_forward(
        ONNX_INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + FEAUTRE_EXTRACTION_SAMPLE.format(
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
        # converts output to namedtuple for pipelines post-processing
        return BaseModelOutput(
            last_hidden_state=torch.from_numpy(outputs[self.model_outputs["last_hidden_state"]]),
        )


QUESTION_ANSWERING_SAMPLE = r"""
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
    Example using `transformers.pipelines`:

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # create {name:idx} dict for model outputs
        self.model_outputs = {output_key.name: idx for idx, output_key in enumerate(self.model.get_outputs())}

    @add_start_docstrings_to_model_forward(
        ONNX_INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + QUESTION_ANSWERING_SAMPLE.format(
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
        # converts output to namedtuple for pipelines post-processing
        return QuestionAnsweringModelOutput(
            start_logits=torch.from_numpy(outputs[self.model_outputs["start_logits"]]),
            end_logits=torch.from_numpy(outputs[self.model_outputs["end_logits"]]),
        )


SEQUENCE_CLASSIFICATION_SAMPLE = r"""
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
    >>> onnx_clx = pipeline("text-classification", model=model, tokenizer=tokenizer)

    >>> text = "Hello, my dog is cute"
    >>> pred = onnx_clx(text)
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # create {name:idx} dict for model outputs
        self.model_outputs = {output_key.name: idx for idx, output_key in enumerate(self.model.get_outputs())}
        self.model_inputs = {output_key.name: idx for idx, output_key in enumerate(self.model.get_inputs())}

    @add_start_docstrings_to_model_forward(
        ONNX_INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + SEQUENCE_CLASSIFICATION_SAMPLE.format(
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
        # converts output to namedtuple for pipelines post-processing
        return SequenceClassifierOutput(
            logits=torch.from_numpy(outputs[self.model_outputs["logits"]]),
        )


TOKEN_CLASSIFICATION_SAMPLE = r"""
    Example of token classification:

    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.onnxruntime import {model_class}
    >>> import torch

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("My Name is Philipp and i live in Germany.", return_tensors="pt")

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

    >>> text = "My Name is Philipp and i live in Germany."
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
    Sequence Classification model for ONNX.
    """

    # used in from_transformers to export model to onnx
    pipeline_task = "token-classification"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # create {name:idx} dict for model outputs
        self.model_outputs = {output_key.name: idx for idx, output_key in enumerate(self.model.get_outputs())}

    @add_start_docstrings_to_model_forward(
        ONNX_INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + TOKEN_CLASSIFICATION_SAMPLE.format(
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
        # converts output to namedtuple for pipelines post-processing
        return TokenClassifierOutput(
            logits=torch.from_numpy(outputs[self.model_outputs["logits"]]),
        )
