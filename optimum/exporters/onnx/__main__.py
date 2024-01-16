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
"""Entry point to the optimum.exporters.onnx command line."""

import argparse
import os
from pathlib import Path

from packaging import version
from requests.exceptions import ConnectionError as RequestsConnectionError
from transformers import AutoConfig, AutoTokenizer
from transformers.utils import is_torch_available

from ...commands.export.onnx import parse_args_onnx
from ...configuration_utils import _transformers_version
from ...utils import DEFAULT_DUMMY_SHAPES, ONNX_WEIGHTS_NAME, logging
from ...utils.modeling_utils import MODEL_TO_PATCH_FOR_PAST
from ...utils.save_utils import maybe_load_preprocessors, maybe_save_preprocessors
from ..error_utils import AtolError, OutputMatchError, ShapeError
from ..tasks import TasksManager
from .base import OnnxConfigWithPast
from .constants import SDPA_ARCHS_ONNX_EXPORT_NOT_SUPPORTED, UNPICKABLE_ARCHS
from .convert import export_models, validate_models_outputs
from .utils import (
    MODEL_TYPES_REQUIRING_POSITION_IDS,
    _get_submodels_for_export_decoder,
    _get_submodels_for_export_encoder_decoder,
    _get_submodels_for_export_stable_diffusion,
    get_decoder_models_for_export,
    get_encoder_decoder_models_for_export,
    get_sam_models_for_export,
    get_speecht5_models_for_export,
    get_stable_diffusion_models_for_export,
)


if is_torch_available():
    import torch

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union


if TYPE_CHECKING:
    from transformers import PreTrainedModel, TFPreTrainedModel

    from .base import OnnxConfig

logger = logging.get_logger()
logger.setLevel(logging.INFO)


def _get_submodels_and_onnx_configs(
    model: Union["PreTrainedModel", "TFPreTrainedModel"],
    task: str,
    monolith: bool,
    custom_onnx_configs: Dict,
    custom_architecture: bool,
    _variant: str,
    int_dtype: str = "int64",
    float_dtype: str = "fp32",
    fn_get_submodels: Optional[Callable] = None,
    preprocessors: Optional[List[Any]] = None,
    legacy: bool = False,
    library_name: str = "transformers",
    model_kwargs: Optional[Dict] = None,
):
    is_stable_diffusion = "stable-diffusion" in task
    if not custom_architecture:
        if is_stable_diffusion:
            onnx_config = None
            models_and_onnx_configs = get_stable_diffusion_models_for_export(
                model, int_dtype=int_dtype, float_dtype=float_dtype
            )
        else:
            onnx_config_constructor = TasksManager.get_exporter_config_constructor(
                model=model, exporter="onnx", task=task, library_name=library_name
            )
            onnx_config = onnx_config_constructor(
                model.config,
                int_dtype=int_dtype,
                float_dtype=float_dtype,
                preprocessors=preprocessors,
                legacy=legacy,
            )

            onnx_config.variant = _variant
            all_variants = "\n".join(
                [f"    - {name}: {description}" for name, description in onnx_config.VARIANTS.items()]
            )
            logger.info(f"Using the export variant {onnx_config.variant}. Available variants are:\n{all_variants}")

            # TODO: this succession of if/else strongly suggests a refactor is needed.
            if (
                model.config.is_encoder_decoder
                and task.startswith(TasksManager._ENCODER_DECODER_TASKS)
                and not monolith
            ):
                models_and_onnx_configs = get_encoder_decoder_models_for_export(model, onnx_config)
            elif task.startswith("text-generation") and not monolith:
                models_and_onnx_configs = get_decoder_models_for_export(model, onnx_config, legacy=legacy)
            elif model.config.model_type == "sam":
                models_and_onnx_configs = get_sam_models_for_export(model, onnx_config)
            elif model.config.model_type == "speecht5":
                models_and_onnx_configs = get_speecht5_models_for_export(model, onnx_config, model_kwargs)
            else:
                models_and_onnx_configs = {"model": (model, onnx_config)}

        # When specifying custom ONNX configs for supported transformers architectures, we do
        # not force to specify a custom ONNX config for each submodel.
        for key, custom_onnx_config in custom_onnx_configs.items():
            models_and_onnx_configs[key] = (models_and_onnx_configs[key][0], custom_onnx_config)
    else:
        onnx_config = None
        submodels_for_export = None
        models_and_onnx_configs = {}

        if fn_get_submodels is not None:
            submodels_for_export = fn_get_submodels(model)
        else:
            if is_stable_diffusion:
                submodels_for_export = _get_submodels_for_export_stable_diffusion(model)
            elif (
                model.config.is_encoder_decoder
                and task.startswith(TasksManager._ENCODER_DECODER_TASKS)
                and not monolith
            ):
                submodels_for_export = _get_submodels_for_export_encoder_decoder(
                    model, use_past=task.endswith("-with-past")
                )
            elif task.startswith("text-generation") and not monolith:
                submodels_for_export = _get_submodels_for_export_decoder(model, use_past=task.endswith("-with-past"))
            else:
                submodels_for_export = {"model": model}

        if submodels_for_export.keys() != custom_onnx_configs.keys():
            logger.error(f"ONNX custom configs for: {', '.join(custom_onnx_configs.keys())}")
            logger.error(f"Submodels to export: {', '.join(submodels_for_export.keys())}")
            raise ValueError(
                "Trying to export a custom model, but could not find as many custom ONNX configs as the number of submodels to export. Please specifiy the fn_get_submodels argument, that should return a dictionary of submodules with as many items as the provided custom_onnx_configs dictionary."
            )

        for key, custom_onnx_config in custom_onnx_configs.items():
            models_and_onnx_configs[key] = (submodels_for_export[key], custom_onnx_config)

    # Default to the first ONNX config for stable-diffusion and custom architecture case.
    if onnx_config is None:
        onnx_config = next(iter(models_and_onnx_configs.values()))[1]

    return onnx_config, models_and_onnx_configs


def main_export(
    model_name_or_path: str,
    output: Union[str, Path],
    task: str = "auto",
    opset: Optional[int] = None,
    device: str = "cpu",
    fp16: Optional[bool] = False,
    optimize: Optional[str] = None,
    monolith: bool = False,
    no_post_process: bool = False,
    framework: Optional[str] = None,
    atol: Optional[float] = None,
    cache_dir: Optional[str] = None,
    trust_remote_code: bool = False,
    pad_token_id: Optional[int] = None,
    subfolder: str = "",
    revision: str = "main",
    force_download: bool = False,
    local_files_only: bool = False,
    use_auth_token: Optional[Union[bool, str]] = None,
    for_ort: bool = False,
    do_validation: bool = True,
    model_kwargs: Optional[Dict[str, Any]] = None,
    custom_onnx_configs: Optional[Dict[str, "OnnxConfig"]] = None,
    fn_get_submodels: Optional[Callable] = None,
    use_subprocess: bool = False,
    _variant: str = "default",
    library_name: Optional[str] = None,
    legacy: bool = False,
    **kwargs_shapes,
):
    """
    Full-suite ONNX export.

    Args:
        > Required parameters

        model_name_or_path (`str`):
            Model ID on huggingface.co or path on disk to the model repository to export.
        output (`Union[str, Path]`):
            Path indicating the directory where to store the generated ONNX model.

        > Optional parameters

        task (`Optional[str]`, defaults to `None`):
            The task to export the model for. If not specified, the task will be auto-inferred based on the model. For decoder models,
            use `xxx-with-past` to export the model using past key values in the decoder.
        opset (`Optional[int]`, defaults to `None`):
            If specified, ONNX opset version to export the model with. Otherwise, the default opset for the given model architecture
            will be used.
        device (`str`, defaults to `"cpu"`):
            The device to use to do the export. Defaults to "cpu".
        fp16 (`Optional[bool]`, defaults to `"False"`):
            Use half precision during the export. PyTorch-only, requires `device="cuda"`.
        optimize (`Optional[str]`, defaults to `None`):
            Allows to run ONNX Runtime optimizations directly during the export. Some of these optimizations are specific to
            ONNX Runtime, and the resulting ONNX will not be usable with other runtime as OpenVINO or TensorRT.
            Available options: `"O1", "O2", "O3", "O4"`. Reference: [`~optimum.onnxruntime.AutoOptimizationConfig`]
        monolith (`bool`, defaults to `False`):
            Forces to export the model as a single ONNX file.
        no_post_process (`bool`, defaults to `False`):
            Allows to disable any post-processing done by default on the exported ONNX models.
        framework (`Optional[str]`, defaults to `None`):
            The framework to use for the ONNX export (`"pt"` or `"tf"`). If not provided, will attempt to automatically detect
            the framework for the checkpoint.
        atol (`Optional[float]`, defaults to `None`):
            If specified, the absolute difference tolerance when validating the model. Otherwise, the default atol for the model will be used.
        cache_dir (`Optional[str]`, defaults to `None`):
            Path indicating where to store cache. The default Hugging Face cache path will be used by default.
        trust_remote_code (`bool`, defaults to `False`):
            Allows to use custom code for the modeling hosted in the model repository. This option should only be set for repositories
            you trust and in which you have read the code, as it will execute on your local machine arbitrary code present in the
            model repository.
        pad_token_id (`Optional[int]`, defaults to `None`):
            This is needed by some models, for some tasks. If not provided, will attempt to use the tokenizer to guess it.
        subfolder (`str`, defaults to `""`):
            In case the relevant files are located inside a subfolder of the model repo either locally or on huggingface.co, you can
            specify the folder name here.
        revision (`str`, defaults to `"main"`):
            Revision is the specific model version to use. It can be a branch name, a tag name, or a commit id.
        force_download (`bool`, defaults to `False`):
            Whether or not to force the (re-)download of the model weights and configuration files, overriding the
            cached versions if they exist.
        local_files_only (`Optional[bool]`, defaults to `False`):
            Whether or not to only look at local files (i.e., do not try to download the model).
        use_auth_token (`Optional[str]`, defaults to `None`):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `transformers-cli login` (stored in `~/.huggingface`).
        model_kwargs (`Optional[Dict[str, Any]]`, defaults to `None`):
            Experimental usage: keyword arguments to pass to the model during
            the export. This argument should be used along the `custom_onnx_configs` argument
            in case, for example, the model inputs/outputs are changed (for example, if
            `model_kwargs={"output_attentions": True}` is passed).
        custom_onnx_configs (`Optional[Dict[str, OnnxConfig]]`, defaults to `None`):
            Experimental usage: override the default ONNX config used for the given model. This argument may be useful for advanced users that desire a finer-grained control on the export. An example is available [here](https://huggingface.co/docs/optimum/main/en/exporters/onnx/usage_guides/export_a_model).
        fn_get_submodels (`Optional[Callable]`, defaults to `None`):
            Experimental usage: Override the default submodels that are used at the export. This is
            especially useful when exporting a custom architecture that needs to split the ONNX (e.g. encoder-decoder). If unspecified with custom models, optimum will try to use the default submodels used for the given task, with no guarantee of success.
        use_subprocess (`bool`, defaults to `False`):
            Do the ONNX exported model validation in subprocesses. This is especially useful when
            exporting on CUDA device, where ORT does not release memory at inference session
            destruction. When set to `True`, the `main_export` call should be guarded in
            `if __name__ == "__main__":` block.
        _variant (`str`, defaults to `default`):
            Specify the variant of the ONNX export to use.
        library_name (`Optional[str]`, defaults to `None`):
            The library of the model (`"transformers"` or `"diffusers"` or `"timm"` or `"sentence_transformers"`). If not provided, will attempt to automatically detect the library name for the checkpoint.
        legacy (`bool`, defaults to `False`):
            Disable the use of position_ids for text-generation models that require it for batched generation. Also enable to export decoder only models in three files (without + with past and the merged model). This argument is introduced for backward compatibility and will be removed in a future release of Optimum.
        **kwargs_shapes (`Dict`):
            Shapes to use during inference. This argument allows to override the default shapes used during the ONNX export.

    Example usage:
    ```python
    >>> from optimum.exporters.onnx import main_export

    >>> main_export("gpt2", output="gpt2_onnx/")
    ```
    """
    if optimize == "O4" and device != "cuda":
        raise ValueError(
            "Requested O4 optimization, but this optimization requires to do the export on GPU."
            " Please pass the argument `--device cuda`."
        )

    if (framework == "tf" and fp16 is True) or not is_torch_available():
        raise ValueError("The --fp16 option is supported only for PyTorch.")

    if fp16:
        if device == "cpu":
            raise ValueError(
                "FP16 export is supported only when exporting on GPU. Please pass the option `--device cuda`."
            )
        float_dtype = "fp16"
    else:
        float_dtype = "fp32"

    output = Path(output)
    if not output.exists():
        output.mkdir(parents=True)

    if for_ort:
        logger.warning(
            "The option --for-ort was passed, but its behavior is now the default in the ONNX exporter"
            " and passing it is not required anymore."
        )

    original_task = task
    task = TasksManager.map_from_synonym(task)

    framework = TasksManager.determine_framework(model_name_or_path, subfolder=subfolder, framework=framework)
    library_name = TasksManager.infer_library_from_model(
        model_name_or_path, subfolder=subfolder, library_name=library_name
    )

    torch_dtype = None if fp16 is False else torch.float16

    if task == "auto":
        try:
            task = TasksManager.infer_task_from_model(model_name_or_path)
        except KeyError as e:
            raise KeyError(
                f"The task could not be automatically inferred. Please provide the argument --task with the relevant task from {', '.join(TasksManager.get_all_tasks())}. Detailed error: {e}"
            )
        except RequestsConnectionError as e:
            raise RequestsConnectionError(
                f"The task could not be automatically inferred as this is available only for models hosted on the Hugging Face Hub. Please provide the argument --task with the relevant task from {', '.join(TasksManager.get_all_tasks())}. Detailed error: {e}"
            )

    custom_architecture = False
    loading_kwargs = {}

    if library_name == "transformers":
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            subfolder=subfolder,
            revision=revision,
            cache_dir=cache_dir,
            use_auth_token=use_auth_token,
            local_files_only=local_files_only,
            force_download=force_download,
            trust_remote_code=trust_remote_code,
        )
        model_type = config.model_type.replace("_", "-")
        if model_type not in TasksManager._SUPPORTED_MODEL_TYPE:
            custom_architecture = True
        elif task not in TasksManager.get_supported_tasks_for_model_type(model_type, "onnx"):
            if original_task == "auto":
                autodetected_message = " (auto-detected)"
            else:
                autodetected_message = ""
            model_tasks = TasksManager.get_supported_tasks_for_model_type(model_type, exporter="onnx")
            raise ValueError(
                f"Asked to export a {model_type} model for the task {task}{autodetected_message}, but the Optimum ONNX exporter only supports the tasks {', '.join(model_tasks.keys())} for {model_type}. Please use a supported task. Please open an issue at https://github.com/huggingface/optimum/issues if you would like the task {task} to be supported in the ONNX export for {model_type}."
            )

        # TODO: Fix in Transformers so that SdpaAttention class can be exported to ONNX. `attn_implementation` is introduced in Transformers 4.36.
        if model_type in SDPA_ARCHS_ONNX_EXPORT_NOT_SUPPORTED and _transformers_version >= version.parse("4.35.99"):
            loading_kwargs["attn_implementation"] = "eager"

    model = TasksManager.get_model_from_task(
        task,
        model_name_or_path,
        subfolder=subfolder,
        revision=revision,
        cache_dir=cache_dir,
        use_auth_token=use_auth_token,
        local_files_only=local_files_only,
        force_download=force_download,
        trust_remote_code=trust_remote_code,
        framework=framework,
        torch_dtype=torch_dtype,
        device=device,
        library_name=library_name,
        **loading_kwargs,
    )

    is_stable_diffusion = "stable-diffusion" in task
    model_type = "stable-diffusion" if is_stable_diffusion else model.config.model_type.replace("_", "-")

    # For MODEL_TO_PATCH_FOR_PAST architectures, when exporting the model with an input of sequence length of 1, a tracer that does not handle
    # controlflows will trace incorrectly the mask generation, resulting in incorrect attention masks for other sequence lengthss.
    # Reference: https://github.com/huggingface/transformers/blob/af3de8d87c717c4bb090f037d0d89413c195a42f/src/transformers/modeling_attn_mask_utils.py#L94
    input_shapes = {}
    for input_name in DEFAULT_DUMMY_SHAPES.keys():
        input_shapes[input_name] = (
            kwargs_shapes[input_name] if input_name in kwargs_shapes else DEFAULT_DUMMY_SHAPES[input_name]
        )

        # TODO: this may be moved rather to the OnnxConfig to avoid bloating this script.
        if (
            model_type in MODEL_TO_PATCH_FOR_PAST
            and input_name == "sequence_length"
            and kwargs_shapes.get(input_name) == 1
        ):
            raise ValueError(
                f"Exporting with a sequence length of 1 a {model_type} model is not supported and can yield unexpected results."
            )

    if legacy and model_type in MODEL_TYPES_REQUIRING_POSITION_IDS and task.startswith("text-generation"):
        logger.warning(
            f"legacy=True was specified in the ONNX export, although the model {model_name_or_path} (model type {model_type}) requires position_ids for batched inference. Passing `legacy=True` is strongly discouraged, and this option will be removed in a future release. Reference: https://github.com/huggingface/optimum/pull/1381"
        )

    if not is_stable_diffusion:
        if model_type in TasksManager._UNSUPPORTED_CLI_MODEL_TYPE:
            raise ValueError(
                f"{model_type} is not supported yet. Only {list(TasksManager._SUPPORTED_CLI_MODEL_TYPE.keys())} are supported. "
                f"If you want to support {model_type} please propose a PR or open up an issue."
            )

    # TODO: support onnx_config.py in the model repo
    if custom_architecture and custom_onnx_configs is None:
        raise ValueError(
            f"Trying to export a {model.config.model_type} model, that is a custom or unsupported architecture for the task {task}, but no custom onnx configuration was passed as `custom_onnx_configs`. Please refer to https://huggingface.co/docs/optimum/main/en/exporters/onnx/usage_guides/export_a_model#custom-export-of-transformers-models for an example on how to export custom models. Please open an issue at https://github.com/huggingface/optimum/issues if you would like the model type {model.config.model_type} to be supported natively in the ONNX export."
        )

    if custom_architecture and original_task == "auto":
        raise ValueError(
            f'Automatic task detection is not supported with custom architectures. Please specify the `task` argument. Suggestion: task="{task}" (or task="{task}-with-past" if the model is decoder-based and supports KV cache)'
        )

    if (
        not custom_architecture
        and not is_stable_diffusion
        and task + "-with-past"
        in TasksManager.get_supported_tasks_for_model_type(model_type, "onnx", library_name=library_name)
    ):
        if original_task == "auto":  # Make -with-past the default if --task was not explicitely specified
            task = task + "-with-past"
        else:
            logger.info(
                f"The task `{task}` was manually specified, and past key values will not be reused in the decoding."
                f" if needed, please pass `--task {task}-with-past` to export using the past key values."
            )

    if task.endswith("-with-past") and monolith is True:
        task_non_past = task.replace("-with-past", "")
        raise ValueError(
            f"The task {task} is not compatible with the --monolith argument. Please either use"
            f" `--task {task_non_past} --monolith`, or `--task {task}` without the monolith argument."
        )

    if original_task == "auto":
        synonyms_for_task = sorted(TasksManager.synonyms_for_task(task))
        if synonyms_for_task:
            synonyms_for_task = ", ".join(synonyms_for_task)
            possible_synonyms = f" (possible synonyms are: {synonyms_for_task})"
        else:
            possible_synonyms = ""
        logger.info(f"Automatic task detection to {task}{possible_synonyms}.")

    # The preprocessors are loaded as they may be useful to export the model. Notably, some of the static input shapes may be stored in the
    # preprocessors config.
    preprocessors = maybe_load_preprocessors(
        model_name_or_path, subfolder=subfolder, trust_remote_code=trust_remote_code
    )
    onnx_config, models_and_onnx_configs = _get_submodels_and_onnx_configs(
        model=model,
        task=task,
        monolith=monolith,
        custom_onnx_configs=custom_onnx_configs if custom_onnx_configs is not None else {},
        custom_architecture=custom_architecture,
        float_dtype=float_dtype,
        fn_get_submodels=fn_get_submodels,
        preprocessors=preprocessors,
        _variant=_variant,
        legacy=legacy,
        library_name=library_name,
        model_kwargs=model_kwargs,
    )

    if not is_stable_diffusion:
        needs_pad_token_id = (
            isinstance(onnx_config, OnnxConfigWithPast)
            and getattr(model.config, "pad_token_id", None) is None
            and task in ["text-classification"]
        )
        if needs_pad_token_id:
            if pad_token_id is not None:
                model.config.pad_token_id = pad_token_id
            else:
                try:
                    tok = AutoTokenizer.from_pretrained(model_name_or_path)
                    model.config.pad_token_id = tok.pad_token_id
                except Exception:
                    raise ValueError(
                        "Could not infer the pad token id, which is needed in this case, please provide it with the --pad_token_id argument"
                    )

        # Ensure the requested opset is sufficient
        if opset is None:
            opset = onnx_config.DEFAULT_ONNX_OPSET

        if opset < onnx_config.DEFAULT_ONNX_OPSET:
            raise ValueError(
                f"Opset {opset} is not sufficient to export {model_type}. "
                f"At least {onnx_config.DEFAULT_ONNX_OPSET} is required."
            )
        if atol is None:
            atol = onnx_config.ATOL_FOR_VALIDATION
            if isinstance(atol, dict):
                atol = atol[task.replace("-with-past", "")]

        # Saving the model config and preprocessor as this is needed sometimes.
        model.config.save_pretrained(output)
        generation_config = getattr(model, "generation_config", None)
        if generation_config is not None:
            generation_config.save_pretrained(output)
        maybe_save_preprocessors(model_name_or_path, output)

        if model.config.is_encoder_decoder and task.startswith("text-generation"):
            raise ValueError(
                f"model.config.is_encoder_decoder is True and task is `{task}`, which are incompatible. If the task was auto-inferred, please fill a bug report"
                f"at https://github.com/huggingface/optimum, if --task was explicitely passed, make sure you selected the right task for the model,"
                f" referring to `optimum.exporters.tasks.TaskManager`'s `_TRANSFORMERS_TASKS_TO_MODEL_LOADERS`."
            )

        onnx_files_subpaths = [key + ".onnx" for key in models_and_onnx_configs.keys()]
    else:
        # save the subcomponent configuration
        for model_name in models_and_onnx_configs:
            subcomponent = models_and_onnx_configs[model_name][0]
            if hasattr(subcomponent, "save_config"):
                subcomponent.save_config(output / model_name)
            elif hasattr(subcomponent, "config") and hasattr(subcomponent.config, "save_pretrained"):
                subcomponent.config.save_pretrained(output / model_name)

        onnx_files_subpaths = [os.path.join(name_dir, ONNX_WEIGHTS_NAME) for name_dir in models_and_onnx_configs]

        # Saving the additional components needed to perform inference.
        model.scheduler.save_pretrained(output.joinpath("scheduler"))

        feature_extractor = getattr(model, "feature_extractor", None)
        if feature_extractor is not None:
            feature_extractor.save_pretrained(output.joinpath("feature_extractor"))

        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.save_pretrained(output.joinpath("tokenizer"))

        tokenizer_2 = getattr(model, "tokenizer_2", None)
        if tokenizer_2 is not None:
            tokenizer_2.save_pretrained(output.joinpath("tokenizer_2"))

        model.save_config(output)

    _, onnx_outputs = export_models(
        models_and_onnx_configs=models_and_onnx_configs,
        opset=opset,
        output_dir=output,
        output_names=onnx_files_subpaths,
        input_shapes=input_shapes,
        device=device,
        dtype="fp16" if fp16 is True else None,
        model_kwargs=model_kwargs,
    )

    if optimize is not None:
        from ...onnxruntime import AutoOptimizationConfig, ORTOptimizer

        optimizer = ORTOptimizer.from_pretrained(output, file_names=onnx_files_subpaths)

        optimization_config = AutoOptimizationConfig.with_optimization_level(optimization_level=optimize)

        optimization_config.disable_shape_inference = True
        optimizer.optimize(save_dir=output, optimization_config=optimization_config, file_suffix="")

    # Optionally post process the obtained ONNX file(s), for example to merge the decoder / decoder with past if any
    # TODO: treating stable diffusion separately is quite ugly
    if not no_post_process and not is_stable_diffusion:
        try:
            logger.info("Post-processing the exported models...")
            models_and_onnx_configs, onnx_files_subpaths = onnx_config.post_process_exported_models(
                output, models_and_onnx_configs, onnx_files_subpaths
            )
        except Exception as e:
            raise Exception(
                f"The post-processing of the ONNX export failed. The export can still be performed by passing the option --no-post-process. Detailed error: {e}"
            )

    if is_stable_diffusion:
        use_subprocess = (
            False  # TODO: fix Can't pickle local object 'get_stable_diffusion_models_for_export.<locals>.<lambda>'
        )
    elif model.config.model_type in UNPICKABLE_ARCHS:
        # Pickling is bugged for nn.utils.weight_norm: https://github.com/pytorch/pytorch/issues/102983
        # TODO: fix "Cowardly refusing to serialize non-leaf tensor" error for wav2vec2-conformer
        use_subprocess = False

    if device == "cpu":
        # Using multiprocessing for validation is useful only on CUDA EP that leaks memory.
        use_subprocess = False

    if do_validation is True:
        try:
            validate_models_outputs(
                models_and_onnx_configs=models_and_onnx_configs,
                onnx_named_outputs=onnx_outputs,
                atol=atol,
                output_dir=output,
                onnx_files_subpaths=onnx_files_subpaths,
                input_shapes=input_shapes,
                device=device,
                dtype=torch_dtype,
                use_subprocess=use_subprocess,
                model_kwargs=model_kwargs,
            )
            logger.info(f"The ONNX export succeeded and the exported model was saved at: {output.as_posix()}")
        except ShapeError as e:
            raise e
        except AtolError as e:
            logger.warning(
                f"The ONNX export succeeded with the warning: {e}.\n The exported model was saved at: {output.as_posix()}"
            )
        except OutputMatchError as e:
            logger.warning(
                f"The ONNX export succeeded with the warning: {e}.\n The exported model was saved at: {output.as_posix()}"
            )
        except Exception as e:
            raise Exception(
                f"An error occured during validation, but the model was saved nonetheless at {output.as_posix()}. Detailed error: {e}."
            )


def main():
    parser = argparse.ArgumentParser("Hugging Face Optimum ONNX exporter")

    parse_args_onnx(parser)

    # Retrieve CLI arguments
    args = parser.parse_args()

    # get the shapes to be used to generate dummy inputs
    input_shapes = {}
    for input_name in DEFAULT_DUMMY_SHAPES.keys():
        input_shapes[input_name] = getattr(args, input_name)

    main_export(
        model_name_or_path=args.model,
        output=args.output,
        task=args.task,
        opset=args.opset,
        device=args.device,
        fp16=args.fp16,
        optimize=args.optimize,
        monolith=args.monolith,
        no_post_process=args.no_post_process,
        framework=args.framework,
        atol=args.atol,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
        pad_token_id=args.pad_token_id,
        for_ort=args.for_ort,
        library_name=args.library_name,
        legacy=args.legacy,
        **input_shapes,
    )


if __name__ == "__main__":
    main()
