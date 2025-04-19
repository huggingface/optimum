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
import warnings
from pathlib import Path

from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from requests.exceptions import ConnectionError as RequestsConnectionError
from transformers import AutoConfig, AutoTokenizer
from transformers.utils import is_torch_available

from ...commands.export.onnx import parse_args_onnx
from ...utils import DEFAULT_DUMMY_SHAPES, logging
from ...utils.import_utils import is_transformers_version
from ...utils.save_utils import maybe_load_preprocessors
from ..tasks import TasksManager
from ..utils import DisableCompileContextManager
from .constants import SDPA_ARCHS_ONNX_EXPORT_NOT_SUPPORTED
from .convert import onnx_export_from_model


if is_torch_available():
    import torch

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Union


if TYPE_CHECKING:
    from .base import OnnxConfig

logger = logging.get_logger()


def main_export(
    model_name_or_path: str,
    output: Union[str, Path],
    task: str = "auto",
    opset: Optional[int] = None,
    device: str = "cpu",
    dtype: Optional[str] = None,
    fp16: Optional[bool] = False,
    optimize: Optional[str] = None,
    monolith: bool = False,
    no_post_process: bool = False,
    framework: Optional[str] = None,
    atol: Optional[float] = None,
    cache_dir: str = HUGGINGFACE_HUB_CACHE,
    trust_remote_code: bool = False,
    pad_token_id: Optional[int] = None,
    subfolder: str = "",
    revision: str = "main",
    force_download: bool = False,
    local_files_only: bool = False,
    use_auth_token: Optional[Union[bool, str]] = None,
    token: Optional[Union[bool, str]] = None,
    for_ort: bool = False,
    do_validation: bool = True,
    model_kwargs: Optional[Dict[str, Any]] = None,
    custom_onnx_configs: Optional[Dict[str, "OnnxConfig"]] = None,
    fn_get_submodels: Optional[Callable] = None,
    use_subprocess: bool = False,
    _variant: str = "default",
    library_name: Optional[str] = None,
    legacy: bool = False,
    no_dynamic_axes: bool = False,
    do_constant_folding: bool = True,
    **kwargs_shapes,
):
    """
    Full-suite ONNX export function, exporting **from a model ID on Hugging Face Hub or a local model repository**.

    Args:
        > Required parameters

        model_name_or_path (`str`):
            Model ID on huggingface.co or path on disk to the model repository to export. Example: `model_name_or_path="BAAI/bge-m3"` or `mode_name_or_path="/path/to/model_folder`.
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
        dtype (`Optional[str]`, defaults to `None`):
            The floating point precision to use for the export. Supported options: `"fp32"` (float32), `"fp16"` (float16), `"bf16"` (bfloat16). Defaults to `"fp32"`.
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
        use_auth_token (`Optional[Union[bool,str]]`, defaults to `None`):
            Deprecated. Please use the `token` argument instead.
        token (`Optional[Union[bool,str]]`, defaults to `None`):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `huggingface-cli login` (stored in `huggingface_hub.constants.HF_TOKEN_PATH`).
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
        no_dynamic_axes (bool, defaults to `False`):
            If True, disables the use of dynamic axes during ONNX export.
        do_constant_folding (bool, defaults to `True`):
            PyTorch-specific argument. If `True`, the PyTorch ONNX export will fold constants into adjacent nodes, if possible.
        **kwargs_shapes (`Dict`):
            Shapes to use during inference. This argument allows to override the default shapes used during the ONNX export.

    Example usage:
    ```python
    >>> from optimum.exporters.onnx import main_export

    >>> main_export("gpt2", output="gpt2_onnx/")
    ```
    """

    if use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed soon. Please use the `token` argument instead.",
            FutureWarning,
        )
        if token is not None:
            raise ValueError("You cannot use both `use_auth_token` and `token` arguments at the same time.")
        token = use_auth_token

    if fp16:
        if dtype is not None:
            raise ValueError(
                f'Both the arguments `fp16` ({fp16}) and `dtype` ({dtype}) were specified in the ONNX export, which is not supported. Please specify only `dtype`. Possible options: "fp32" (default), "fp16", "bf16".'
            )

        logger.warning(
            'The argument `fp16` is deprecated in the ONNX export. Please use the argument `dtype="fp16"` instead, or `--dtype fp16` from the command-line.'
        )

        dtype = "fp16"
    elif dtype is None:
        dtype = "fp32"  # Defaults to float32.

    if optimize == "O4" and device != "cuda":
        raise ValueError(
            "Requested O4 optimization, but this optimization requires to do the export on GPU."
            " Please pass the argument `--device cuda`."
        )

    if (framework == "tf" and fp16) or not is_torch_available():
        raise ValueError("The --fp16 option is supported only for PyTorch.")

    if dtype == "fp16" and device == "cpu":
        raise ValueError(
            "FP16 export is supported only when exporting on GPU. Please pass the option `--device cuda`."
        )

    if for_ort:
        logger.warning(
            "The option --for-ort was passed, but its behavior is now the default in the ONNX exporter"
            " and passing it is not required anymore."
        )

    if task in ["stable-diffusion", "stable-diffusion-xl"]:
        logger.warning(
            f"The task `{task}` is deprecated and will be removed in a future release of Optimum. "
            "Please use one of the following tasks instead: `text-to-image`, `image-to-image`, `inpainting`."
        )

    original_task = task
    task = TasksManager.map_from_synonym(task)

    if framework is None:
        framework = TasksManager.determine_framework(
            model_name_or_path, subfolder=subfolder, revision=revision, cache_dir=cache_dir, token=token
        )

    if library_name is None:
        library_name = TasksManager.infer_library_from_model(
            model_name_or_path, subfolder=subfolder, revision=revision, cache_dir=cache_dir, token=token
        )

    torch_dtype = None
    if framework == "pt":
        if dtype == "fp16":
            torch_dtype = torch.float16
        elif dtype == "bf16":
            torch_dtype = torch.bfloat16

    if task.endswith("-with-past") and monolith:
        task_non_past = task.replace("-with-past", "")
        raise ValueError(
            f"The task {task} is not compatible with the --monolith argument. Please either use"
            f" `--task {task_non_past} --monolith`, or `--task {task}` without the monolith argument."
        )

    if task == "auto":
        try:
            task = TasksManager.infer_task_from_model(model_name_or_path, library_name=library_name)
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
            token=token,
            local_files_only=local_files_only,
            force_download=force_download,
            trust_remote_code=trust_remote_code,
        )
        model_type = config.model_type.replace("_", "-")

        if model_type not in TasksManager._SUPPORTED_MODEL_TYPE:
            custom_architecture = True
        elif task not in TasksManager.get_supported_tasks_for_model_type(
            model_type, "onnx", library_name=library_name
        ):
            if original_task == "auto":
                autodetected_message = " (auto-detected)"
            else:
                autodetected_message = ""
            model_tasks = TasksManager.get_supported_tasks_for_model_type(
                model_type, exporter="onnx", library_name=library_name
            )
            raise ValueError(
                f"Asked to export a {model_type} model for the task {task}{autodetected_message}, but the Optimum ONNX exporter only supports the tasks {', '.join(model_tasks.keys())} for {model_type}. Please use a supported task. Please open an issue at https://github.com/huggingface/optimum/issues if you would like the task {task} to be supported in the ONNX export for {model_type}."
            )

        # TODO: Fix in Transformers so that SdpaAttention class can be exported to ONNX. `attn_implementation` is introduced in Transformers 4.36.
        if model_type in SDPA_ARCHS_ONNX_EXPORT_NOT_SUPPORTED and is_transformers_version(">=", "4.35.99"):
            loading_kwargs["attn_implementation"] = "eager"

    with DisableCompileContextManager():
        model = TasksManager.get_model_from_task(
            task,
            model_name_or_path,
            subfolder=subfolder,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            force_download=force_download,
            trust_remote_code=trust_remote_code,
            framework=framework,
            torch_dtype=torch_dtype,
            device=device,
            library_name=library_name,
            **loading_kwargs,
        )

    needs_pad_token_id = task == "text-classification" and getattr(model.config, "pad_token_id", None) is None

    if needs_pad_token_id:
        if pad_token_id is not None:
            model.config.pad_token_id = pad_token_id
        else:
            tok = AutoTokenizer.from_pretrained(model_name_or_path)
            pad_token_id = getattr(tok, "pad_token_id", None)
            if pad_token_id is None:
                raise ValueError(
                    "Could not infer the pad token id, which is needed in this case, please provide it with the --pad_token_id argument"
                )
            model.config.pad_token_id = pad_token_id

    if hasattr(model.config, "export_model_type"):
        model_type = model.config.export_model_type.replace("_", "-")
    else:
        model_type = model.config.model_type.replace("_", "-")

    if (
        not custom_architecture
        and library_name != "diffusers"
        and task + "-with-past"
        in TasksManager.get_supported_tasks_for_model_type(model_type, "onnx", library_name=library_name)
    ):
        # Make -with-past the default if --task was not explicitely specified
        if original_task == "auto" and not monolith:
            task = task + "-with-past"
        else:
            logger.info(
                f"The task `{task}` was manually specified, and past key values will not be reused in the decoding."
                f" if needed, please pass `--task {task}-with-past` to export using the past key values."
            )
            model.config.use_cache = False

    if task.endswith("with-past"):
        model.config.use_cache = True

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

    onnx_export_from_model(
        model=model,
        output=output,
        opset=opset,
        optimize=optimize,
        monolith=monolith,
        no_post_process=no_post_process,
        atol=atol,
        do_validation=do_validation,
        model_kwargs=model_kwargs,
        custom_onnx_configs=custom_onnx_configs,
        fn_get_submodels=fn_get_submodels,
        _variant=_variant,
        legacy=legacy,
        preprocessors=preprocessors,
        device=device,
        no_dynamic_axes=no_dynamic_axes,
        task=task,
        use_subprocess=use_subprocess,
        do_constant_folding=do_constant_folding,
        **kwargs_shapes,
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
        do_constant_folding=not args.no_constant_folding,
        **input_shapes,
    )


if __name__ == "__main__":
    main()
