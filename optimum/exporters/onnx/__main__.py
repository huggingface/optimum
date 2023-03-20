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
from pathlib import Path

from transformers import AutoTokenizer
from transformers.utils import is_torch_available

from ...utils import DEFAULT_DUMMY_SHAPES, logging
from ...utils.save_utils import maybe_save_preprocessors
from ..error_utils import AtolError, OutputMatchError, ShapeError
from ..tasks import TasksManager
from .base import OnnxConfigWithPast
from .convert import export_models, validate_models_outputs
from .utils import (
    get_decoder_models_for_export,
    get_encoder_decoder_models_for_export,
    get_stable_diffusion_models_for_export,
)


if is_torch_available():
    import torch

from typing import Optional, Union


logger = logging.get_logger()
logger.setLevel(logging.INFO)


def parse_args_onnx(parser):
    required_group = parser.add_argument_group("Required arguments")
    required_group.add_argument(
        "-m", "--model", type=str, required=True, help="Model ID on huggingface.co or path on disk to load model from."
    )
    required_group.add_argument(
        "output", type=Path, help="Path indicating the directory where to store generated ONNX model."
    )

    optional_group = parser.add_argument_group("Optional arguments")
    optional_group.add_argument(
        "--task",
        default="auto",
        help=(
            "The task to export the model for. If not specified, the task will be auto-inferred based on the model. Available tasks depend on the model, but are among:"
            f" {str(list(TasksManager._TASKS_TO_AUTOMODELS.keys()))}. For decoder models, use `xxx-with-past` to export the model using past key values in the decoder."
        ),
    )
    optional_group.add_argument(
        "--opset",
        type=int,
        default=None,
        help="If specified, ONNX opset version to export the model with. Otherwise, the default opset for the given model architecture will be used.",
    )
    optional_group.add_argument(
        "--device",
        type=str,
        default="cpu",
        help='The device to use to do the export. Defaults to "cpu".',
    )
    optional_group.add_argument(
        "--fp16",
        action="store_true",
        help="Use half precision during the export. PyTorch-only, requires `--device cuda`.",
    )
    optional_group.add_argument(
        "--optimize",
        type=str,
        default=None,
        choices=["O1", "O2", "O3", "O4"],
        help=(
            "Allows to run ONNX Runtime optimizations directly during the export. Some of these optimizations are specific to ONNX Runtime, and the resulting ONNX will not be usable with other runtime as OpenVINO or TensorRT. Possible options:\n"
            "    - O1: Basic general optimizations\n"
            "    - O2: Basic and extended general optimizations, transformers-specific fusions\n"
            "    - O3: Same as O2 with GELU approximation\n"
            "    - O4: Same as O3 with mixed precision (fp16, GPU-only, requires `--device cuda`)"
        ),
    )
    optional_group.add_argument(
        "--monolith",
        action="store_true",
        help=(
            "Force to export the model as a single ONNX file. By default, the ONNX exporter may break the model in several"
            " ONNX files, for example for encoder-decoder models where the encoder should be run only once while the"
            " decoder is looped over."
        ),
    )
    optional_group.add_argument(
        "--no-post-process",
        action="store_true",
        help=(
            "Allows to disable any post-processing done by default on the exported ONNX models. For example, the merging of decoder"
            " and decoder-with-past models into a single ONNX model file to reduce memory usage."
        ),
    )
    optional_group.add_argument(
        "--framework",
        type=str,
        choices=["pt", "tf"],
        default=None,
        help=(
            "The framework to use for the ONNX export."
            " If not provided, will attempt to use the local checkpoint's original framework"
            " or what is available in the environment."
        ),
    )
    optional_group.add_argument(
        "--atol",
        type=float,
        default=None,
        help="If specified, the absolute difference tolerance when validating the model. Otherwise, the default atol for the model will be used.",
    )
    optional_group.add_argument("--cache_dir", type=str, default=None, help="Path indicating where to store cache.")
    optional_group.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allows to use custom code for the modeling hosted in the model repository. This option should only be set for repositories you trust and in which you have read the code, as it will execute on your local machine arbitrary code present in the model repository.",
    )
    optional_group.add_argument(
        "--pad_token_id",
        type=int,
        default=None,
        help=(
            "This is needed by some models, for some tasks. If not provided, will attempt to use the tokenizer to guess"
            " it."
        ),
    )

    input_group = parser.add_argument_group(
        "Input shapes (if necessary, this allows to override the shapes of the input given to the ONNX exporter, that requires an example input)."
    )
    doc_input = "to use in the example input given to the ONNX export."
    input_group.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_DUMMY_SHAPES["batch_size"],
        help=f"Text tasks only. Batch size {doc_input}",
    )
    input_group.add_argument(
        "--sequence_length",
        type=int,
        default=DEFAULT_DUMMY_SHAPES["sequence_length"],
        help=f"Text tasks only. Sequence length {doc_input}",
    )
    input_group.add_argument(
        "--num_choices",
        type=int,
        default=DEFAULT_DUMMY_SHAPES["num_choices"],
        help=f"Text tasks only. Num choices {doc_input}",
    )
    input_group.add_argument(
        "--width",
        type=int,
        default=DEFAULT_DUMMY_SHAPES["width"],
        help=f"Image tasks only. Width {doc_input}",
    )
    input_group.add_argument(
        "--height",
        type=int,
        default=DEFAULT_DUMMY_SHAPES["height"],
        help=f"Image tasks only. Height {doc_input}",
    )
    input_group.add_argument(
        "--num_channels",
        type=int,
        default=DEFAULT_DUMMY_SHAPES["num_channels"],
        help=f"Image tasks only. Number of channels {doc_input}",
    )
    input_group.add_argument(
        "--feature_size",
        type=int,
        default=DEFAULT_DUMMY_SHAPES["feature_size"],
        help=f"Audio tasks only. Feature size {doc_input}",
    )
    input_group.add_argument(
        "--nb_max_frames",
        type=int,
        default=DEFAULT_DUMMY_SHAPES["nb_max_frames"],
        help=f"Audio tasks only. Maximum number of frames {doc_input}",
    )
    input_group.add_argument(
        "--audio_sequence_length",
        type=int,
        default=DEFAULT_DUMMY_SHAPES["audio_sequence_length"],
        help=f"Audio tasks only. Audio sequence length {doc_input}",
    )

    # deprecated argument
    parser.add_argument("--for-ort", action="store_true", help=argparse.SUPPRESS)


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
    **kwargs_shapes,
):
    """
    Full-suite ONNX export.

    Args:
        > Required parameters

        model_name_or_path (`str`):
            Model ID on huggingface.co or path on disk to the model repository to export.
        output (`Union[str, Path]`):
            Path indicating the directory where to store generated ONNX model.

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
            Force to export the model as a single ONNX file.
        no_post_process (`bool`, defaults to `False`):
            Allows to disable any post-processing done by default on the exported ONNX models.
        framework (`Optional[str]`, defaults to `None`):
            The framework to use for the ONNX export (`"pt"` or `"tf"`). If not provided, will attempt to use to automatically detect
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
        **kwargs_shapes (`Dict`):
            Shapes to use during inference. This argument allows to override the default shapes used during the ONNX export.

    Example usage:
    ```python
    >>> from optimum.exporters.onnx import main_export

    >>> main_export("gpt2", output="gpt2_onnx/")
    ```
    """
    output = Path(output)
    if not output.exists():
        output.mkdir(parents=True)

    if for_ort:
        logger.warning(
            "The option --for-ort was passed, but its behavior is now the default in the ONNX exporter"
            " and passing it is not required anymore."
        )

    original_task = task
    # Infer the task
    if task == "auto":
        try:
            task = TasksManager.infer_task_from_model(model_name_or_path)
        except KeyError as e:
            raise KeyError(
                f"The task could not be automatically inferred. Please provide the argument --task with the task from {', '.join(TasksManager.get_all_tasks())}. Detailed error: {e}"
            )

    framework = TasksManager.determine_framework(model_name_or_path, subfolder=subfolder, framework=framework)

    if (framework == "tf" and fp16 is True) or not is_torch_available():
        raise ValueError("The --fp16 option is supported only for PyTorch.")

    if fp16 is True and device == "cpu":
        raise ValueError(
            "The --fp16 option is supported only when exporting on GPU. Please pass the option `--device cuda`."
        )

    # get the shapes to be used to generate dummy inputs
    input_shapes = {}
    for input_name in DEFAULT_DUMMY_SHAPES.keys():
        input_shapes[input_name] = (
            kwargs_shapes[input_name] if input_name in input_shapes else DEFAULT_DUMMY_SHAPES[input_name]
        )

    torch_dtype = None if fp16 is False else torch.float16
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
    )

    if task != "stable-diffusion" and task + "-with-past" in TasksManager.get_supported_tasks_for_model_type(
        model.config.model_type.replace("_", "-"), "onnx"
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
        logger.info(f"Automatic task detection to {task}.")

    if task != "stable-diffusion":
        onnx_config_constructor = TasksManager.get_exporter_config_constructor(model=model, exporter="onnx", task=task)
        onnx_config = onnx_config_constructor(model.config)

        needs_pad_token_id = (
            isinstance(onnx_config, OnnxConfigWithPast)
            and getattr(model.config, "pad_token_id", None) is None
            and task in ["sequence_classification"]
        )
        if needs_pad_token_id:
            if pad_token_id is not None:
                model.config.pad_token_id = pad_token_id
            else:
                try:
                    tok = AutoTokenizer.from_pretrained(model)
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
                f"Opset {opset} is not sufficient to export {model.config.model_type}. "
                f"At least  {onnx_config.DEFAULT_ONNX_OPSET} is required."
            )
        if atol is None:
            atol = onnx_config.ATOL_FOR_VALIDATION
            if isinstance(atol, dict):
                atol = atol[task.replace("-with-past", "")]

        # Saving the model config and preprocessor as this is needed sometimes.
        model.config.save_pretrained(output)
        maybe_save_preprocessors(model, output)

    if task == "stable-diffusion":
        onnx_files_subpaths = [
            "text_encoder/model.onnx",
            "unet/model.onnx",
            "vae_encoder/model.onnx",
            "vae_decoder/model.onnx",
        ]
        models_and_onnx_configs = get_stable_diffusion_models_for_export(model)
        # Saving the additional components needed to perform inference.
        model.tokenizer.save_pretrained(output.joinpath("tokenizer"))
        model.scheduler.save_pretrained(output.joinpath("scheduler"))
        if model.feature_extractor is not None:
            model.feature_extractor.save_pretrained(output.joinpath("feature_extractor"))
        model.save_config(output)
    else:
        if model.config.is_encoder_decoder and task.startswith("causal-lm"):
            raise ValueError(
                f"model.config.is_encoder_decoder is True and task is `{task}`, which are incompatible. If the task was auto-inferred, please fill a bug report"
                f"at https://github.com/huggingface/optimum, if --task was explicitely passed, make sure you selected the right task for the model,"
                f" referring to `optimum.exporters.tasks.TaskManager`'s `_TASKS_TO_AUTOMODELS`."
            )

        onnx_files_subpaths = None
        if (
            model.config.is_encoder_decoder
            and task.startswith(("seq2seq-lm", "speech2seq-lm", "vision2seq-lm", "default-with-past"))
            and not monolith
        ):
            models_and_onnx_configs = get_encoder_decoder_models_for_export(model, onnx_config)
        elif task.startswith("causal-lm") and not monolith:
            models_and_onnx_configs = get_decoder_models_for_export(model, onnx_config)
        else:
            models_and_onnx_configs = {"model": (model, onnx_config)}

    _, onnx_outputs = export_models(
        models_and_onnx_configs=models_and_onnx_configs,
        opset=opset,
        output_dir=output,
        output_names=onnx_files_subpaths,
        input_shapes=input_shapes,
        device=device,
        dtype="fp16" if fp16 is True else None,
    )

    if optimize == "O4" and device != "cuda":
        raise ValueError(
            "Requested O4 optimization, but this optimization requires to do the export on GPU."
            " Please pass the argument `--device cuda`."
        )

    if optimize is not None:
        from ...onnxruntime import AutoOptimizationConfig, ORTOptimizer

        if onnx_files_subpaths is None:
            onnx_files_subpaths = [key + ".onnx" for key in models_and_onnx_configs.keys()]
        optimizer = ORTOptimizer.from_pretrained(output, file_names=onnx_files_subpaths)

        optimization_config = AutoOptimizationConfig.with_optimization_level(optimization_level=optimize)

        optimization_config.disable_shape_inference = True
        optimizer.optimize(save_dir=output, optimization_config=optimization_config, file_suffix="")

    # Optionally post process the obtained ONNX file(s), for example to merge the decoder / decoder with past if any
    # TODO: treating stable diffusion separately is quite ugly
    if not no_post_process and task != "stable-diffusion":
        try:
            models_and_onnx_configs, onnx_files_subpaths = onnx_config.post_process_exported_models(
                output, models_and_onnx_configs, onnx_files_subpaths
            )
        except Exception as e:
            raise Exception(
                f"The post-processing of the ONNX export failed. The export can still be performed by passing the option --no-post-process. Detailed error: {e}"
            )

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
        **input_shapes,
    )


if __name__ == "__main__":
    main()
