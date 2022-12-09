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

from argparse import ArgumentParser
from pathlib import Path

from transformers import AutoTokenizer

from optimum.utils import is_diffusers_available

from ...utils import logging
from ...utils.save_utils import maybe_save_preprocessors
from ..tasks import TasksManager
from .base import OnnxConfigWithPast
from .convert import (
    export,
    export_models,
    validate_model_outputs,
    validate_models_outputs,
)
from .utils import (
    get_decoder_models_for_export,
    get_encoder_decoder_models_for_export,
    get_stable_diffusion_models_for_export,
)


if is_diffusers_available():
    from diffusers import OnnxRuntimeModel, OnnxStableDiffusionPipeline, StableDiffusionPipeline

logger = logging.get_logger()
logger.setLevel(logging.INFO)


def main():
    parser = ArgumentParser("Hugging Face Optimum ONNX exporter")
    parser.add_argument(
        "-m", "--model", type=str, required=True, help="Model ID on huggingface.co or path on disk to load model from."
    )
    parser.add_argument(
        "--task",
        default="auto",
        help="The type of task to export the model with.",
    )
    parser.add_argument("--opset", type=int, default=None, help="ONNX opset version to export the model with.")
    parser.add_argument(
        "--atol", type=float, default=None, help="Absolute difference tolerance when validating the model."
    )
    parser.add_argument(
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
    parser.add_argument(
        "--pad_token_id",
        type=int,
        default=None,
        help=(
            "This is needed by some models, for some tasks. If not provided, will attempt to use the tokenizer to guess"
            " it."
        ),
    )
    parser.add_argument("--cache_dir", type=str, default=None, help="Path indicating where to store cache.")
    parser.add_argument(
        "--for-ort",
        action="store_true",
        help=(
            "This exports models ready to be run with optimum.onnxruntime. Useful for encoder-decoder models for"
            "conditional generation. If enabled the encoder and decoder of the model are exported separately."
        ),
    )
    parser.add_argument("output", type=Path, help="Path indicating the directory where to store generated ONNX model.")

    # Retrieve CLI arguments
    args = parser.parse_args()
    args.output = args.output.joinpath("model.onnx")

    if not args.output.parent.exists():
        args.output.parent.mkdir(parents=True)

    # Infer the task
    task = args.task
    if task == "auto":
        try:
            task = TasksManager.infer_task_from_model(args.model)
        except KeyError as e:
            raise KeyError(
                f"The task could not be automatically inferred. Please provide the argument --task with the task from {', '.join(TasksManager.get_all_tasks())}. Detailed error: {e}"
            )
    # TODO : infer stable-diffusion when auto
    elif task == "stable-diffusion":
        pipeline = StableDiffusionPipeline.from_pretrained(args.model)
        onnx_inputs, onnx_outputs = export_models(
            model=pipeline,
            onnx_config=None,
            opset=args.opset,
            output_dir=args.output.parent,
            fn_get_models_from_config=get_stable_diffusion_models_for_export,
            output_names=["text_encoder/model.onnx", "unet/model.onnx", "vae_decoder/model.onnx"],
        )
        try:
            validate_models_outputs(
                onnx_config=None,
                reference_model=pipeline,
                onnx_named_outputs=onnx_outputs,
                atol=args.atol or 14,
                output_dir=args.output.parent,
                fn_get_models_from_config=get_stable_diffusion_models_for_export,
                output_names=["text_encoder/model.onnx", "unet/model.onnx", "vae_decoder/model.onnx"],
            )
        except ValueError:
            logger.error(f"An error occured, but the model was saved at: {args.output.parent.as_posix()}")
            return
        logger.info(f"All good, model saved at: {args.output.parent.as_posix()}")
        return

    # Allocate the model
    model = TasksManager.get_model_from_task(task, args.model, framework=args.framework, cache_dir=args.cache_dir)
    model_type = model.config.model_type.replace("_", "-")
    model_name = getattr(model, "name", None)

    onnx_config_constructor = TasksManager.get_exporter_config_constructor(
        model_type, "onnx", task=task, model_name=model_name
    )
    onnx_config = onnx_config_constructor(model.config)

    needs_pad_token_id = (
        isinstance(onnx_config, OnnxConfigWithPast)
        and getattr(model.config, "pad_token_id", None) is None
        and task in ["sequence_classification"]
    )
    if needs_pad_token_id:
        if args.pad_token_id is not None:
            model.config.pad_token_id = args.pad_token_id
        else:
            try:
                tok = AutoTokenizer.from_pretrained(args.model)
                model.config.pad_token_id = tok.pad_token_id
            except Exception:
                raise ValueError(
                    "Could not infer the pad token id, which is needed in this case, please provide it with the --pad_token_id argument"
                )

    # Ensure the requested opset is sufficient
    if args.opset is None:
        args.opset = onnx_config.DEFAULT_ONNX_OPSET

    if args.opset < onnx_config.DEFAULT_ONNX_OPSET:
        raise ValueError(
            f"Opset {args.opset} is not sufficient to export {model.config.model_type}. "
            f"At least  {onnx_config.DEFAULT_ONNX_OPSET} is required."
        )
    if args.for_ort and (model.config.is_encoder_decoder or task.startswith("causal-lm")):
        if model.config.is_encoder_decoder and task.startswith("causal-lm"):
            raise ValueError(
                f"model.config.is_encoder_decoder is True and task is `{task}`, which are incompatible. If the task was auto-inferred, please fill a bug report"
                f"at https://github.com/huggingface/optimum, if --task was explicitely passed, make sure you selected the right task for the model,"
                f" referring to `optimum.exporters.tasks.TaskManager`'s `_TASKS_TO_AUTOMODELS`."
            )
        fn_get_models_from_config = (
            get_encoder_decoder_models_for_export if model.config.is_encoder_decoder else get_decoder_models_for_export
        )
        onnx_inputs, onnx_outputs = export_models(
            model=model,
            onnx_config=onnx_config,
            opset=args.opset,
            output_dir=args.output.parent,
            fn_get_models_from_config=fn_get_models_from_config,
        )
    else:
        onnx_inputs, onnx_outputs = export(model, onnx_config, args.opset, args.output)

    # Saving the model config as this is needed sometimes.
    model.config.save_pretrained(args.output.parent)

    maybe_save_preprocessors(args.model, args.output.parent)

    if args.atol is None:
        args.atol = onnx_config.ATOL_FOR_VALIDATION
        if isinstance(args.atol, dict):
            args.atol = args.atol[task.replace("-with-past", "")]

    try:
        if args.for_ort and (model.config.is_encoder_decoder or task.startswith("causal-lm")):
            fn_get_models_from_config = (
                get_encoder_decoder_models_for_export
                if model.config.is_encoder_decoder
                else get_decoder_models_for_export
            )
            validate_models_outputs(
                onnx_config=onnx_config,
                reference_model=model,
                onnx_named_outputs=onnx_outputs,
                atol=args.atol,
                output_dir=args.output.parent,
                fn_get_models_from_config=fn_get_models_from_config,
            )
        else:
            validate_model_outputs(onnx_config, model, args.output, onnx_outputs, args.atol)
    except ValueError:
        logger.error(f"An error occured, but the model was saved at: {args.output.parent.as_posix()}")
        return
    logger.info(f"All good, model saved at: {args.output.parent.as_posix()}")


if __name__ == "__main__":
    main()
