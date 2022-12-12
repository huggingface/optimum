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

from ...utils import DEFAULT_DUMMY_SHAPES, logging
from ...utils.save_utils import maybe_save_preprocessors
from ..tasks import TasksManager
from .base import OnnxConfigWithPast
from .convert import export, export_models, validate_model_outputs, validate_models_outputs
from .utils import get_decoder_models_for_export, get_encoder_decoder_models_for_export


logger = logging.get_logger()
logger.setLevel(logging.INFO)


def main():
    parser = argparse.ArgumentParser(
        "Hugging Face Optimum ONNX exporter", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

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
        "--for-ort",
        action="store_true",
        help=(
            "This exports models ready to be run with Optimum's ORTModel. Useful for encoder-decoder models for"
            "conditional generation. If enabled the encoder and decoder of the model are exported separately."
        ),
    )
    optional_group.add_argument(
        "--opset",
        type=int,
        default=None,
        help="If specified, ONNX opset version to export the model with. Otherwise, the default opset will be used.",
    )
    optional_group.add_argument(
        "--atol",
        type=float,
        default=None,
        help="If specified, the absolute difference tolerance when validating the model. Otherwise, the default atol for the model will be used.",
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
        "--pad_token_id",
        type=int,
        default=None,
        help=(
            "This is needed by some models, for some tasks. If not provided, will attempt to use the tokenizer to guess"
            " it."
        ),
    )
    optional_group.add_argument("--cache_dir", type=str, default=None, help="Path indicating where to store cache.")

    input_group = parser.add_argument_group(
        "Input shapes (if necessary, this allows to override the shapes of the input given to the ONNX exporter, that requires an example input.)"
    )
    doc_input = " to use in the example input given to the ONNX export."
    input_group.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_DUMMY_SHAPES["batch_size"],
        help="Text tasks only. Batch size" + doc_input,
    )
    input_group.add_argument(
        "--sequence_length",
        type=int,
        default=DEFAULT_DUMMY_SHAPES["sequence_length"],
        help="Text tasks only. Sequence length " + doc_input,
    )
    input_group.add_argument(
        "--num_choices",
        type=int,
        default=DEFAULT_DUMMY_SHAPES["num_choices"],
        help="Text tasks only. Num choices " + doc_input,
    )
    input_group.add_argument(
        "--width", type=int, default=DEFAULT_DUMMY_SHAPES["width"], help="Image tasks only. Width " + doc_input
    )
    input_group.add_argument(
        "--height", type=int, default=DEFAULT_DUMMY_SHAPES["height"], help="Image tasks only. Height " + doc_input
    )
    input_group.add_argument(
        "--num_channels",
        type=int,
        default=DEFAULT_DUMMY_SHAPES["num_channels"],
        help="Image tasks only. Number of channels " + doc_input,
    )
    input_group.add_argument(
        "--feature_size",
        type=int,
        default=DEFAULT_DUMMY_SHAPES["feature_size"],
        help="Audio tasks only. Feature size " + doc_input,
    )
    input_group.add_argument(
        "--nb_max_frames",
        type=int,
        default=DEFAULT_DUMMY_SHAPES["nb_max_frames"],
        help="Audio tasks only. Maximum number of frames " + doc_input,
    )
    input_group.add_argument(
        "--audio_sequence_length",
        type=int,
        default=DEFAULT_DUMMY_SHAPES["audio_sequence_length"],
        help="Audio tasks only. Audio sequence length " + doc_input,
    )

    # Retrieve CLI arguments
    args = parser.parse_args()
    args.output = args.output.joinpath("model.onnx")

    if not args.output.parent.exists():
        args.output.parent.mkdir(parents=True)

    # Infer the task
    task = args.task
    if task == "auto":
        task = TasksManager.infer_task_from_model(args.model)

    # get input shapes
    input_shapes = {}
    for input_name in DEFAULT_DUMMY_SHAPES.keys():
        input_shapes[input_name] = getattr(args, input_name)

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
            input_shapes=input_shapes,
        )
    else:
        onnx_inputs, onnx_outputs = export(model, onnx_config, args.opset, args.output, input_shapes=input_shapes)

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
