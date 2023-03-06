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

from transformers import AutoTokenizer
from transformers.utils import is_torch_available

from ...commands.export.onnx import parse_args_onnx
from ...onnxruntime import AutoOptimizationConfig, ORTOptimizer
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


logger = logging.get_logger()
logger.setLevel(logging.INFO)


def main():
    parser = ArgumentParser("Hugging Face Optimum ONNX exporter")

    parse_args_onnx(parser)

    # Retrieve CLI arguments
    args = parser.parse_args()

    if not args.output.exists():
        args.output.mkdir(parents=True)

    if args.for_ort:
        logger.warning(
            "The option --for-ort was passed, but its behavior is now the default in the ONNX exporter"
            " and passing it is not required anymore."
        )

    # Infer the task
    task = args.task
    if task == "auto":
        try:
            task = TasksManager.infer_task_from_model(args.model)
        except KeyError as e:
            raise KeyError(
                f"The task could not be automatically inferred. Please provide the argument --task with the task from {', '.join(TasksManager.get_all_tasks())}. Detailed error: {e}"
            )

    if (args.framework == "tf" and args.fp16 is True) or not is_torch_available():
        raise ValueError("The --fp16 option is supported only for PyTorch.")

    if args.fp16 is True and args.device == "cpu":
        raise ValueError(
            "The --fp16 option is supported only when exporting on GPU. Please pass the option `--device cuda`."
        )

    # get the shapes to be used to generate dummy inputs
    input_shapes = {}
    for input_name in DEFAULT_DUMMY_SHAPES.keys():
        input_shapes[input_name] = getattr(args, input_name)

    torch_dtype = None if args.fp16 is False else torch.float16
    model = TasksManager.get_model_from_task(
        task,
        args.model,
        framework=args.framework,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch_dtype,
    )

    if task.endswith("-with-past") and args.monolith is True:
        task_non_past = task.replace("-with-past", "")
        raise ValueError(
            f"The task {task} is not compatible with the --monolith argument. Please either use"
            f" `--task {task_non_past} --monolith`, or `--task {task}` without the monolith argument."
        )

    if task != "stable-diffusion" and task + "-with-past" in TasksManager.get_supported_tasks_for_model_type(
        model.config.model_type.replace("_", "-"), "onnx"
    ):
        if args.task == "auto":  # Make -with-past the default if --task was not explicitely specified
            task = task + "-with-past"
        else:
            logger.info(
                f"The task `{task}` was manually specified, and past key values will not be reused in the decoding."
                f" if needed, please pass `--task {task}-with-past` to export using the past key values."
            )

    if args.task == "auto":
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
        if args.atol is None:
            args.atol = onnx_config.ATOL_FOR_VALIDATION
            if isinstance(args.atol, dict):
                args.atol = args.atol[task.replace("-with-past", "")]

        # Saving the model config and preprocessor as this is needed sometimes.
        model.config.save_pretrained(args.output)
        maybe_save_preprocessors(args.model, args.output)

    if task == "stable-diffusion":
        onnx_files_subpaths = [
            "text_encoder/model.onnx",
            "unet/model.onnx",
            "vae_encoder/model.onnx",
            "vae_decoder/model.onnx",
        ]
        models_and_onnx_configs = get_stable_diffusion_models_for_export(model)
        # Saving the additional components needed to perform inference.
        model.tokenizer.save_pretrained(args.output.joinpath("tokenizer"))
        model.scheduler.save_pretrained(args.output.joinpath("scheduler"))
        model.feature_extractor.save_pretrained(args.output.joinpath("feature_extractor"))
        model.save_config(args.output)
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
            and not args.monolith
        ):
            models_and_onnx_configs = get_encoder_decoder_models_for_export(model, onnx_config)
        elif task.startswith("causal-lm") and not args.monolith:
            models_and_onnx_configs = get_decoder_models_for_export(model, onnx_config)
        else:
            models_and_onnx_configs = {"model": (model, onnx_config)}

    _, onnx_outputs = export_models(
        models_and_onnx_configs=models_and_onnx_configs,
        opset=args.opset,
        output_dir=args.output,
        output_names=onnx_files_subpaths,
        input_shapes=input_shapes,
        device=args.device,
        dtype="fp16" if args.fp16 is True else None,
    )

    if args.optimize == "O4" and args.device != "cuda":
        raise ValueError(
            "Requested O4 optimization, but this optimization requires to do the export on GPU."
            " Please pass the argument `--device cuda`."
        )

    if args.optimize is not None:
        if onnx_files_subpaths is None:
            onnx_files_subpaths = [key + ".onnx" for key in models_and_onnx_configs.keys()]
        optimizer = ORTOptimizer.from_pretrained(args.output, file_names=onnx_files_subpaths)

        optimization_config = AutoOptimizationConfig.with_optimization_level(optimization_level=args.optimize)

        optimization_config.disable_shape_inference = True
        optimizer.optimize(save_dir=args.output, optimization_config=optimization_config, file_suffix="")

    # Optionally post process the obtained ONNX file(s), for example to merge the decoder / decoder with past if any
    # TODO: treating stable diffusion separately is quite ugly
    if not args.no_post_process and task != "stable-diffusion":
        try:
            models_and_onnx_configs, onnx_files_subpaths = onnx_config.post_process_exported_models(
                args.output, models_and_onnx_configs, onnx_files_subpaths
            )
        except Exception as e:
            raise Exception(
                f"The post-processing of the ONNX export failed. The export can still be performed by passing the option --no-post-process. Detailed error: {e}"
            )

    try:
        validate_models_outputs(
            models_and_onnx_configs=models_and_onnx_configs,
            onnx_named_outputs=onnx_outputs,
            atol=args.atol,
            output_dir=args.output,
            onnx_files_subpaths=onnx_files_subpaths,
            input_shapes=input_shapes,
            device=args.device,
            dtype=torch_dtype,
        )
        logger.info(f"The ONNX export succeeded and the exported model was saved at: {args.output.as_posix()}")
    except ShapeError as e:
        raise e
    except AtolError as e:
        logger.warning(
            f"The ONNX export succeeded with the warning: {e}.\n The exported model was saved at: {args.output.as_posix()}"
        )
    except OutputMatchError as e:
        logger.warning(
            f"The ONNX export succeeded with the warning: {e}.\n The exported model was saved at: {args.output.as_posix()}"
        )
    except Exception as e:
        raise Exception(
            f"An error occured during validation, but the model was saved nonetheless at {args.output.as_posix()}. Detailed error: {e}."
        )


if __name__ == "__main__":
    main()
