# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
"""Entry point to the optimum.exporters.tflite command line."""

from argparse import ArgumentParser
from pathlib import Path

from transformers import AutoTokenizer

from ...commands.export.tflite import parse_args_tflite
from ...utils import DEFAULT_DUMMY_SHAPES, logging
from ...utils.save_utils import maybe_save_preprocessors
from ..tasks import TasksManager

# from .base import OnnxConfigWithPast
from .convert import (  # export_models,; validate_models_outputs,
    AtolError,
    OutputMatchError,
    ShapeError,
    export,
    validate_model_outputs,
)


# from .utils import (
#     get_decoder_models_for_export,
#     get_encoder_decoder_models_for_export,
#     get_stable_diffusion_models_for_export,
# )


logger = logging.get_logger()
logger.setLevel(logging.INFO)


def main():
    parser = ArgumentParser("Hugging Face Optimum ONNX exporter")

    parse_args_tflite(parser)

    # Retrieve CLI arguments
    args = parser.parse_args()
    args.output = args.output.joinpath("model.tflite")

    if not args.output.parent.exists():
        args.output.parent.mkdir(parents=True)

    # Infer the task
    task = args.task
    if task == "auto":
        try:
            task = TasksManager.infer_task_from_model(args.model)
        except KeyError as e:
            raise KeyError(
                "The task could not be automatically inferred. Please provide the argument --task with the task "
                f"from {', '.join(TasksManager.get_all_tasks())}. Detailed error: {e}"
            )
    # get the shapes to be used to generate dummy inputs
    # input_shapes = {}
    # for input_name in DEFAULT_DUMMY_SHAPES.keys():
    #     input_shapes[input_name] = getattr(args, input_name)

    model = TasksManager.get_model_from_task(
        task, args.model, framework="tf", cache_dir=args.cache_dir, trust_remote_code=args.trust_remote_code
    )

    if task != "stable-diffusion":
        tflite_config_constructor = TasksManager.get_exporter_config_constructor(
            model=model, exporter="tflite", task=task
        )
        tflite_config = tflite_config_constructor(model.config)
        for name in tflite_config.mandatory_axes:
            setattr(tflite_config, name, getattr(args, name))

        # needs_pad_token_id = (
        #     isinstance(onnx_config, OnnxConfigWithPast)
        #     and getattr(model.config, "pad_token_id", None) is None
        #     and task in ["sequence_classification"]
        # )
        # if needs_pad_token_id:
        #     if args.pad_token_id is not None:
        #         model.config.pad_token_id = args.pad_token_id
        #     else:
        #         try:
        #             tok = AutoTokenizer.from_pretrained(args.model)
        #             model.config.pad_token_id = tok.pad_token_id
        #         except Exception:
        #             raise ValueError(
        #                 "Could not infer the pad token id, which is needed in this case, please provide it with the --pad_token_id argument"
        #             )

        # Ensure the requested opset is sufficient
        if args.atol is None:
            args.atol = tflite_config.ATOL_FOR_VALIDATION
            if isinstance(args.atol, dict):
                args.atol = args.atol[task.replace("-with-past", "")]

        # Saving the model config and preprocessor as this is needed sometimes.
        model.config.save_pretrained(args.output.parent)
        maybe_save_preprocessors(args.model, args.output.parent)

    # if task == "stable-diffusion" or (
    #     args.for_ort and (model.config.is_encoder_decoder or task.startswith("causal-lm"))
    # ):
    #     if task == "stable-diffusion":
    #         output_names = [
    #             "text_encoder/model.onnx",
    #             "unet/model.onnx",
    #             "vae_encoder/model.onnx",
    #             "vae_decoder/model.onnx",
    #         ]
    #         models_and_onnx_configs = get_stable_diffusion_models_for_export(model)
    #         # Saving the additional components needed to perform inference.
    #         model.tokenizer.save_pretrained(args.output.parent.joinpath("tokenizer"))
    #         model.scheduler.save_pretrained(args.output.parent.joinpath("scheduler"))
    #         model.feature_extractor.save_pretrained(args.output.parent.joinpath("feature_extractor"))
    #         model.save_config(args.output.parent)
    #     else:
    #         if model.config.is_encoder_decoder and task.startswith("causal-lm"):
    #             raise ValueError(
    #                 f"model.config.is_encoder_decoder is True and task is `{task}`, which are incompatible. If the task was auto-inferred, please fill a bug report"
    #                 f"at https://github.com/huggingface/optimum, if --task was explicitely passed, make sure you selected the right task for the model,"
    #                 f" referring to `optimum.exporters.tasks.TaskManager`'s `_TASKS_TO_AUTOMODELS`."
    #             )
    #         if model.config.is_encoder_decoder:
    #             models_and_onnx_configs = get_encoder_decoder_models_for_export(model, onnx_config)
    #         else:
    #             models_and_onnx_configs = get_decoder_models_for_export(model, onnx_config)
    #         output_names = None

    #     onnx_inputs, onnx_outputs = export_models(
    #         models_and_onnx_configs=models_and_onnx_configs,
    #         opset=args.opset,
    #         output_dir=args.output.parent,
    #         output_names=output_names,
    #         input_shapes=input_shapes,
    #         device=args.device,
    #     )
    # else:
    tflite_inputs, tflite_outputs = export(
        model=model,
        config=tflite_config,
        output=args.output,
    )

    try:
        # if task == "stable-diffusion" or (
        #     args.for_ort and (model.config.is_encoder_decoder or task.startswith("causal-lm"))
        # ):
        #     validate_models_outputs(
        #         models_and_onnx_configs=models_and_onnx_configs,
        #         onnx_named_outputs=onnx_outputs,
        #         atol=args.atol,
        #         output_dir=args.output.parent,
        #         output_names=output_names,
        #         device=args.device,
        #     )
        # else:
        validate_model_outputs(
            config=tflite_config,
            reference_model=model,
            tflite_model_path=args.output,
            # TODO: add this argument.
            # onnx_named_outputs=onnx_outputs,
            atol=args.atol,
        )

        logger.info(
            f"The TensorFlow Lite export succeeded and the exported model was saved at: {args.output.parent.as_posix()}"
        )
    except ShapeError as e:
        raise e
    except AtolError as e:
        logger.warning(
            f"The TensorFlow Lite export succeeded with the warning: {e}.\n The exported model was saved at: {args.output.parent.as_posix()}"
        )
    except OutputMatchError as e:
        logger.warning(
            f"The TensorFlow Lite export succeeded with the warning: {e}.\n The exported model was saved at: {args.output.parent.as_posix()}"
        )
    except Exception as e:
        logger.error(
            f"An error occured with the error message: {e}.\n The exported model was saved at: {args.output.parent.as_posix()}"
        )


if __name__ == "__main__":
    main()
