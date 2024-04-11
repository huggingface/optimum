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

import logging
import os
import shutil
from pathlib import Path
import subprocess
from typing import Any, Dict, Optional, Union

# import torch
# from transformers import AutoTokenizer, PretrainedConfig
# from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_model_forward, default_cache_path
# from transformers.modeling_outputs import (
#     BaseModelOutput,
#     CausalLMOutputWithCrossAttentions,
#     QuestionAnsweringModelOutput,
#     SequenceClassifierOutput,
#     TokenClassifierOutput,
# )

# from transformers.onnx import FeatureManager, export

import onnxruntime_genai as og
from .builder import create_model
from huggingface_hub import HfApi, hf_hub_download

from ..modeling_base import OptimizedModel
from .utils import ONNX_WEIGHTS_NAME, _is_gpu_available


logger = logging.getLogger(__name__)

# default using huggingface model
# or from the optimized onnxruntime model

class OGModel(OptimizedModel):
    base_model_prefix = "onnx_model"

    def __init__(self, model=None, config=None, *args, **kwargs):
        self.model = model
        self.config = config
        self.model_save_dir = kwargs.get("model_save_dir", None)
        self.latest_model_name = kwargs.get("latest_model_name", "model.onnx")
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def build_model(model_name: str, input_path: str, output_dir: str, precision: str, execution_provider: str, cache_dir: str, extra_options: str) -> str:
        """
        build ONNX model using onnxruntime_genai.models.builder
        Arguments:
            model_name: Model name in Hugging Face.
            execution_provider: Execution provider to target with precision of model (e.g. FP16 CUDA, INT4 CPU)
                Only Support: cpu and cuda for now
            precision: Precision of model, e.g. int4, fp16, fp32
            output: Path to folder to store ONNX model and additional files
        Return:
            The path of the model
        """

        output_dir = os.path.abspath(output_dir)

        # check if the model already existed
        if os.path.exists(output_dir):
            logging.info(f"{model_name} is already existed")
        else:
            # use create model to get the optimized model
            '''model_name, input_path, output_dir, precision, execution_provider, cache_dir'''
            create_model(model_name=model_name,
                         input_path=input_path,
                         output_dir=output_dir,
                         precision=precision,
                         execution_provider=execution_provider,
                         cache_dir=cache_dir,
                         extra_options=extra_options,
                )
            logging.info(f"{model_name} is successfully saved")

        return output_dir

    @staticmethod
    def load_model(path: Union[str, Path], provider=None):
        """
        load model with provider ---> ****d .
        Arguments:
            path: (:obj: `str` or obj: `Path`):
                Directory from which to load from local folder or huggingface hub
            provider (:obj:`str`):
                Onnxruntime provider to use for the model
        """
        path = os.path.abspath(path)

        # load the GGUF or PT model
        try:
            model = og.Model(path)
        except Exception:
            raise ValueError("model failed to be loaded")

        return model

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

class OGModelForTextGeneration(OGModel):
    """
    Text Generation Task
    """
    pipeline_task = "text_generation"

    def __init__(self, model=None, config=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = model
        self.tokenizer = og.Tokenizer(self.model)

        self.search_options = {"max_length": kwargs.get("max_length", 512), 
                                "top_p": kwargs.get("top_p", 0.9),
                                "top_k": kwargs.get("top_k", 50),
                                "temperature": kwargs.get("temperature", 1),
                                "repetition_penalty": kwargs.get("repetition_penalty", 1)
        }
        
    def forward(
        self,
        prompt,
        **kwargs,
    ):

        tokens = self.tokenizer.encode(prompt)
        params = og.GeneratorParams(self.model)

        # TODO: how to update the search options 
        params.set_search_options(self.search_options)

        params.input_ids = tokens

        output_tokens = self.model.generate(params)

        text = self.tokenizer.decode(output_tokens)

        print("Output: ")
        print(text)


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    # Argument for model optimization
    parser.add_argument(
        "-m",
        "--model_name",
        required=False,
        default=None,
        help="Model name in Hugging Face. Do not use if providing an input path to a Hugging Face directory in -i/--input."
    )

    parser.add_argument(
        "-i",
        "--input",
        required=False,
        default="",
        help=textwrap.dedent("""\
            Input model source. Currently supported options are:
                hf_path: Path to folder on disk containing the Hugging Face config, model, tokenizer, etc.
                gguf_path: Path to float16/float32 GGUF file on disk containing the GGUF model
            """),
    )

    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Path to folder to store ONNX model and additional files (e.g. GenAI config, external data files, etc.)",
    )

    parser.add_argument(
        "-p",
        "--precision",
        required=True,
        choices=["int4", "fp16", "fp32"],
        help="Precision of model",
    )

    parser.add_argument(
        "-e",
        "--execution_provider",
        required=True,
        choices=["cpu", "cuda"],
        help="Execution provider to target with precision of model (e.g. FP16 CUDA, INT4 CPU)",
    )

    parser.add_argument(
        "-c",
        "--cache_dir",
        required=False,
        type=str,
        default=os.path.join('.', 'cache_dir'),
        help="Cache directory for Hugging Face files and temporary ONNX external data files",
    )

    parser.add_argument(
        "--extra_options",
        required=False,
        metavar="KEY=VALUE",
        nargs='+',
        help=textwrap.dedent("""\
            Key value pairs for various options. Currently supports:
                int4_block_size = 16/32/64/128/256: Specify the block_size for int4 quantization.
                int4_accuracy_level = 1/2/3/4: Specify the minimum accuracy level for activation of MatMul in int4 quantization.
                    4 is int8, which means input A of int4 quantized MatMul is quantized to int8 and input B is upcasted to int8 for computation.
                    3 is bf16.
                    2 is fp16.
                    1 is fp32.
                num_hidden_layers = Manually specify the number of layers in your ONNX model (for unit testing purposes).
                filename = Filename for ONNX model (default is 'model.onnx').
                    For models with multiple components, each component is exported to its own ONNX model.
                    The filename for each component will be '<filename>_<component-name>.onnx' (ex: '<filename>_encoder.onnx', '<filename>_decoder.onnx').
                config_only = Generate config and pre/post processing files only.
                    Use this option when you already have your optimized and/or quantized ONNX model.
            """),
    )

    # Argument for inference configuration
    parser.add_argument('-l', '--max_length', type=int, default=512, help='Max number of tokens to generate after prompt')
    parser.add_argument('-p', '--top_p', type=float, default=0.9, help='Top p probability to sample with')
    parser.add_argument('-k', '--top_k', type=int, default=50, help='Top k tokens to sample from')
    parser.add_argument('-t', '--temperature', type=float, default=1.0, help='Temperature to sample with')
    parser.add_argument('-r', '--repetition_penalty', type=float, default=1.0, help='Repetition penalty to sample with')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print verbose output')

    args = parser.parse_args()
    print("Valid precision + execution provider combinations are: FP32 CPU, FP32 CUDA, FP16 CUDA, INT4 CPU, INT4 CUDA")
    return args


if __name__ == "__main__":
    args, options = get_args()
    extra_options = parse_extra_options(args.extra_options)
    
