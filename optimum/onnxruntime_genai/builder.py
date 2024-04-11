# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""
Run this script to create the desired ONNX model.
"""

from onnx import helper, numpy_helper, TensorProto, external_data_helper, save_model
from onnxruntime.quantization.matmul_4bits_quantizer import MatMul4BitsQuantizer
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import numpy as np
import torch

import argparse
import gc
import json
import os
import textwrap

class Model:
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        self.context_length = config.max_position_embeddings
        self.window_size = config.sliding_window if hasattr(config, "sliding_window") else -1  # default is -1 in GroupQueryAttention kernel
        self.intermediate_size = config.intermediate_size
        self.hidden_size = config.hidden_size
        self.num_kv_heads = config.num_key_value_heads if hasattr(config, "num_key_value_heads") else config.num_attention_heads
        self.num_attn_heads = config.num_attention_heads
        self.head_size = config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads
        self.num_layers = int(extra_options["num_hidden_layers"]) if "num_hidden_layers" in extra_options else config.num_hidden_layers
        self.vocab_size = config.vocab_size
        self.activation = config.hidden_act

        self.model_name_or_path = config._name_or_path
        self.model_type = config.architectures[0]
        self.io_dtype = io_dtype      # {'fp16', 'fp32'}
        self.onnx_dtype = onnx_dtype  # {"int4", "fp16", "fp32"}
        self.ep = ep

        self.cache_dir = cache_dir
        self.filename = extra_options["filename"] if "filename" in extra_options else "model.onnx"
        self.extra_options = extra_options
        
        self.inputs = []
        self.outputs = []
        self.initializers = []
        self.value_infos = []
        self.nodes = []

        # Map input names to input shapes
        self.input_shapes = {
            "input_ids": ["batch_size", "sequence_length"],
            "attention_mask": ["batch_size", "total_sequence_length"],
            "position_ids": ["batch_size", "sequence_length"],
        }

        # Store names of Constant ops already created
        self.constants = set()

        # Map TensorProto dtypes to NumPy dtypes
        self.to_numpy_dtype = {
            TensorProto.INT8: np.uint8,
            TensorProto.INT32: np.int32,
            TensorProto.INT64: np.int64,
            TensorProto.FLOAT16: np.float16,
            TensorProto.FLOAT: np.float32,
        }

        # Map TensorProto dtypes to string dtypes
        self.to_str_dtype = {
            TensorProto.INT8: "TensorProto.INT8",
            TensorProto.INT32: "TensorProto.INT32",
            TensorProto.INT64: "TensorProto.INT64",
            TensorProto.FLOAT16: "TensorProto.FLOAT16",
            TensorProto.FLOAT: "TensorProto.FLOAT",
        }
        
        # Mask-specific variables
        self.mask_attrs = {
            "mask_name": "",            # Name of node that outputs 4D causal attention mask (used as add_qk in MultiHeadAttention)
            "seqlens_k": "",            # Sum of each row in attention mask - 1 (used as input to GroupQueryAttention)
            "total_seq_len": "",        # Size of total sequence length in attention mask (used as input to GroupQueryAttention)
        }

        # Embedding-specific variables
        self.embed_attrs = {
            "normalize": False,         # Normalize output of Embedding layer
        }

        # LayerNorm-specific variables
        self.layernorm_attrs = {
            "simple": True,             # Use SimplifiedLayerNorm/SkipSimplifiedLayerNorm vs. LayerNorm/SkipLayerNorm
            "first_layernorm": True,    # 1st LayerNorm = LayerNorm, then SkipLayerNorm for all subsequent LayerNorms
            "last_layernorm": False,    # Last LayerNorm = SkipLayerNorm with only output 0 (no output 3)
            "root_input": "",           # Root input from parent node for LayerNorm and SkipLayerNorm
            "skip_input": "",           # Skip input from parent node for SkipLayerNorm
            "output_0": "",             # Output 0 for LayerNorm and SkipLayerNorm
            "output_3": "",             # Output 3 for SkipLayerNorm
            "add_offset": 0,            # Offset value for LayerNorm weight
        }

        # RotaryEmbedding-specific variables
        partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
        rope_theta = config.rope_theta if hasattr(config, "rope_theta") else 10000
        self.rotemb_attrs = {
            "create_rotary_embedding_caches": True,          # Create cos/sin caches for rotary embeddings
            "partial_rotary_factor": partial_rotary_factor,  # Factor for partial rotary embeddings
            "num_heads": 0,                                  # For partial rotary embeddings (RotaryEmbedding kernel expects a default value of 0)
            "rotary_embedding_dim": 0,                       # For partial rotary embeddings (RotaryEmbedding kernel expects a default value of 0)
            "theta": rope_theta,                             # Base value if calculating cos/sin caches from scratch
        }

        # Attention-specific variables (MHA, GQA, GQA + Rot.Emb., etc.)
        self.attention_attrs = {
            "op_type": "MultiHeadAttention",                                # Attention op to use
            "use_gqa": ep == "cuda" and io_dtype == TensorProto.FLOAT16     # Check if GroupQueryAttention can be used
        }
        if self.attention_attrs["use_gqa"]:
            self.attention_attrs["op_type"] = "GroupQueryAttention"

        # Quantization-specific variables (INT4, INT8, etc.)
        self.quant_attrs = {
            "int4": {
                "block_size": int(extra_options["int4_block_size"]) if "int4_block_size" in extra_options else 32,
                "accuracy_level": int(extra_options["int4_accuracy_level"]) if "int4_accuracy_level" in extra_options else None,
            }
        }

    def make_genai_config(self, model_name_or_path, extra_kwargs, out_dir):
        config = GenerationConfig.from_pretrained(model_name_or_path, **extra_kwargs)
        inputs = dict(zip(self.input_shapes.keys(), self.input_shapes.keys()))
        inputs.update({
            "past_key_names": "past_key_values.%d.key",
            "past_value_names": "past_key_values.%d.value",
        })
        genai_config = {
            "model": {
                "bos_token_id": config.bos_token_id,
                "context_length": self.context_length,
                "decoder": {
                    "session_options" : {
                        "log_id": "onnxruntime-genai",
                        "provider_options" : []
                    },
                    "filename": self.filename,
                    "head_size": self.head_size,
                    "hidden_size": self.hidden_size,
                    "inputs": inputs,
                    "outputs": {
                        "logits": "logits",
                        "present_key_names": "present.%d.key",
                        "present_value_names": "present.%d.value",
                    },
                    "num_attention_heads": self.num_attn_heads,
                    "num_hidden_layers": self.num_layers,
                    "num_key_value_heads": self.num_kv_heads,
                },
                "eos_token_id": config.eos_token_id,
                "pad_token_id": config.pad_token_id if hasattr(config, "pad_token_id") and config.pad_token_id != None else config.eos_token_id,
                "type": self.model_type[ : self.model_type.find("For")].lower(),
                "vocab_size": self.vocab_size,
            },
            "search": {
                "diversity_penalty": config.diversity_penalty if hasattr(config, "diversity_penalty") else 0.0,
                "do_sample": config.do_sample if hasattr(config, "do_sample") else False,
                "early_stopping": True,
                "length_penalty": config.length_penalty if hasattr(config, "length_penalty") else 1.0,
                "max_length": self.context_length,
                "min_length": 0,
                "no_repeat_ngram_size": config.no_repeat_ngram_size if hasattr(config, "no_repeat_ngram_size") else 0,
                "num_beams": config.num_beams if hasattr(config, "num_beams") else 1,
                "num_return_sequences": config.num_return_sequences if hasattr(config, "num_return_sequences") else 1,
                "past_present_share_buffer": self.attention_attrs["op_type"] == "GroupQueryAttention",
                "repetition_penalty": config.repetition_penalty if hasattr(config, "repetition_penalty") else 1.0,
                "temperature": config.temperature if hasattr(config, "temperature") else 1.0,
                "top_k": 1,
                "top_p": config.top_p if hasattr(config, "top_p") else 1.0,
            },
        }

        if self.ep == "cuda":
            cuda_options = { "cuda" : { } }
            genai_config["model"]["decoder"]["session_options"]["provider_options"].append(cuda_options)

        print(f"Saving GenAI config in {out_dir}")
        with open(os.path.join(out_dir,"genai_config.json"), "w") as f:
            json.dump(genai_config, f, indent=4)

    def save_processing(self, model_name_or_path, extra_kwargs, out_dir):
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **extra_kwargs)
        print(f"Saving processing files in {out_dir} for GenAI")
        tokenizer.save_pretrained(out_dir)

    def save_model(self, out_dir):
        print(f"Saving ONNX model in {out_dir}")
        gc.collect()

        # Create ONNX model
        model = helper.make_model(
            opset_imports=[self.clear_field(helper.make_operatorsetid('', 14), 'domain'), helper.make_operatorsetid('com.microsoft', 1)],
            ir_version=7,
            producer_name="onnxruntime-genai",
            producer_version="0.0.0",
            graph=self.make_graph(
                name="main_graph",
                inputs=self.inputs,
                outputs=self.outputs,
                initializer=self.initializers,
                value_info=self.value_infos,
                nodes=self.nodes,
            )
        )

        # Load external data into ONNX model
        external_data_helper.load_external_data_for_model(model, self.cache_dir)

        # Delete external data files on disk before re-saving
        for path in os.listdir(self.cache_dir):
            if path.endswith(".bin"):
                os.remove(os.path.join(self.cache_dir, path))

        # Delete temporary cache dir if empty
        if len(os.listdir(self.cache_dir)) == 0:
            os.rmdir(self.cache_dir)

        # Quantize ONNX model to desired precision
        # TODO: Replace by quantizing the MatMuls as they are created
        if self.onnx_dtype == "int4":
            model = self.to_int4(model)

        # Save ONNX model with only one external data file and delete any existing duplicate copies
        out_path = os.path.join(out_dir, self.filename)
        data_path = os.path.join(out_dir, os.path.basename(out_path) + ".data")
        if os.path.exists(out_path):
            print(f"Overwriting {out_path}")
            os.remove(out_path)
        if os.path.exists(data_path):
            print(f"Overwriting {data_path}")
            os.remove(data_path)
        save_model(
            model,
            out_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=os.path.basename(data_path),
            size_threshold=0,
            convert_attribute=False,
        )

    def to_int4(self, model):
        quant = MatMul4BitsQuantizer(
            model=model,
            block_size=self.quant_attrs["int4"]["block_size"],
            is_symmetric=True,
            accuracy_level=self.quant_attrs["int4"]["accuracy_level"],
            nodes_to_exclude=[],
        )
        quant.process()
        return quant.model.model

    def clear_field(self, proto, field):
        proto.ClearField(field)
        return proto

    def order_repeated_field(self, repeated_proto, key_name, order):
        order = list(order)
        repeated_proto.sort(key=lambda x: order.index(getattr(x, key_name)))

    def make_external_tensor(self, np_data, name, **kwargs):
        tensor = numpy_helper.from_array(np_data)
        tensor.name = name

        filename = f"{name}.bin"
        external_data_helper.set_external_data(tensor, location=filename)
        with open(os.path.join(self.cache_dir, filename), "wb") as f:
            f.write(tensor.raw_data)
        tensor.ClearField("raw_data")
        tensor.data_location = TensorProto.EXTERNAL

        self.initializers.append(tensor)

    def make_node(self, op_type, inputs, outputs, name=None, doc_string=None, domain=None, **kwargs):
        # Save any constants as nodes
        for input_name in inputs:
            if input_name.startswith("/model/constants") and input_name not in self.constants:
                self.make_constant(input_name)

        node = helper.make_node(op_type, inputs, outputs, name, doc_string, domain, **kwargs)
        if doc_string == '':
            node.doc_string = ''
        self.order_repeated_field(node.attribute, 'name', kwargs.keys())
        self.nodes.append(node)

    def make_value_info(self, name, dtype, shape):
        value_info = helper.make_tensor_value_info(name, dtype, shape=shape)
        self.value_infos.append(value_info)

    def make_graph(self, *args, doc_string=None, **kwargs):
        graph = helper.make_graph(*args, doc_string=doc_string, **kwargs)
        if doc_string == '':
            graph.doc_string = ''
        return graph

    def make_inputs_and_outputs(self):
        # Add model-specific inputs to list of model inputs
        inputs = []
        for name in self.model_inputs:
            shape = self.input_shapes[name]
            inputs.append(helper.make_tensor_value_info(name, TensorProto.INT64, shape=shape))

        # Add model-specific outputs to list of model outputs
        outputs = [
            helper.make_tensor_value_info("logits", self.io_dtype, shape=["batch_size", "sequence_length", self.vocab_size])
        ]

        # Add KV cache to inputs and outputs
        for i in range(self.num_layers):
            # Add KV cache to inputs
            key_name = f"past_key_values.{i}.key"
            inputs.append(helper.make_tensor_value_info(key_name, self.io_dtype, shape=["batch_size", self.num_kv_heads, "past_sequence_length", self.head_size]))
            value_name = f"past_key_values.{i}.value"
            inputs.append(helper.make_tensor_value_info(value_name, self.io_dtype, shape=["batch_size", self.num_kv_heads, "past_sequence_length", self.head_size]))

            # Add KV cache to outputs
            key_name = f"present.{i}.key"
            outputs.append(helper.make_tensor_value_info(key_name, self.io_dtype, shape=["batch_size", self.num_kv_heads, "total_sequence_length", self.head_size]))
            value_name = f"present.{i}.value"
            outputs.append(helper.make_tensor_value_info(value_name, self.io_dtype, shape=["batch_size", self.num_kv_heads, "total_sequence_length", self.head_size]))

        self.inputs = inputs
        self.outputs = outputs

    def make_constant(self, name):
        # Make constant ops for 0, 1, 2, 3, etc.
        # Format of name is "/model/constants/{dtype}/{shape}/{num}"
        path = name.split("/")
        onnx_dtype, dims, num = eval(path[-3]), path[-2], eval(path[-1])
        np_dtype = self.to_numpy_dtype[onnx_dtype]
        value = numpy_helper.from_array(np.array(num if dims == "0D" else list(num) if type(num) == tuple else [num], dtype=np_dtype), name=name.replace("constants", "numpy_helper"))

        node_name = name.replace("constants", "constant_nodes")
        self.make_node("Constant", inputs=[], outputs=[name], name=node_name, value=value)
        self.make_value_info(name, onnx_dtype, shape=[])
        self.constants.add(name)

    def make_gather(self, name, inputs, axis):
        output = f"{name}/output_0"
        self.make_node("Gather", inputs=inputs, outputs=[output], name=name, axis=axis)
        self.make_value_info(output, TensorProto.INT64, shape=[])

    def make_reshape(self, name, inputs, dtype, shape):
        output = f"{name}/output_0"
        self.make_node("Reshape", inputs=inputs, outputs=[output], name=name)
        self.make_value_info(output, dtype, shape=shape)

    def make_shape(self, name, root_input, shape):
        output = f"{name}/output_0"
        self.make_node("Shape", inputs=[root_input], outputs=[output], name=name)
        self.make_value_info(output, TensorProto.INT64, shape=shape)

    def make_constant_of_shape(self, name, root_input, value, dtype, shape):
        output = f"{name}/output_0"
        self.make_node("ConstantOfShape", inputs=[root_input], outputs=[output], name=name, value=value)
        self.make_value_info(output, dtype, shape=shape)

    def make_unsqueeze(self, name, inputs, dtype, shape):
        output = f"{name}/output_0"
        self.make_node("Unsqueeze", inputs=inputs, outputs=[output], name=name)
        self.make_value_info(output, dtype, shape=shape)

    def make_squeeze(self, name, inputs):
        output = f"{name}/output_0"
        self.make_node("Squeeze", inputs=inputs, outputs=[output], name=name)
        self.make_value_info(output, TensorProto.INT64, shape=[])

    def make_concat(self, name, inputs, dtype, shape, axis=0):
        output = f"{name}/output_0"
        self.make_node("Concat", inputs=inputs, outputs=[output], name=name, axis=axis)
        self.make_value_info(output, dtype, shape=shape)

    def make_equal(self, name, inputs, shape):
        output = f"{name}/output_0"
        self.make_node("Equal", inputs=inputs, outputs=[output], name=name)
        self.make_value_info(output, TensorProto.BOOL, shape=shape)

    def make_where(self, name, inputs, dtype, shape):
        output = f"{name}/output_0"
        self.make_node("Where", inputs=inputs, outputs=[output], name=name)
        self.make_value_info(output, dtype, shape=shape)

    def make_expand(self, name, inputs, dtype, shape):
        output = f"{name}/output_0"
        self.make_node("Expand", inputs=inputs, outputs=[output], name=name)
        self.make_value_info(output, dtype, shape=shape)

    def make_reduce_sum(self, name, inputs, dtype, shape):
        output = f"{name}/output_0"
        self.make_node("ReduceSum", inputs=inputs, outputs=[output], name=name)
        self.make_value_info(output, dtype, shape=shape)

    def make_cast(self, name, root_input, dtype, shape):
        output = f"{name}/output_0"
        self.make_node("Cast", inputs=[root_input], outputs=[output], name=name, to=dtype)
        self.make_value_info(output, dtype, shape=shape)

    def make_add(self, name, inputs, dtype, shape):
        output = f"{name}/output_0"
        self.make_node("Add", inputs=inputs, outputs=[output], name=name)
        self.make_value_info(output, dtype, shape=shape)

    def make_sub(self, name, inputs, dtype, shape):
        output = f"{name}/output_0"
        self.make_node("Sub", inputs=inputs, outputs=[output], name=name)
        self.make_value_info(output, dtype, shape=shape)

    def make_less(self, name, inputs):
        output = f"{name}/output_0"
        self.make_node("Less", inputs=inputs, outputs=[output], name=name)
        self.make_value_info(output, TensorProto.BOOL, shape=None)

    def make_range(self, name, inputs):
        output = f"{name}/output_0"
        self.make_node("Range", inputs=inputs, outputs=[output], name=name)
        self.make_value_info(output, TensorProto.INT64, shape=["unk"])

    def make_slice(self, name, inputs):
        output = f"{name}/output_0"
        self.make_node("Slice", inputs=inputs, outputs=[output], name=name)
        self.make_value_info(output, TensorProto.INT64, shape=[1])

    def make_sigmoid(self, name, root_input, dtype, shape):
        output = f"{name}/output_0"
        self.make_node("Sigmoid", inputs=[root_input], outputs=[output], name=name)
        self.make_value_info(output, dtype, shape=shape)

    def make_mul(self, name, inputs, dtype, shape):
        output = f"{name}/output_0"
        self.make_node("Mul", inputs=inputs, outputs=[output], name=name)
        self.make_value_info(output, dtype, shape=shape)

    def make_transpose(self, name, root_input, dtype, shape, perm):
        output = f"{name}/output_0"
        self.make_node("Transpose", inputs=[root_input], outputs=[output], perm=perm)
        self.make_value_info(output, dtype, shape=shape)

    def make_matmul(self, matmul, name, root_input, **kwargs):
        self.make_matmul_fp16_or_fp32(matmul, name, root_input, **kwargs)

        # TODO: add other dtypes
        # if self.onnx_dtype in {"fp16", "fp32"}:
        #     self.make_matmul_fp16_or_fp32(matmul, name, root_input, **kwargs)
        # elif self.onnx_dtype == "int8":
        #     pass
        # elif self.onnx_dtype == "int4":
        #     int4_name = f"{name}NBits"
        #     self.make_matmul_int4(matmul, int4_name, root_input, **kwargs)

    def make_matmul_fp16_or_fp32(self, matmul, name, root_input, **kwargs):
        weight = name[1:].replace("/", ".") + ".weight"
        self.make_external_tensor(matmul.transpose().astype(self.to_numpy_dtype[self.io_dtype]), weight)

        last_dim = matmul.shape[0]
        output = "logits" if kwargs.get("logits", False) else f"{name}/output_0"
        self.make_node("MatMul", inputs=[root_input, weight], outputs=[output], name=name)
        self.make_value_info(output, self.io_dtype, shape=['batch_size', 'sequence_length', last_dim])

    # TODO: quantize weights, then save new MatMul numpy weights for onnx model
    # def make_matmul_int4(self, matmul, name, root_input, **kwargs):
    #     weight = name[1:].replace("/", ".") + ".weight"
    #     scales = name[1:].replace("/", ".") + ".scales"

    #     output = "logits" if kwargs.get("logits", False) else f"{name}/output_0"
    #     self.make_node("MatMulNBits", inputs=[root_input, weight, scales], outputs=[output], name=name)
    #     self.make_value_info(output, self.io_dtype, shape=['batch_size', 'sequence_length', self.hidden_size])

    # TODO: make packed QKV MatMul
    # def make_packed_matmul(self, q_matmul, k_matmul, v_matmul, name, root_input, **kwargs):
    #     pass

    def make_add_bias(self, add, name, root_input, **kwargs):
        bias = name[1:].replace("/", ".") + ".bias"
        self.make_external_tensor(add.astype(self.to_numpy_dtype[self.io_dtype]), bias)
        
        add_bias_inputs = [root_input, bias]
        shape = ['batch_size', 'sequence_length', add.shape[0]]

        if "logits" in kwargs:
            output = "logits"
            self.make_node("Add", inputs=add_bias_inputs, outputs=[output], name=name)
            self.make_value_info(output, dtype=self.io_dtype, shape=shape)
        else:
            self.make_add(name, add_bias_inputs, dtype=self.io_dtype, shape=shape)

    def make_embedding(self, embedding):
        weight = "model.embed_tokens.weight"
        self.make_external_tensor(embedding.astype(self.to_numpy_dtype[self.io_dtype]), weight)

        basename = "/model/embed_tokens"
        gather_name = f"{basename}/Gather"
        gather_output = f"{gather_name}/output_0"
        self.make_node('Gather', inputs=[weight, 'input_ids'], outputs=[gather_output], name=gather_name)
        self.make_value_info(gather_output, self.io_dtype, shape=['batch_size', 'sequence_length', self.hidden_size])

        if self.embed_attrs["normalize"]:
            # Normalize the embeddings
            norm_name = f"{basename}/Mul"
            norm_inputs = [gather_output, f"/model/constants/{self.to_str_dtype[self.io_dtype]}/0D/{np.round(np.sqrt(self.hidden_size), decimals=2)}"]
            norm_output = f"{norm_name}/output_0"
            self.make_node('Mul', inputs=norm_inputs, outputs=[norm_output], name=norm_name)
            self.make_value_info(norm_output, self.io_dtype, shape=['batch_size', 'sequence_length', self.hidden_size])

            layernorm_attrs_value = norm_output
        else:
            layernorm_attrs_value = gather_output

        self.layernorm_attrs["root_input"] = layernorm_attrs_value
        self.layernorm_attrs["skip_input"] = layernorm_attrs_value

    def make_layernorm(self, layer_id, layernorm, skip, simple, location):
        root_input = self.layernorm_attrs["root_input"]
        skip_input = self.layernorm_attrs["skip_input"]

        weight = f"model.layers.{layer_id}.{location}_layernorm.weight"
        self.make_external_tensor(layernorm.weight.detach().numpy().astype(self.to_numpy_dtype[self.io_dtype]) + self.layernorm_attrs["add_offset"], weight)
        bias = f"model.layers.{layer_id}.{location}_layernorm.bias"
        if not simple:
            self.make_external_tensor(layernorm.bias.detach().numpy().astype(self.to_numpy_dtype[self.io_dtype]), bias)

        inputs = [root_input, skip_input, weight] if skip else [root_input, weight]
        if not simple:
            inputs.append(bias)

        name = f"/model/layers.{layer_id}/{location}_layernorm/{'Skip' if skip else ''}LayerNorm"
        op_type = f"{'Skip' if skip else ''}{'Simplified' if simple else ''}LayerNormalization"
        kwargs = {"epsilon": 9.999999747378752e-06}
        if not skip:
            kwargs.update({"axis": -1, "stash_type": 1})

        output_0 = f"/model/layers.{layer_id}/{location}_layernorm/output_0"
        output_3 = f"/model/layers.{layer_id}/{location}_layernorm/output_3"
        outputs = [output_0, "", "", output_3] if skip and not self.layernorm_attrs["last_layernorm"] else [output_0]

        self.make_node(op_type, inputs=inputs, outputs=outputs, name=name, domain=("com.microsoft" if skip else None), **kwargs)
        self.make_value_info(output_0, self.io_dtype, shape=['batch_size', 'sequence_length', self.hidden_size])
        if skip and not self.layernorm_attrs["last_layernorm"]:
            self.make_value_info(output_3, self.io_dtype, shape=['batch_size', 'sequence_length', self.hidden_size])

        # Update LayerNorm attributes
        self.layernorm_attrs["output_0"] = output_0
        if skip and not self.layernorm_attrs["last_layernorm"]:
            self.layernorm_attrs["output_3"] = output_3

            # Assign output 3 of current SkipLayerNorm as root input to next SkipLayerNorm
            self.layernorm_attrs["root_input"] = output_3

        return output_0

    def make_rotary_embedding_caches(self, rotemb):
        cos_cache_name, sin_cache_name = "cos_cache", "sin_cache"

        if self.rotemb_attrs["create_rotary_embedding_caches"]:
            if not hasattr(rotemb, "cos_cached"):
                # Create cos/sin caches if not already created
                dim = int(self.rotemb_attrs["partial_rotary_factor"] * self.head_size)
                inv_freq = 1.0 / (self.rotemb_attrs["theta"] ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
                t = torch.arange(self.context_length, dtype=torch.int64).type_as(inv_freq)
                freqs = torch.outer(t, inv_freq)
                emb = torch.cat((freqs, freqs), dim=-1)
                cos_cache, sin_cache = emb.cos(), emb.sin()
            else:
                cos_cache, sin_cache = rotemb.cos_cached, rotemb.sin_cached

            # Reshape cos/sin cache from (M, H) to (M, H/2)
            hidden_dim = cos_cache.shape[-1]
            cos_cache = cos_cache.squeeze()[:, : (hidden_dim // 2)].detach().numpy()
            self.make_external_tensor(cos_cache.astype(self.to_numpy_dtype[self.io_dtype]), cos_cache_name)
            sin_cache = sin_cache.squeeze()[:, : (hidden_dim // 2)].detach().numpy()
            self.make_external_tensor(sin_cache.astype(self.to_numpy_dtype[self.io_dtype]), sin_cache_name)

            self.rotemb_attrs["create_rotary_embedding_caches"] = False

        return cos_cache_name, sin_cache_name

    def make_rotary_embedding(self, rotemb, name, root_input, **kwargs):
        cos_cache_name, sin_cache_name = self.make_rotary_embedding_caches(rotemb)

        inputs = [root_input, kwargs.pop("position_ids"), cos_cache_name, sin_cache_name]
        output = f"{name}/output_0"
        self.make_node("RotaryEmbedding", inputs=inputs, outputs=[output], name=name, domain="com.microsoft", interleaved=0, **kwargs)
        self.make_value_info(output, self.io_dtype, shape=['batch_size', 'sequence_length', self.head_size * (self.num_kv_heads if "k_rotary" in name else self.num_attn_heads)])

    # TODO: This function and any corresponding changes to support it are temporary until ORT supports GQA for CPU
    def make_repeat_kv(self, layer_id, root_input, past_kv, present_kv, **kwargs):
        # Make subgraph that repeats tensor of shape (batch_size, sequence_length, num_kv_heads, head_size)
        # to shape (batch_size, sequence_length, num_attn_heads, head_size) in an interleaved pattern
        # and updates the KV caches
        #
        #           root_input
        #                |
        #             Reshape
        #                |
        #            Transpose
        #                |
        #                |   past_kv
        #                |  /
        #             Concat
        #                |  \
        #                |   present_kv
        #                |
        #        +-------+---------+
        #        |                 |        
        #        |               Shape
        #        |                 |
        #        |     +-----------+-----------+-----------+
        #        |     |           |           |           |
        #        |   Gather     Gather      Gather      Gather
        #        |   (idx=0)    (idx=1)     (idx=2)     (idx=3)
        #        |     |           |           |           |
        #        | Unsqueeze   Unsqueeze   Unsqueeze   Unsqueeze
        #        |     |           |           |           |
        #        |     +-----------+-----------+-----------+
        #        |                 |
        #        |                 +-----------------------+
        #        |                 |                       |
        #        |                 |                      Mul
        #        |                 |                       |
        #        |              Concat                   Concat
        #        |               (5D)                     (4D)
        #        |                 |                       |
        #        |              Reshape                    |
        #        |             /   |   \                   |
        #        |            /    |    \                  |
        #        |           /     |     \                /
        #        |          /      |      \              /
        #        |         /       |       \            /
        #        |        /      Shape      \          /
        #        |       /         |         \        /
        #        |      |   ConstantOfShape   \      /
        #        |       \         |       \   \    /
        #        |        \        |       Mul  |  /
        #        |         \       |        |  /  /
        #        |          \      |      Equal  /
        #        |           \     |       /    /
        #         \           \    |      /    /
        #          \           \   |     /    /
        #           \           \  |    /    /
        #            \           \ |   /    /
        #         Unsqueeze       Where    /
        #             \           /       /
        #              \         /       /
        #               \       /       /
        #                \     /       /
        #                 Expand      /
        #                    |       /
        #                    |      /
        #                    |     /
        #                    |    /
        #                    |   /
        #                 Reshape
        #                    |
        #                Transpose
        #                    |
        #                 Reshape
        basename = f"/model/layers.{layer_id}/attn/{'k_proj' if past_kv.endswith('key') else 'v_proj'}/repeat_kv"

        # Make the initial subgraph
        #
        #                                                       +------> Gather --> Unsqueeze -----+
        #                                                       |                                  |
        #                                         past_kv       +------> Gather --> Unsqueeze -----+---> Mul --> Concat (4D)
        #                                            |          |                                  |
        # root_input --> Reshape --> Transpose --> Concat --> Shape ---> Gather --> Unsqueeze -----+---> Concat (5D)
        #                                            |          |                                  |
        #                                        present_kv     +------> Gather --> Unsqueeze -----+
        reshape_1_name = f"{basename}/Reshape_1"
        reshape_1_inputs = [root_input, f"/model/constants/TensorProto.INT64/1D/0, 0, {self.num_kv_heads}, -1"]
        self.make_reshape(reshape_1_name, reshape_1_inputs, dtype=self.io_dtype, shape=['batch_size', 'sequence_length', self.num_kv_heads, self.head_size])
        transpose_1_name = f"{basename}/Transpose_1"
        transpose_1_input = f"{reshape_1_name}/output_0"
        self.make_transpose(transpose_1_name, transpose_1_input, dtype=self.io_dtype, shape=['batch_size', self.num_kv_heads, 'sequence_length', self.head_size], perm=[0,2,1,3])
        concat_1_name = f"{basename}/Concat_1"
        concat_1_inputs = [past_kv, f"{transpose_1_name}/output_0"]
        self.make_node("Concat", inputs=concat_1_inputs, outputs=[present_kv], name=concat_1_name, axis=2)
        
        shape_1_name = f"{basename}/Shape_1"
        self.make_shape(shape_1_name, present_kv, shape=[4])
        gather_1_name = f"{basename}/Gather_1"
        gather_1_inputs = [f"{shape_1_name}/output_0", "/model/constants/TensorProto.INT64/0D/0"]
        self.make_gather(gather_1_name, gather_1_inputs, axis=0)
        unsqueeze_1_name = f"{basename}/Unsqueeze_1"
        unsqueeze_1_inputs = [f"{gather_1_name}/output_0", "/model/constants/TensorProto.INT64/1D/0"]
        self.make_unsqueeze(unsqueeze_1_name, unsqueeze_1_inputs, dtype=TensorProto.INT64, shape=[1])
        gather_2_name = f"{basename}/Gather_2"
        gather_2_inputs = [f"{shape_1_name}/output_0", "/model/constants/TensorProto.INT64/0D/1"]
        self.make_gather(gather_2_name, gather_2_inputs, axis=0)
        unsqueeze_2_name = f"{basename}/Unsqueeze_2"
        unsqueeze_2_inputs = [f"{gather_2_name}/output_0", "/model/constants/TensorProto.INT64/1D/0"]
        self.make_unsqueeze(unsqueeze_2_name, unsqueeze_2_inputs, dtype=TensorProto.INT64, shape=[1])
        gather_3_name = f"{basename}/Gather_3"
        gather_3_inputs = [f"{shape_1_name}/output_0", "/model/constants/TensorProto.INT64/0D/2"]
        self.make_gather(gather_3_name, gather_3_inputs, axis=0)
        unsqueeze_3_name = f"{basename}/Unsqueeze_3"
        unsqueeze_3_inputs = [f"{gather_3_name}/output_0", "/model/constants/TensorProto.INT64/1D/0"]
        self.make_unsqueeze(unsqueeze_3_name, unsqueeze_3_inputs, dtype=TensorProto.INT64, shape=[1])
        gather_4_name = f"{basename}/Gather_4"
        gather_4_inputs = [f"{shape_1_name}/output_0", "/model/constants/TensorProto.INT64/0D/3"]
        self.make_gather(gather_4_name, gather_4_inputs, axis=0)
        unsqueeze_4_name = f"{basename}/Unsqueeze_4"
        unsqueeze_4_inputs = [f"{gather_4_name}/output_0", "/model/constants/TensorProto.INT64/1D/0"]
        self.make_unsqueeze(unsqueeze_4_name, unsqueeze_4_inputs, dtype=TensorProto.INT64, shape=[1])
        concat_2_name = f"{basename}/Concat_2"
        concat_2_inputs = [f"{unsqueeze_1_name}/output_0", f"{unsqueeze_2_name}/output_0", f"/model/constants/TensorProto.INT64/1D/{self.num_attn_heads // self.num_kv_heads}", f"{unsqueeze_3_name}/output_0", f"{unsqueeze_4_name}/output_0"]
        self.make_concat(concat_2_name, concat_2_inputs, dtype=TensorProto.INT64, shape=[5], axis=0)

        mul_1_name = f"{basename}/Mul_1"
        mul_1_inputs = [f"{unsqueeze_2_name}/output_0", f"/model/constants/TensorProto.INT64/0D/{self.num_attn_heads // self.num_kv_heads}"]
        self.make_mul(mul_1_name, mul_1_inputs, dtype=TensorProto.INT64, shape=None)
        concat_3_name = f"{basename}/Concat_3"
        concat_3_inputs = [f"{unsqueeze_1_name}/output_0", f"{mul_1_name}/output_0", f"{unsqueeze_3_name}/output_0", f"{unsqueeze_4_name}/output_0"]
        self.make_concat(concat_3_name, concat_3_inputs, dtype=TensorProto.INT64, shape=[4], axis=0)

        # Make the subgraph that follows the initial subgraph
        #
        #                               Mul ---> Equal
        #                              /              \
        # Reshape --> Shape --> ConstantOfShape --> Where
        #    |                                        |
        #    +----------------------------------------+
        reshape_2_name = f"{basename}/Reshape_2"
        reshape_2_inputs = [f"{concat_2_name}/output_0", "/model/constants/TensorProto.INT64/1D/-1"]
        self.make_reshape(reshape_2_name, reshape_2_inputs, dtype=TensorProto.INT64, shape=None)
        shape_2_name = f"{basename}/Shape_2"
        self.make_shape(shape_2_name, f"{reshape_2_name}/output_0", shape=[1])
        constant_shape_name = f"{basename}/ConstantOfShape"
        constant_shape_value = numpy_helper.from_array(np.array([1], dtype="int64"))
        self.make_constant_of_shape(constant_shape_name, f"{shape_2_name}/output_0", value=constant_shape_value, dtype=TensorProto.INT64, shape=[5])
        mul_2_name = f"{basename}/Mul"
        mul_2_inputs = [f"{constant_shape_name}/output_0", "/model/constants/TensorProto.INT64/0D/-1"]
        self.make_mul(mul_2_name, mul_2_inputs, dtype=TensorProto.INT64, shape=[5])
        equal_name = f"{basename}/Equal"
        equal_inputs = [f"{reshape_2_name}/output_0", f"{mul_2_name}/output_0"]
        self.make_equal(equal_name, equal_inputs, shape=[5])
        where_name = f"{basename}/Where"
        where_inputs = [f"{equal_name}/output_0", f"{constant_shape_name}/output_0", f"{reshape_2_name}/output_0"]
        self.make_where(where_name, where_inputs, dtype=TensorProto.INT64, shape=[5])

        # Make the final nodes
        #
        # Where (from above)  Concat (from above)
        #                   \           \
        # Unsqueeze --> Expand --> Reshape --> Transpose --> Reshape
        unsqueeze_5_name = f"{basename}/Unsqueeze_5"
        unsqueeze_5_inputs = [present_kv, "/model/constants/TensorProto.INT64/1D/2"]
        self.make_unsqueeze(unsqueeze_5_name, unsqueeze_5_inputs, dtype=self.io_dtype, shape=['batch_size', self.num_kv_heads, 1, 'sequence_length', self.head_size])
        expand_name = f"{basename}/Expand"
        expand_inputs = [f"{unsqueeze_5_name}/output_0", f"{where_name}/output_0"]
        self.make_expand(expand_name, expand_inputs, dtype=self.io_dtype, shape=['batch_size', self.num_kv_heads, self.num_attn_heads // self.num_kv_heads, 'sequence_length', self.head_size])
        reshape_3_name = f"{basename}/Reshape_3"
        reshape_3_inputs = [f"{expand_name}/output_0", f"{concat_3_name}/output_0"]
        self.make_reshape(reshape_3_name, reshape_3_inputs, dtype=self.io_dtype, shape=['batch_size', self.num_attn_heads, 'sequence_length', self.head_size])
        transpose_2_name = f"{basename}/Transpose_2"
        transpose_2_input = f"{reshape_3_name}/output_0"
        self.make_transpose(transpose_2_name, transpose_2_input, dtype=self.io_dtype, shape=['batch_size', 'sequence_length', self.num_attn_heads, self.head_size], perm=[0,2,1,3])
        reshape_4_name = f"{basename}/Reshape_4"
        reshape_4_inputs = [f"{transpose_2_name}/output_0", f"/model/constants/TensorProto.INT64/1D/0, 0, {self.num_attn_heads * self.head_size}"]
        self.make_reshape(reshape_4_name, reshape_4_inputs, dtype=self.io_dtype, shape=['batch_size', 'sequence_length', self.num_attn_heads * self.head_size])

        input_to_attention = f"{reshape_4_name}/output_0"
        return input_to_attention

    def make_attention_op(self, name, **kwargs):
        op_type = self.attention_attrs["op_type"]
        
        if op_type == "MultiHeadAttention":
            self.make_multi_head_attention(name, add_qk=f"{self.mask_attrs['mask_name']}/output_0", **kwargs)
        elif op_type == "GroupQueryAttention":
            self.make_group_query_attention(name, seqlens_k=f"{self.mask_attrs['seqlens_k']}/output_0", total_seq_len=f"{self.mask_attrs['total_seq_len']}/output_0", **kwargs)
        else:
            raise NotImplementedError(f"The {op_type} op is not currently supported.")

    def make_multi_head_attention(self, name, **kwargs):
        inputs = [
            kwargs["q_path"], kwargs["k_path"], kwargs["v_path"], kwargs.get("bias", ""),
            kwargs.get("attn_mask", ""), kwargs.get("add_qk", ""),
            kwargs.get("past_k", ""), kwargs.get("past_v", ""),
        ]
        output = f"{name}/output_0"
        outputs = [output, kwargs.get("present_k", ""), kwargs.get("present_v", "")]
        self.make_node("MultiHeadAttention", inputs=inputs, outputs=outputs, name=name, domain="com.microsoft", num_heads=self.num_attn_heads)
        self.make_value_info(output, self.io_dtype, shape=['batch_size', 'sequence_length', self.head_size * self.num_attn_heads])

    def make_group_query_attention(self, name, **kwargs):
        inputs = [
            kwargs["q_path"], kwargs["k_path"], kwargs["v_path"],
            kwargs.get("past_k", ""), kwargs.get("past_v", ""),
            kwargs.get("seqlens_k", ""), kwargs.get("total_seq_len", ""),
        ]
        output = f"{name}/output_0"
        outputs = [output, kwargs.get("present_k", ""), kwargs.get("present_v", "")]
        self.make_node("GroupQueryAttention", inputs=inputs, outputs=outputs, name=name, domain="com.microsoft", num_heads=self.num_attn_heads, kv_num_heads=self.num_kv_heads, local_window_size=self.window_size)
        self.make_value_info(output, self.io_dtype, shape=['batch_size', 'sequence_length', self.head_size * self.num_attn_heads])

    def make_attention(self, layer_id, attention, root_input, **kwargs):
        # Make nodes for the Attention subgraph
        #
        # MultiHeadAttention example:
        #
        #               root_input
        #              /     |     \
        #       Q_MatMul  K_MatMul  V_MatMul  4D causal mask  past_key  past_value
        #           |        |         |            |            |           |
        #         Q_Add    K_Add     V_Add          +------------+-----------+
        #           |        |         |                         |
        #       Q_Rotary  K_Rotary     |                         |
        #           \        |        /                          |
        #            MultiHeadAttention--------------------------+
        #                    |
        #                O_MatMul
        #                    |
        #                  O_Add
        #
        # GroupQueryAttention example:
        #
        #               root_input
        #              /     |     \
        #       Q_MatMul  K_MatMul  V_MatMul  seqlens_k  total_seq_len  past_key  past_value
        #           |        |         |          |            |           |          |
        #         Q_Add    K_Add     V_Add        +------------+-----------+----------+
        #           |        |         |                       |
        #       Q_Rotary  K_Rotary     |                       |
        #           \        |        /                        |
        #            GroupQueryAttention-----------------------+
        #                    |
        #                O_MatMul
        #                    |
        #                  O_Add

        q_input_to_attention = ""
        k_input_to_attention = ""
        v_input_to_attention = ""

        # Make MatMul nodes
        q_matmul_name = f"/model/layers.{layer_id}/attn/q_proj/MatMul"
        self.make_matmul(attention.q_proj.weight.detach().numpy(), q_matmul_name, root_input)
        q_input_to_attention = f"{q_matmul_name}/output_0"
        k_matmul_name = f"/model/layers.{layer_id}/attn/k_proj/MatMul"
        self.make_matmul(attention.k_proj.weight.detach().numpy(), k_matmul_name, root_input)
        k_input_to_attention = f"{k_matmul_name}/output_0"
        v_matmul_name = f"/model/layers.{layer_id}/attn/v_proj/MatMul"
        self.make_matmul(attention.v_proj.weight.detach().numpy(), v_matmul_name, root_input)
        v_input_to_attention = f"{v_matmul_name}/output_0"

        # Make Add nodes (if bias exists)
        q_bias_exists = attention.q_proj.bias is not None
        k_bias_exists = attention.k_proj.bias is not None
        v_bias_exists = attention.v_proj.bias is not None

        if q_bias_exists:
            q_add_name = f"/model/layers.{layer_id}/attn/q_proj/Add"
            self.make_add_bias(attention.q_proj.bias.detach().numpy(), q_add_name, root_input=f"{q_matmul_name}/output_0")
            q_input_to_attention = f"{q_add_name}/output_0"
        if k_bias_exists:
            k_add_name = f"/model/layers.{layer_id}/attn/k_proj/Add"
            self.make_add_bias(attention.k_proj.bias.detach().numpy(), k_add_name, root_input=f"{k_matmul_name}/output_0")
            k_input_to_attention = f"{k_add_name}/output_0"
        if v_bias_exists:
            v_add_name = f"/model/layers.{layer_id}/attn/v_proj/Add"
            self.make_add_bias(attention.v_proj.bias.detach().numpy(), v_add_name, root_input=f"{v_matmul_name}/output_0")
            v_input_to_attention = f"{v_add_name}/output_0"

        # Make RotaryEmbedding nodes
        q_rotary_name = f"/model/layers.{layer_id}/attn/q_rotary/RotaryEmbedding"
        q_rotary_input = f"{q_matmul_name if not q_bias_exists else q_add_name}/output_0"
        self.make_rotary_embedding(attention.rotary_emb, q_rotary_name, q_rotary_input, position_ids=kwargs.get("position_ids", "position_ids"))
        q_input_to_attention = f"{q_rotary_name}/output_0"

        k_rotary_name = f"/model/layers.{layer_id}/attn/k_rotary/RotaryEmbedding"
        k_rotary_input = f"{k_matmul_name if not k_bias_exists else k_add_name}/output_0"
        self.make_rotary_embedding(attention.rotary_emb, k_rotary_name, k_rotary_input, position_ids=kwargs.get("position_ids", "position_ids"))
        k_input_to_attention = f"{k_rotary_name}/output_0"

        # Make repeat KV nodes (TODO: remove once ORT supports GQA for CPU)
        past_k = f"past_key_values.{layer_id}.key"
        past_v = f"past_key_values.{layer_id}.value"
        present_k = f"present.{layer_id}.key"
        present_v = f"present.{layer_id}.value"
        if self.num_attn_heads != self.num_kv_heads and not self.attention_attrs['use_gqa']:
            k_input_to_attention = self.make_repeat_kv(layer_id, k_input_to_attention, past_k, present_k)
            v_input_to_attention = self.make_repeat_kv(layer_id, v_input_to_attention, past_v, present_v)
            past_k, past_v, present_k, present_v = "", "", "", ""

        # Make attention node (e.g. MultiHeadAttention, GroupQueryAttention, etc.)
        attn_name = f"/model/layers.{layer_id}/attn/{self.attention_attrs['op_type']}"
        self.make_attention_op(
            attn_name, q_path=q_input_to_attention, k_path=k_input_to_attention, v_path=v_input_to_attention,
            past_k=past_k, past_v=past_v, present_k=present_k, present_v=present_v, **kwargs,
        )

        # Make MatMul node (output projection weight node)
        o_proj = 'o_proj' if hasattr(attention, 'o_proj') else 'dense'
        o_matmul_name = f"/model/layers.{layer_id}/attn/o_proj/MatMul"
        o_weight = eval(f"attention.{o_proj}.weight.detach().numpy()")
        self.make_matmul(o_weight, o_matmul_name, f"{attn_name}/output_0")

        # Make Add node (output projection bias node if bias exists)
        o_bias_exists = eval(f"attention.{o_proj}.bias") is not None
        if o_bias_exists:
            o_add_name = f"/model/layers.{layer_id}/attn/o_proj/Add"
            o_bias = eval(f"attention.{o_proj}.bias.detach().numpy()")
            self.make_add_bias(o_bias, o_add_name, root_input=f"{o_matmul_name}/output_0")

        # Assign output 0 of previous output node as skip input to next SkipLayerNorm
        self.layernorm_attrs["skip_input"] = f"{o_matmul_name if not o_bias_exists else o_add_name}/output_0"

    def make_mlp(self, layer_id, mlp, root_input):
        # Make nodes for the MLP subgraph
        #
        #           root_input
        #          /          \
        #   UpProjMatMul    GateProjMatMul
        #          \          |
        #           \     ActFunc
        #            \   /
        #             Mul
        #              |
        #        DownProjMatMul
        
        # Make MatMul nodes
        gate_name = f"/model/layers.{layer_id}/mlp/gate_proj/MatMul"
        self.make_matmul(mlp.gate_proj.weight.detach().numpy(), gate_name, root_input)
        up_name = f"/model/layers.{layer_id}/mlp/up_proj/MatMul"
        self.make_matmul(mlp.up_proj.weight.detach().numpy(), up_name, root_input)

        # Make activation node(s)
        act_fn_name = self.make_activation(layer_id, root_input=f"{gate_name}/output_0")

        # Make Mul node after activation
        mul_name = f"/model/layers.{layer_id}/mlp/Mul"
        mul_inputs = [f"{act_fn_name}/output_0", f"{up_name}/output_0"]
        self.make_mul(mul_name, mul_inputs, dtype=self.io_dtype, shape=["batch_size", "sequence_length", self.intermediate_size])

        # Make output MatMul node
        down_name = f"/model/layers.{layer_id}/mlp/down_proj/MatMul"
        self.make_matmul(mlp.down_proj.weight.detach().numpy(), down_name, f"{mul_name}/output_0")

        # Assign output 0 of previous MatMul as skip input to next SkipLayerNorm
        self.layernorm_attrs["skip_input"] = f"{down_name}/output_0"

    def make_activation_with_mul(self, layer_id, root_input, activation, domain):
        # Make nodes for this activation subgraph
        #
        #       root_input (GateProjMatMul)
        #         /  |
        #   ActFunc  |
        #          \ |
        #           Mul
        act_name = f"/model/layers.{layer_id}/mlp/act_fn/{activation}"
        act_output = f"{act_name}/output_0"
        self.make_node(activation, inputs=[root_input], outputs=[act_output], name=act_name, domain=domain)
        self.make_value_info(act_output, dtype=self.io_dtype, shape=["batch_size", "sequence_length", self.intermediate_size])

        mul_act_name = f"/model/layers.{layer_id}/mlp/act_fn/Mul"
        mul_act_inputs = [root_input, act_output]
        self.make_mul(mul_act_name, mul_act_inputs, dtype=self.io_dtype, shape=["batch_size", "sequence_length", self.intermediate_size])

        return mul_act_name

    def make_gelu(self, layer_id, root_input, activation):
        # Make nodes for this activation subgraph
        #
        #       root_input (Add)
        #           |
        #        GeluAct
        gelu_name = f"/model/layers.{layer_id}/mlp/act_fn/{activation}"
        output = f"{gelu_name}/output_0"
        self.make_node(activation, inputs=[root_input], outputs=[output], name=gelu_name, domain="com.microsoft")
        self.make_value_info(output, self.io_dtype, shape=['batch_size', 'sequence_length', self.intermediate_size])

        return gelu_name

    def make_activation(self, layer_id, root_input):
        if self.activation in {"silu", "swish"}:
            output_name = self.make_activation_with_mul(layer_id, root_input, activation="Sigmoid", domain=None)
        elif self.activation in {"gelu_new", "gelu_fast"}:
            output_name = self.make_gelu(layer_id, root_input, activation="FastGelu")
        elif self.activation in {"gelu"}:
            output_name = self.make_gelu(layer_id, root_input, activation="Gelu")
        else:
            raise NotImplementedError(f"The {self.activation} activation function is not currently supported.")
        return output_name

    def make_lm_head(self, lm_head):
        bias_exists = lm_head.bias is not None
        matmul_name = "/lm_head/MatMul"
        root_input = self.layernorm_attrs["output_0"]
        self.make_matmul(lm_head.weight.detach().numpy(), matmul_name, root_input, logits=not bias_exists)

        if bias_exists:
            add_name = "/lm_head/Add"
            self.make_add_bias(lm_head.bias.detach().numpy(), add_name, root_input=f"{matmul_name}/output_0", logits=True)

    def make_layer(self, layer_id, layer):
        # Each LLM decoder layer is typically defined as:
        # input_layernorm --> attention --> MLP --> output_layernorm
        self.make_layernorm(layer_id, layer.input_layernorm, skip=not self.layernorm_attrs["first_layernorm"], simple=self.layernorm_attrs["simple"], location="input")
        self.make_attention(layer_id, layer.self_attn, self.layernorm_attrs["output_0"])
        self.make_layernorm(layer_id, layer.post_attention_layernorm, skip=True, simple=self.layernorm_attrs["simple"], location="post_attention")
        self.make_mlp(layer_id, layer.mlp, self.layernorm_attrs["output_0"])

        self.layernorm_attrs["first_layernorm"] = False
        if layer_id == self.num_layers - 1:
            # Norm after last decoder layer of model (last layer --> norm)
            self.layernorm_attrs["last_layernorm"] = True

    def make_model(self, input_path):
        # Make inputs and outputs to ONNX model
        self.make_inputs_and_outputs()

        # Make attention mask reformatting nodes
        #
        #           2D attention mask
        #                   |
        #    attention mask reformatting subgraph
        #                   |
        #         4D causal attention mask
        self.make_attention_mask_reformatting()

        # Load weights of original model
        if input_path.endswith(".gguf"):
            # Load GGUF model
            from gguf_model import GGUFModel
            model = GGUFModel.from_pretrained(self.model_type, input_path, self.head_size, self.hidden_size, self.intermediate_size, self.num_attn_heads, self.num_kv_heads, self.vocab_size)
            self.layernorm_attrs["add_offset"] = 0  # add offset already done for GGUF models
        else:
            # Load PyTorch model
            extra_kwargs = {} if os.path.exists(self.model_name_or_path) else {"num_hidden_layers": self.num_layers} if "num_hidden_layers" in self.extra_options else {"cache_dir": self.cache_dir, "use_auth_token": True}
            model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path, **extra_kwargs)
        
        # Loop through model and map each module to ONNX/ORT ops
        self.layer_id = 0
        for module in model.modules():
            if isinstance(module, torch.nn.Embedding) or (hasattr(model, "embedding") and module == model.embedding):
                # Checks (Hugging Face logic) or (GGUF logic)
                # Embedding layer
                print("Reading embedding layer")
                self.make_embedding(module.weight.detach().numpy())
            elif module.__class__.__name__.endswith("DecoderLayer"):
                # Each decoder layer of model
                print(f"Reading decoder layer {self.layer_id}")
                self.make_layer(self.layer_id, module)
                self.layer_id += 1
            elif self.layer_id == self.num_layers and self.has_final_norm(module, model):
                # SkipLayerNorm after last decoder layer (MatMul --> SkipLayerNorm)
                print("Reading final norm")
                self.make_layernorm(self.layer_id, module, skip=True, simple=self.layernorm_attrs["simple"], location="final_norm")
            elif (isinstance(module, torch.nn.Linear) and module.out_features == self.vocab_size) or (hasattr(model, "lm_head") and module == model.lm_head):
                # Checks (Hugging Face logic) or (GGUF logic)
                # Language modeling head (SkipLayerNorm --> logits)
                print("Reading LM head")
                self.make_lm_head(module)

        del model

    def has_final_norm(self, module, model):
        # Hugging Face names
        hf_norm = hasattr(model, "model") and hasattr(model.model, "norm") and module == model.model.norm
        hf_final_layernorm = hasattr(model, "model") and hasattr(model.model, "final_layernorm") and module == model.model.final_layernorm
        # GGUF names
        gguf_final_norm = hasattr(model, "final_norm") and module == model.final_norm
        return hf_norm or hf_final_layernorm or gguf_final_norm

    def make_attention_mask_reformatting(self):
        # Make nodes for the attention mask subgraphs that reformat the
        # 2D attention mask (B, S) to 4D causal attention mask (B, N, S, T)
        #
        #             input_ids       past_key_values.0.key
        #            /         \               |
        #         Shape       Shape          Shape
        #          |            |              |
        #        Gather       Gather         Gather                              
        #       (idx=0)       (idx=1)        (idx=2)
        #          |            |    |\      /                                        
        #          |            |    | \    /
        #          |            |    |   Add                                      attention_mask--------+
        #          |            |    |    |                                       /           \         |
        #      Unsqueeze   Unsqueeze | Unsqueeze                                Shape       Shape       |
        #              \        |    |  /                                         |           |         |
        #               \       |    +-/--------+----------+----------+         Gather      Gather    Unsqueeze
        #                \      |     /         |          |          |        (idx=0)     (idx=1)      |
        #                 \     |    /          |          |          |           |           |         |
        #                  \    |   /       Unsqueeze  Unsqueeze  Unsqueeze   Unsqueeze   Unsqueeze   Unsqueeze
        #                   \   |  /                \ /                    \      |      /              |
        #                    Concat               Concat                    \     |     /               |
        #                   /   |   \                |                       \    |    /                |
        #                  /    |    \               |                        \   |   /                 |
        #                 /     |     \        ConstantOfShape                  Concat                  |
        #                /      |      \           /   \      \                /  |   \                 |
        #               /     Shape     \      Shape   Shape   |              /   |    \                |
        #              /        |        \       |       |     |             /    |     \               |
        #             /         |         \    Slice   Slice   |            /     |      \              |
        #             \   ConstantOfShape  |     |       |     |           /    Shape     \             |
        #              \        |     |    |  Squeeze  Squeeze |          /       |        \            |
        #               \      Mul    |    |     |       |     |         /        |         \           |
        #                \      |     |    | Unsqueeze  Range  |         \  ConstantOfShape  \         /
        #                 \     |     |    |     |       |  |  |          \       |      |    |       /
        #                  \    |     |    |   Concat  Add  |  |           \     Mul     |    |      /
        #                   \   |     |    |     |    /     |  |            \     |      |    |     /
        #                     Equal   |   /    Reshape      |  |             \    |      |    |    /
        #                          \  |  /       |          |  |              \   |      |    |   /
        #                           Where      Less---------+  |               \  |      |    |  /
        #                             |          |             |                 Equal   |   /  /
        #                             |        Where-----------+                      \  |  /  /
        #                             |          |                                     Where  /
        #                             |      Unsqueeze                                   |   /
        #                             |          |                                    Expand
        #                             |      Unsqueeze                                   |
        #                              \    /                                          Cast
        #                              Expand                                            |
        #                                 |                                             Sub
        #                                 |                                            / |
        #                                 |                                           / Cast
        #                                 |                                           |  |
        #                                 |                                           Where
        #                                 |                                             |
        #                                 +----------------------+----------------------+
        #                                                        |
        #                                                       Add
        #                                                        |
        #                                                      Concat

        basename = "/model/attn_mask_reformat"
        input_ids_basename = f"{basename}/input_ids_subgraph"
        past_key_basename = f"{basename}/past_key_subgraph"
        attn_mask_basename = f"{basename}/attn_mask_subgraph"

        # Make past_key_values.0.key subgraph
        past_key_gather_name = self.make_past_key_subgraph(past_key_basename)

        # Make common attention mask subgraphs, one each for input_ids and attention_mask
        shared_unsqueeze_name, end_expand_name = self.make_input_ids_subgraph(input_ids_basename, past_key_gather_name)
        end_where_name = self.make_attention_mask_subgraph(attn_mask_basename, shared_unsqueeze_name)

        end_add_name = f"{basename}/Add"
        end_add_inputs = [f"{end_where_name}/output_0", f"{end_expand_name}/output_0"]
        end_add_shape = ["batch_size", 1, "source_sequence_length", "target_sequence_length"]
        self.make_add(end_add_name, end_add_inputs, dtype=self.io_dtype, shape=end_add_shape) # Shape of mask is now (B, 1, S, T)

        # TODO: replace Concat with Expand for performance gains
        concat_name = f"{basename}/Concat"
        concat_inputs = [f"{end_add_name}/output_0" for _ in range(self.num_attn_heads)]
        concat_shape = ["batch_size", self.num_attn_heads, "source_sequence_length", "target_sequence_length"]
        self.make_concat(concat_name, concat_inputs, dtype=self.io_dtype, shape=concat_shape, axis=1) # Shape of mask is now (B, N, S, T)

        self.mask_attrs["mask_name"] = concat_name

    def make_past_key_subgraph(self, basename):
        shape_name = f"{basename}/Shape"
        self.make_shape(shape_name, "past_key_values.0.key", shape=[4])
        gather_name = f"{basename}/Gather"
        gather_inputs = [f"{shape_name}/output_0", "/model/constants/TensorProto.INT64/0D/2"]
        self.make_gather(gather_name, gather_inputs, axis=0)
        return gather_name

    def make_input_ids_subgraph(self, basename, past_key_gather_name):
        # Make shared nodes between past_key_values.0.key (Gather with idx=2) and input_ids (Gather with idx=1) subgraphs
        #
        #       Gather          Gather
        #       (idx=1)         (idx=2)
        #              \       /
        #               \     /
        #                \   /
        #                 Add
        #                  |
        #              Unsqueeze
        shared_add_name = f"{basename}/Add_1"
        shared_add_inputs = [f"{basename}/Gather_2/output_0", f"{past_key_gather_name}/output_0"]
        self.make_add(shared_add_name, shared_add_inputs, dtype=TensorProto.INT64, shape=[])
        unsqueeze_3_name = f"{basename}/Unsqueeze_3"  # shared unsqueeze for input_ids and past_key_values.0.key
        unsqueeze_3_inputs = [f"{shared_add_name}/output_0", "/model/constants/TensorProto.INT64/1D/0"]
        self.make_unsqueeze(unsqueeze_3_name, unsqueeze_3_inputs, dtype=TensorProto.INT64, shape=[1])

        # Make the additional subgraph for input_ids
        #
        #       Unsqueeze (unsqueeze_4)                   Shape --> Slice --> Squeeze --> Unsqueeze --> Concat
        #      /          \                              /                                                    \
        # Gather (idx=1)   --> Concat --> ConstantOfShape                                                      Reshape --> Less --> Where --> Unsqueeze --> Unsqueeze --> Expand
        #      \          /                              \                                                     |
        #       Unsqueeze (unsqueeze_5)                   Shape --> Slice --> Squeeze --> Range --> Add -------+
        unsqueeze_inputs = [f"{basename}/Gather_2/output_0", "/model/constants/TensorProto.INT64/1D/0"]
        unsqueeze_4_name = f"{basename}/Unsqueeze_4"
        self.make_unsqueeze(unsqueeze_4_name, unsqueeze_inputs, dtype=TensorProto.INT64, shape=[1])
        unsqueeze_5_name = f"{basename}/Unsqueeze_5"
        self.make_unsqueeze(unsqueeze_5_name, unsqueeze_inputs, dtype=TensorProto.INT64, shape=[1])
        unsqueeze_6_name = f"{basename}/Unsqueeze_6"  # shared unsqueeze for input_ids and attention_mask
        self.make_unsqueeze(unsqueeze_6_name, unsqueeze_inputs, dtype=TensorProto.INT64, shape=[1])
        concat_2_name = f"{basename}/Concat_2"
        concat_inputs = [f"{unsqueeze_4_name}/output_0", f"{unsqueeze_5_name}/output_0"]
        self.make_concat(concat_2_name, concat_inputs, dtype=TensorProto.INT64, shape=[2], axis=0)
        constant_shape_name = f"{basename}/ConstantOfShape_2"
        constant_shape_numpy_dtype = self.to_numpy_dtype[self.io_dtype]
        constant_shape_value = numpy_helper.from_array(np.array([np.finfo(constant_shape_numpy_dtype).min], dtype=constant_shape_numpy_dtype))
        self.make_constant_of_shape(constant_shape_name, f"{concat_2_name}/output_0", value=constant_shape_value, dtype=self.io_dtype, shape=['unk', 'unk'])

        # Top path
        shape_4_name = f"{basename}/Shape_4"
        self.make_shape(shape_4_name, f"{constant_shape_name}/output_0", shape=[2])
        slice_1_name = f"{basename}/Slice_1"
        slice_1_inputs = [f"{shape_4_name}/output_0", "/model/constants/TensorProto.INT64/1D/-1", f"/model/constants/TensorProto.INT64/1D/{np.iinfo(np.int64).max}", "/model/constants/TensorProto.INT64/1D/0"]
        self.make_slice(slice_1_name, slice_1_inputs)
        squeeze_1_name = f"{basename}/Squeeze_1"
        squeeze_1_inputs = [f"{slice_1_name}/output_0", "/model/constants/TensorProto.INT64/1D/0"]
        self.make_squeeze(squeeze_1_name, squeeze_1_inputs)
        unsqueeze_7_name = f"{basename}/output_0"
        unsqueeze_7_inputs = [f"{squeeze_1_name}/output_0", "/model/constants/TensorProto.INT64/1D/0"]
        self.make_unsqueeze(unsqueeze_7_name, unsqueeze_7_inputs, dtype=TensorProto.INT64, shape=[1])
        concat_3_name = f"{basename}/Concat_3"
        concat_3_inputs = [f"{unsqueeze_7_name}/output_0", "/model/constants/TensorProto.INT64/1D/1"]
        self.make_concat(concat_3_name, concat_3_inputs, dtype=TensorProto.INT64, shape=[2], axis=0)
        
        # Bottom path
        shape_5_name = f"{basename}/Shape_5"
        self.make_shape(shape_5_name, f"{constant_shape_name}/output_0", shape=[2])
        slice_2_name = f"{basename}/Slice_2"
        slice_2_inputs = [f"{shape_5_name}/output_0", "/model/constants/TensorProto.INT64/1D/-1", f"/model/constants/TensorProto.INT64/1D/{np.iinfo(np.int64).max}", "/model/constants/TensorProto.INT64/1D/0"]
        self.make_slice(slice_2_name, slice_2_inputs)
        squeeze_2_name = f"{basename}/Squeeze_2"
        squeeze_2_inputs = [f"{slice_2_name}/output_0", "/model/constants/TensorProto.INT64/1D/0"]
        self.make_squeeze(squeeze_2_name, squeeze_2_inputs)
        range_name = f"{basename}/Range"
        range_inputs = ["/model/constants/TensorProto.INT64/0D/0", f"{squeeze_2_name}/output_0", "/model/constants/TensorProto.INT64/0D/1"]
        self.make_range(range_name, range_inputs)
        add_2_name = f"{basename}/Add_2"
        add_inputs = [f"{range_name}/output_0", "/model/constants/TensorProto.INT64/0D/1"]
        self.make_add(add_2_name, add_inputs, dtype=TensorProto.INT64, shape=["unk"])

        # Merged path
        reshape_name = f"{basename}/Reshape"
        reshape_inputs = [f"{add_2_name}/output_0", f"{concat_3_name}/output_0"]
        self.make_reshape(reshape_name, reshape_inputs, dtype=TensorProto.INT64, shape=None)
        less_name = f"{basename}/Less"
        less_inputs = [f"{range_name}/output_0", f"{reshape_name}/output_0"]
        self.make_less(less_name, less_inputs)
        where_2_name = f"{basename}/Where_2"
        where_2_inputs = [f"{less_name}/output_0", f"/model/constants/{self.to_str_dtype[self.io_dtype]}/0D/0", f"{constant_shape_name}/output_0"]
        self.make_where(where_2_name, where_2_inputs, dtype=self.io_dtype, shape=None)
        unsqueeze_8_name = f"{basename}/Unsqueeze_8"
        unsqueeze_8_inputs = [f"{where_2_name}/output_0", "/model/constants/TensorProto.INT64/1D/0"]
        self.make_unsqueeze(unsqueeze_8_name, unsqueeze_8_inputs, dtype=self.io_dtype, shape=None)
        unsqueeze_9_name = f"{basename}/Unsqueeze_9"
        unsqueeze_9_inputs = [f"{unsqueeze_8_name}/output_0", "/model/constants/TensorProto.INT64/1D/1"]
        self.make_unsqueeze(unsqueeze_9_name, unsqueeze_9_inputs, dtype=self.io_dtype, shape=None)

        expand_name = self.make_common_mask_reformat_subgraph(basename, root_input="input_ids", unsqueeze_for_concat=unsqueeze_3_name, unsqueeze_for_expand=unsqueeze_9_name, input_ids_subgraph=True)
        return unsqueeze_6_name, expand_name

    def make_attention_mask_subgraph(self, basename, unsqueeze_for_concat):
        # Make the additional subgraph to join Expand:
        # attention_mask --> Unsqueeze --> Unsqueeze --> Expand
        attention_mask_shape = self.input_shapes["attention_mask"]

        unsqueeze_3_name = f"{basename}/Unsqueeze_3"
        unsqueeze_3_inputs = ["attention_mask", "/model/constants/TensorProto.INT64/1D/1"]
        attention_mask_shape.insert(1, 1) # ['batch_size', 'total_sequence_length'] --> ['batch_size', 1, 'total_sequence_length']
        self.make_unsqueeze(unsqueeze_3_name, unsqueeze_3_inputs, dtype=TensorProto.INT64, shape=attention_mask_shape)
        unsqueeze_4_name = f"{basename}/Unsqueeze_4"
        unsqueeze_4_inputs = [f"{unsqueeze_3_name}/output_0", "/model/constants/TensorProto.INT64/1D/2"]
        attention_mask_shape.insert(1, 1) # ['batch_size', 1, 'total_sequence_length'] --> ['batch_size', 1, 1, 'total_sequence_length']
        self.make_unsqueeze(unsqueeze_4_name, unsqueeze_4_inputs, dtype=TensorProto.INT64, shape=attention_mask_shape)

        # Make the main subgraph
        expand_name = self.make_common_mask_reformat_subgraph(basename, root_input="attention_mask", unsqueeze_for_concat=unsqueeze_for_concat, unsqueeze_for_expand=unsqueeze_4_name)

        # Make the additional subgraph after Expand:
        #                      +-----------------+
        #                      |                 |
        # Expand --> Cast --> Sub --> Cast --> Where
        cast_1_name = f"{basename}/Cast_1"
        self.make_cast(cast_1_name, f"{expand_name}/output_0", dtype=self.io_dtype, shape=["unk", "unk", "unk", "unk"])
        sub_name = f"{basename}/Sub"
        sub_inputs = [f"/model/constants/{self.to_str_dtype[self.io_dtype]}/0D/1", f"{cast_1_name}/output_0"]
        self.make_sub(sub_name, sub_inputs, dtype=self.io_dtype, shape=["unk", "unk", "unk", "unk"])
        cast_2_name = f"{basename}/Cast_2"
        self.make_cast(cast_2_name, f"{sub_name}/output_0", dtype=TensorProto.BOOL, shape=["unk", "unk", "unk", "unk"])
        where_2_name = f"{basename}/Where_2"
        where_2_inputs = [f"{cast_2_name}/output_0", f"/model/constants/{self.to_str_dtype[self.io_dtype]}/0D/{np.finfo(self.to_numpy_dtype[self.io_dtype]).min}", f"{sub_name}/output_0"]
        self.make_where(where_2_name, where_2_inputs, dtype=self.io_dtype, shape=["unk", "unk", "unk", "unk"])

        return where_2_name

    def make_common_mask_reformat_subgraph(self, basename, root_input, unsqueeze_for_concat, unsqueeze_for_expand, input_ids_subgraph=False):
        #             root_input
        #            /         \
        #         Shape       Shape
        #          |            |
        #        Gather       Gather                              
        #       (idx=0)       (idx=1)
        #          |            |
        #      Unsqueeze   Unsqueeze   Unsqueeze (unsqueeze_for_concat)
        #              \        |       /
        #               \       |      /
        #                \      |     /
        #                 \     |    /
        #                  \    |   /
        #                   \   |  /
        #                    Concat
        #                   /   |   \
        #                  /    |    \
        #                 /     |     \
        #                /      |      \
        #               /     Shape     \
        #              /        |        \
        #             /         |         \
        #             \   ConstantOfShape  |
        #              \        |     |    |
        #               \      Mul    |    |
        #                \      |     |    |
        #                 \     |     |    |
        #                  \    |     |    |
        #                   \   |     |    |
        #                     Equal   |   /
        #                          \  |  /
        #                           Where
        #                             |   Unsqueeze (unsqueeze_for_expand)
        #                              \    /
        #                              Expand

        shape_1_name = f"{basename}/Shape_1"
        self.make_shape(shape_1_name, root_input, shape=[2])
        shape_2_name = f"{basename}/Shape_2"
        self.make_shape(shape_2_name, root_input, shape=[2])
        gather_1_name = f"{basename}/Gather_1"
        gather_1_inputs = [f"{shape_1_name}/output_0", "/model/constants/TensorProto.INT64/0D/0"]
        self.make_gather(gather_1_name, gather_1_inputs, axis=0)
        gather_2_name = f"{basename}/Gather_2"
        gather_2_inputs = [f"{shape_2_name}/output_0", "/model/constants/TensorProto.INT64/0D/1"]
        self.make_gather(gather_2_name, gather_2_inputs, axis=0)
        unsqueeze_1_name = f"{basename}/Unsqueeze_1"
        unsqueeze_1_inputs = [f"{gather_1_name}/output_0", "/model/constants/TensorProto.INT64/1D/0"]
        self.make_unsqueeze(unsqueeze_1_name, unsqueeze_1_inputs, dtype=TensorProto.INT64, shape=[1])
        unsqueeze_2_name = f"{basename}/Unsqueeze_2"
        unsqueeze_2_inputs = [f"{gather_2_name}/output_0", "/model/constants/TensorProto.INT64/1D/0"]
        self.make_unsqueeze(unsqueeze_2_name, unsqueeze_2_inputs, dtype=TensorProto.INT64, shape=[1])
        
        concat_name = f"{basename}/Concat" if not input_ids_subgraph else f"{basename}/Concat_1"
        concat_first_two_inputs = [f"{unsqueeze_1_name}/output_0", "/model/constants/TensorProto.INT64/1D/1"]
        concat_last_two_inputs = [f"{unsqueeze_for_concat}/output_0", f"{unsqueeze_2_name}/output_0"] if not input_ids_subgraph else [f"{unsqueeze_2_name}/output_0", f"{unsqueeze_for_concat}/output_0"]
        concat_inputs = concat_first_two_inputs + concat_last_two_inputs
        self.make_concat(concat_name, concat_inputs, dtype=TensorProto.INT64, shape=[4], axis=0)
        shape_3_name = f"{basename}/Shape_3"
        self.make_shape(shape_3_name, f"{concat_name}/output_0", shape=[1])
        constant_shape_name = f"{basename}/ConstantOfShape" if not input_ids_subgraph else f"{basename}/ConstantOfShape_1"
        constant_shape_value = numpy_helper.from_array(np.array([1], dtype="int64"))
        self.make_constant_of_shape(constant_shape_name, f"{shape_3_name}/output_0", value=constant_shape_value, dtype=TensorProto.INT64, shape=["unk"])
        mul_name = f"{basename}/Mul"
        mul_inputs = [f"{constant_shape_name}/output_0", "/model/constants/TensorProto.INT64/0D/-1"]
        self.make_mul(mul_name, mul_inputs, dtype=TensorProto.INT64, shape=["unk"])
        equal_name = f"{basename}/Equal"
        equal_inputs = [f"{concat_name}/output_0", f"{mul_name}/output_0"]
        self.make_equal(equal_name, equal_inputs, shape=[4])
        
        where_name = f"{basename}/Where_1"
        where_inputs = [f"{equal_name}/output_0", f"{constant_shape_name}/output_0", f"{concat_name}/output_0"]
        self.make_where(where_name, where_inputs, dtype=TensorProto.INT64, shape=[4])
        expand_name = f"{basename}/Expand"
        expand_inputs = [f"{unsqueeze_for_expand}/output_0", f"{where_name}/output_0"]
        expand_dtype = self.io_dtype if input_ids_subgraph else TensorProto.INT64
        expand_shape = None if input_ids_subgraph else ["unk", "unk", "unk", "unk"]
        self.make_expand(expand_name, expand_inputs, dtype=expand_dtype, shape=expand_shape)

        return expand_name


class LlamaModel(Model):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        self.model_inputs = ["input_ids", "attention_mask", "position_ids"]

    def make_attention_mask_reformatting(self):
        if not self.attention_attrs["use_gqa"]:
            super().make_attention_mask_reformatting()
            return

        # Make nodes for the attention mask subgraph that calculates 
        # attributes about the 2D attention mask to use in GroupQueryAttention
        #
        #                attention_mask
        #               /              \
        #          ReduceSum          Shape
        #              |                |
        #             Sub             Gather
        #              |                |
        #        Cast to int32    Cast to int32
        #              |                |
        #          seqlens_k      total_seq_len
        #            (1D)             (int)

        basename = "/model/attn_mask_reformat"
        attn_mask_basename = f"{basename}/attn_mask_subgraph"

        # Left path
        reduce_sum_name = f"{attn_mask_basename}/ReduceSum"
        reduce_sum_inputs = ["attention_mask", "/model/constants/TensorProto.INT64/1D/1"]
        self.make_reduce_sum(reduce_sum_name, reduce_sum_inputs, dtype=TensorProto.INT64, shape=["batch_size", 1])
        sub_name = f"{attn_mask_basename}/Sub"
        sub_inputs = [f"{reduce_sum_name}/output_0", "/model/constants/TensorProto.INT64/1D/1"]
        self.make_sub(sub_name, sub_inputs, dtype=TensorProto.INT64, shape=["batch_size", 1])
        cast_1_name = f"{attn_mask_basename}/Cast_1"
        self.make_cast(cast_1_name, f"{sub_name}/output_0", dtype=TensorProto.INT32, shape=["batch_size", 1])

        # Right path
        shape_name = f"{attn_mask_basename}/Shape"
        self.make_shape(shape_name, "attention_mask", shape=[2])
        gather_name = f"{attn_mask_basename}/Gather"
        gather_inputs = [f"{shape_name}/output_0", "/model/constants/TensorProto.INT64/0D/1"]
        self.make_gather(gather_name, gather_inputs, axis=0)
        cast_2_name = f"{attn_mask_basename}/Cast_2"
        self.make_cast(cast_2_name, f"{gather_name}/output_0", dtype=TensorProto.INT32, shape=None)

        self.mask_attrs["seqlens_k"] = cast_1_name
        self.mask_attrs["total_seq_len"] = cast_2_name


class MistralModel(LlamaModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        self.position_ids_name = self.make_position_ids_reformatting()

    def make_position_ids_reformatting(self):
        # Make nodes for the position ids reformatting subgraph
        #
        #          input_ids   position_ids
        #              |            |
        #            Shape          |
        #              |            |
        #            Gather         |
        #              |            |
        #          Unsqueeze        |
        #              |            |
        #            Concat         |
        #                  \       /
        #                   Reshape
        #                      |
        #      position_ids input for RotaryEmbedding       

        basename = "/model/pos_ids_reformat"
        shape_name = f"{basename}/Shape"
        self.make_shape(shape_name, "input_ids", shape=[2])
        gather_name = f"{basename}/Gather"
        gather_inputs = [f"{shape_name}/output_0", "/model/constants/TensorProto.INT64/0D/1"]
        self.make_gather(gather_name, gather_inputs, axis=0)
        unsqueeze_name = f"{basename}/Unsqueeze"
        unsqueeze_inputs = [f"{gather_name}/output_0", "/model/constants/TensorProto.INT64/1D/0"]
        self.make_unsqueeze(unsqueeze_name, unsqueeze_inputs, dtype=TensorProto.INT64, shape=[1])
        concat_name = f"{basename}/Concat"
        concat_inputs = ["/model/constants/TensorProto.INT64/1D/-1", f"{unsqueeze_name}/output_0"]
        self.make_concat(concat_name, concat_inputs, dtype=TensorProto.INT64, shape=[2], axis=0)
        reshape_name = f"{basename}/Reshape"
        reshape_inputs = ["position_ids", f"{concat_name}/output_0"]
        self.make_reshape(reshape_name, reshape_inputs, dtype=TensorProto.INT64, shape=None)

        return reshape_name

    def make_attention(self, layer_id, attention, root_input, **kwargs):
        super().make_attention(layer_id, attention, root_input, position_ids=f"{self.position_ids_name}/output_0", **kwargs)


class PhiModel(LlamaModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        # self.input_shapes["position_ids"] = [1]  # Note: This is optional and only needed if you want position_ids to be an int instead of a 2D tensor
        self.layernorm_attrs["simple"] = False
        self.rotemb_attrs["num_heads"] = self.num_attn_heads
        self.rotemb_attrs["rotary_embedding_dim"] = int(self.head_size * self.rotemb_attrs["partial_rotary_factor"])

    def make_rotary_embedding(self, rotemb, name, root_input, **kwargs):
        super().make_rotary_embedding(rotemb, name, root_input, num_heads=self.rotemb_attrs["num_heads"], rotary_embedding_dim=self.rotemb_attrs["rotary_embedding_dim"], **kwargs)
        
    def make_mlp(self, layer_id, mlp, root_input):
        # Make nodes for the MLP subgraph
        #
        #          root_input
        #              |
        #          FC1_MatMul
        #              |
        #           FC1_Add
        #              |
        #           FastGelu
        #              |
        #          FC2_MatMul
        #              |
        #           FC2_Add

        # Make first layer of fully connected nodes (FC1)
        fc1_matmul_name = f"/model/layers.{layer_id}/mlp/fc1/MatMul"
        self.make_matmul(mlp.fc1.weight.detach().numpy(), fc1_matmul_name, root_input)
        fc1_add_name = f"/model/layers.{layer_id}/mlp/fc1/Add"
        self.make_add_bias(mlp.fc1.bias.detach().numpy(), fc1_add_name, root_input=f"{fc1_matmul_name}/output_0")

        # Make activation function
        fast_gelu_name = self.make_activation(layer_id, root_input=f"{fc1_add_name}/output_0")

        # Make second layer of fully connected nodes (FC2)
        fc2_matmul_name = f"/model/layers.{layer_id}/mlp/fc2/MatMul"
        self.make_matmul(mlp.fc2.weight.detach().numpy(), fc2_matmul_name, root_input=f"{fast_gelu_name}/output_0")
        fc2_add_name = f"/model/layers.{layer_id}/mlp/fc2/Add"
        self.make_add_bias(mlp.fc2.bias.detach().numpy(), fc2_add_name, root_input=f"{fc2_matmul_name}/output_0")

        return fc2_add_name

    def make_layer(self, layer_id, layer):
        # Each Phi decoder layer is defined as:
        # input_layernorm --> attention --> MLP --> residual_add
        self.make_layernorm(layer_id, layer.input_layernorm, skip=not self.layernorm_attrs["first_layernorm"], simple=self.layernorm_attrs["simple"], location="input")
        self.make_attention(layer_id, layer.self_attn, self.layernorm_attrs["output_0"])
        fc2_add_name = self.make_mlp(layer_id, layer.mlp, self.layernorm_attrs["output_0"])

        residual_add_name = f"/model/layers.{layer_id}/residual_add/Add"
        residual_add_inputs = [self.layernorm_attrs['skip_input'], f"{fc2_add_name}/output_0"]
        self.make_add(residual_add_name, residual_add_inputs, dtype=self.io_dtype, shape=['batch_size', 'sequence_length', self.hidden_size])

        self.layernorm_attrs["first_layernorm"] = False
        if layer_id == self.num_layers - 1:
            # Norm after last decoder layer of model (last layer --> norm)
            self.layernorm_attrs["last_layernorm"] = True
        
        # Assign output 0 of residual Add as skip input to next SkipLayerNorm
        self.layernorm_attrs["skip_input"] = f"{residual_add_name}/output_0"


class GemmaModel(MistralModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        self.embed_attrs["normalize"] = True
        self.layernorm_attrs["add_offset"] = 1


def parse_extra_options(kv_items):
    """
    Parse key value pairs that are separated by '='
    """
    kv_pairs = {}
    
    if kv_items:
        for kv_str in kv_items:
            kv = kv_str.split('=')
            kv_pairs[kv[0].strip()] = kv[1].strip()

    print(f"Extra options: {kv_pairs}")
    return kv_pairs


def create_model(model_name, input_path, output_dir, precision, execution_provider, cache_dir, **extra_options):
    # Create cache and output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    # Load model config
    extra_kwargs = {} if os.path.isdir(input_path) else {"cache_dir": cache_dir, "use_auth_token": True}
    hf_name = input_path if os.path.isdir(input_path) else model_name
    config = AutoConfig.from_pretrained(hf_name, **extra_kwargs)

    # Set input/output precision of ONNX model
    io_dtype = TensorProto.FLOAT if precision in {"int8", "fp32"} or (precision == "int4" and execution_provider == "cpu") else TensorProto.FLOAT16

    if "config_only" not in extra_options:
        # List architecture options in alphabetical order
        if config.architectures[0] == "GemmaForCausalLM":
            onnx_model = GemmaModel(config, io_dtype, precision, execution_provider, cache_dir, extra_options)
        elif config.architectures[0] == "LlamaForCausalLM":
            onnx_model = LlamaModel(config, io_dtype, precision, execution_provider, cache_dir, extra_options)
        elif config.architectures[0] == "MistralForCausalLM":
            onnx_model = MistralModel(config, io_dtype, precision, execution_provider, cache_dir, extra_options)
        elif config.architectures[0] == "PhiForCausalLM":
            onnx_model = PhiModel(config, io_dtype, precision, execution_provider, cache_dir, extra_options)
        else:
            raise NotImplementedError(f"The {hf_name} model is not currently supported.")

        # Make ONNX model
        onnx_model.make_model(input_path)

        # Save ONNX model
        onnx_model.save_model(output_dir)
    else:
        onnx_model = Model(config, io_dtype, precision, execution_provider, cache_dir, extra_options)

    # Make GenAI config
    onnx_model.make_genai_config(hf_name, extra_kwargs, output_dir)

    # Copy Hugging Face processing files to output folder
    onnx_model.save_processing(hf_name, extra_kwargs, output_dir)


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "-m",
        "--model_name",
        required=False,
        default=None,
        help="Model name in Hugging Face. Do not use if providing an input path to a Hugging Face directory in -i/--input.",
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

    args = parser.parse_args()
    print("Valid precision + execution provider combinations are: FP32 CPU, FP32 CUDA, FP16 CUDA, INT4 CPU, INT4 CUDA")
    return args

if __name__ == '__main__':
    args = get_args()
    extra_options = parse_extra_options(args.extra_options)
    create_model(args.model_name, args.input, args.output, args.precision, args.execution_provider, args.cache_dir, **extra_options)