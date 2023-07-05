import copy
from pathlib import Path
from typing import Optional, Union

import onnx
from optimum.litmus import simplify_onnx, utils

TASKS = ["text-generation-with-past"]
BATCH_SIZE = 1
INPUT_LENGTH = 8


def simplify(
    input_model: Union[Path, onnx.ModelProto],
    output_dir: Path,
    batch_size: int,
    input_length: int,
    generation_step: Optional[int] = None,
    task: str = "text-generation-with-past",
) -> onnx.ModelProto:
    if task != "text-generation-with-past":
        raise Exception("Unsupported model task: {task}")
    model_opt = simplify_text_generation_onnx(
        input_model, output_dir, batch_size, input_length, generation_step
    )
    return model_opt


def simplify_text_generation_onnx(
    input_model: Union[Path, onnx.ModelProto],
    output_dir: Path,
    batch_size: int,
    input_len: int,
    gen_step: int,
) -> onnx.ModelProto:
    onnx_model = utils.load_onnx(input_model) if isinstance(input_model, Path) else input_model
    decoder_model = separate_merged_graph(copy.deepcopy(onnx_model), gen_step)

    overwrite_input_shapes = {
        "input_ids": [batch_size, input_len],
        "attention_mask": [batch_size, input_len],
    }
    if gen_step > 0:
        overwrite_input_shapes = {"input_ids": [batch_size, 1], "attention_mask": [batch_size, 1]}
        for vi in onnx_model.graph.input:
            if "past_key_values" not in vi.name:
                continue
            utils.make_dynamic_axis_fixed(vi, "batch_size", batch_size)
            utils.make_dynamic_axis_fixed(vi, "past_sequence_length", input_len + gen_step)
            if utils.has_dynamic_axis(vi):
                raise Exception(
                    f"ModelProto has dim_param in graph.input. Dynamic axis should be fixed: {vi}"
                )
            overwrite_input_shapes.update(
                {vi.name: [dim.dim_value for dim in vi.type.tensor_type.shape.dim]}
            )

    output_model = output_dir / f"decoder_model-opt_gen_step={gen_step}.onnx"
    model_opt = simplify_onnx(decoder_model, output_model, overwrite_input_shapes)
    return model_opt


def separate_merged_graph(merged_model: onnx.ModelProto, gen_step: int) -> onnx.ModelProto:
    subgraphs = utils.get_subgraphs(merged_model)
    if not subgraphs:
        raise Exception("ModelProto does not have subgraph(s).")
    if len(subgraphs) > 2:
        raise Exception("ModelProto has more than 2 subgraphs, which is out of scope.")

    model_initializers = {init.name: init for init in merged_model.graph.initializer}

    subgraph_inputs = [
        vi
        for vi in merged_model.graph.input
        if any(vi.name == vi_name for vi_name in ["input_ids", "attention_mask"])
    ]
    if gen_step == 0:
        subgraph_nodes = subgraphs[0].node
        subgraph_outputs = subgraphs[0].output
    if gen_step > 0:
        subgraph_inputs.extend(
            [vi for vi in merged_model.graph.input if "past_key_values" in vi.name]
        )
        subgraph_nodes = subgraphs[1].node
        subgraph_outputs = subgraphs[1].output

    subgraph_initializers = []
    for node in subgraph_nodes:
        for node_input in node.input:
            if node_input in model_initializers:
                if model_initializers[node_input] in subgraph_initializers:
                    continue
                subgraph_initializers.append(model_initializers[node_input])

    separated_model = utils.make_onnx_model(
        inputs=subgraph_inputs,
        nodes=subgraph_nodes,
        initializers=subgraph_initializers,
        outputs=subgraph_outputs,
        opset_imports=merged_model.opset_import,
        graph_name=f"gen_step={gen_step}",
        producer_name=merged_model.producer_name,
    )
    return separated_model
