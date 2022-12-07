from typing import Set, List, Dict, Union

import torch
from torch import nn
from transformers import BatchEncoding, PreTrainedModel
from transformers.utils.logging import get_logger
from torch_mlir import ExampleArgs, TensorPlaceholder


LOGGER = get_logger("tir.pytorch")


class _TirOutputWrapper(nn.Module):

    def __init__(self, model: PreTrainedModel):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        output = self.model(*args, **kwargs)

        return output[0]
        # if len(output) == 1:
        #     return output[0]
        # else:
        #     return output


def sanitize_pretrained_model_for_mlir(model: PreTrainedModel):
    model = model.eval()

    if not model.config.torchscript:
        LOGGER.debug("Setting config.torchscript = True")
        model.config.torchscript = True

    if model.config.output_attentions:
        LOGGER.debug("Disabling output attentions.")
        model.config.output_attentions = False

    if model.config.output_hidden_states:
        LOGGER.debug("Disabling output hidden states.")
        model.config.output_hidden_states = False

    return _TirOutputWrapper(model)


def create_attention_mask_from_input_ids(input_ids: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(input_ids)


def convert_encodings_to_example_args(
    encoding: Union[BatchEncoding, Dict],
    ensure_attention_mask: bool = True,
    dynamic_axes: List[int] = None
) -> ExampleArgs:

    if ensure_attention_mask and "attention_mask" not in encoding:
        LOGGER.debug("Creating attention_mask from input_ids.")
        encoding["attention_mask"] = create_attention_mask_from_input_ids(encoding["input_ids"])

    args = ExampleArgs()

    if dynamic_axes is None:
        fw_input_placeholders = list(encoding.values())
    else:
        fw_input_placeholders = list(map(
            lambda tensor: TensorPlaceholder.like(tensor, dynamic_axes=dynamic_axes),
            encoding.values()
        ))

    args.add_method("forward", fw_input_placeholders)
    args.add_method("__call__", fw_input_placeholders)
    return args

