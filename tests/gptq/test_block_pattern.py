# coding=utf-8
from torch import nn

from optimum.gptq.utils import get_block_name_with_pattern


def test_short_pattern_does_not_match_unrelated_module_prefix():
    class HeadOnly(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = nn.Linear(4, 4)

    try:
        get_block_name_with_pattern(HeadOnly())
    except ValueError as error:
        assert "Block pattern could not be match" in str(error)
    else:
        raise AssertionError("module named head was treated as a transformer.h block")


def test_transformer_h_blocks_still_match():
    class GptLike(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = nn.Module()
            self.transformer.h = nn.ModuleList([nn.Linear(4, 4)])
            self.head = nn.Linear(4, 4)

    assert get_block_name_with_pattern(GptLike()) == "transformer.h"


def test_model_layers_still_match():
    class LlamaLike(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([nn.Linear(4, 4)])

    assert get_block_name_with_pattern(LlamaLike()) == "model.layers"

