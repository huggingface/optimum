from .import_utils import DummyObject, requires_backends


class LlamaAttention(metaclass=DummyObject):
    _backends = ["transformers_431"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["transformers_431"])


class BarkSelfAttention(metaclass=DummyObject):
    _backends = ["transformers_431"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["transformers_431"])


class FalconAttention(metaclass=DummyObject):
    _backends = ["transformers_434"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["transformers_434"])


def _llama_prepare_decoder_attention_mask(*args, **kwargs):
    requires_backends(_llama_prepare_decoder_attention_mask, ["transformers_431"])


# The guard is already on LlamaAttention so none of those function will actually ever be called.
def _expand_mask(*arg, **kwargs):
    pass


def _make_causal_mask(*arg, **kwargs):
    pass


def apply_rotary_pos_emb(*arg, **kwargs):
    pass


def repeat_kv(*arg, **kwargs):
    pass
