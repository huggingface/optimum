from .import_utils import DummyObject, requires_backends


class BarkSelfAttention(metaclass=DummyObject):
    _backends = ["transformers_431"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["transformers_431"])
