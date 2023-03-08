from typing import Tuple

import iree.compiler.ir as mlir
from iree.compiler.transforms import ireec


class MLIRValidationException(Exception):
    def __init__(self, module: mlir.Module):
        super().__init__("MLIR verification failed.")

        self.module = module


def import_from_mlir(ir_module: str) -> Tuple[mlir.Context, mlir.Module]:
    context = mlir.Context()
    ireec.register_all_dialects(context)
    context.allow_unregistered_dialects = True

    module = mlir.Module.parse(ir_module, context)
    return context, module


