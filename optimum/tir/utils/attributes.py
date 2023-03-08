from logging import getLogger
from tempfile import NamedTemporaryFile

import iree.compiler.ir as mlir
from optimum.tir.utils.mlir import import_from_mlir, MLIRValidationException


LOGGER = getLogger("optimum.tir.attributes")


def _walk_operation(operation: mlir.Operation):
    for region in operation.regions:
        for block in region.blocks:
            for child_op in block.operations:

                if isinstance(child_op, mlir.OpView):
                    child_op = child_op.operation

                if child_op.name in op_names:
                    print(child_op)

                # TODO: This is dumb. Both Operation and OpView should expose
                # 'operation' and 'name' attributes.
                if child_op.name in [
                    "linalg.conv_2d_nchw_fchw",
                    "linalg.conv_2d_nhwc_hwcf",
                ]:
                    print(f"Conv: {child_op}")
                #     add_winograd_attribute(child_op, configs)
                # if child_op.name in op_names:
                #     if child_op.name == "linalg.generic":
                #         # This is for generic op that has contractionOpInterface
                #         # which is basically einsum("mk,bkn->bmn")
                #         op_result = str(child_op.results[0])
                #         op_iterator = str(child_op.attributes["iterator_types"])
                #         if len(child_op.operands) != 3:
                #             continue
                #         if "reduction" not in op_iterator:
                #             continue
                #         if "arith.addf" not in op_result or "arith.mulf" not in op_result:
                #             continue
                #         if "arith.subf" in op_result:
                #             continue

                # child_op_shape = get_op_shape(child_op, search_op)
                # if (
                #         child_op_shape in configs.keys()
                #         and configs[child_op_shape]["options"][0] != None
                # ):
                #     add_attributes(
                #         child_op, configs[child_op_shape]["options"][0]
                #     )

                _walk_operation(child_op)


def annotate_module(self, ir: str) -> mlir.Module:
    context, module = import_from_mlir(ir)

    if not module.operation.verify():
        with NamedTemporaryFile(mode="w", encoding="utf-8") as f:
            f.write(str(module))
            LOGGER.warning(
                "Failed to validate annotated MLIR module. "
                f"A reproducing file as been stored at {f.name}."
            )
        raise MLIRValidationException(module)

    return module
