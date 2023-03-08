from functools import partial

import numpy as np

import iree.runtime as ireert


class TirCompiledModule:
    __slots__ = ("_context", "_config")

    def __init__(self, context: ireert.SystemContext, config: ireert.Config):
        self._context = context
        self._config = config

        # functions = {
        #     function: module[function]
        #     for name, module in context.modules.items() if name != "hal"
        #     for function in module.vm_module.function_names if not function.startswith("__")
        # }
        #
        # self.

    def __call__(self, *inputs, function: str = "forward"):
        """Runs a .vmfb file given inputs and config and returns output."""
        device_inputs = [ireert.asdevicearray(self._config.device, a) for a in inputs]
        result = self._context.modules.module[function](*device_inputs)

        result_tensors = []
        if isinstance(result, tuple):
            # if send_to_host:
            for val in result:
                result_tensors.append(np.asarray(val, val.dtype))
            else:
                for val in result:
                    result_tensors.append(val)
            return result_tensors

        elif isinstance(result, dict):
            data = list(result.items())
            # if send_to_host:
            res = np.array(data, dtype=object)
            return np.copy(res)
            # return data
        else:
            # if send_to_host and result is not None:
            return result.to_host()
            # return result

    def __getattr__(self, item):
        return partial(self.__call__(function=item))

