from unittest import TestCase

from tir import TirConfig


class TirConfigTest(TestCase):
    def empty_config(self):
        config = TirConfig()
        self.assertEquals(config.get_compiler_args(), [])

    def debug_config(self):
        config = TirConfig().with_debug_flags()
        args = config.get_compiler_args()

        self.assertIn("--mlir-print-ir-before-all", args)
        self.assertIn("--mlir-elide-elementattrs-if-larger", args)

    def cpu_target_config(self):

        with self.subTest("default - host"):
            config = TirConfig().with_cpu_target()
            args = config.get_compiler_args()
            self.assertIn("--iree-llvm-target-cpu-features=host", args)

        with self.subTest("non default - specific target"):
            config = TirConfig().with_cpu_target("sapphirerapids")
            args = config.get_compiler_args()
            self.assertIn("--iree-llvm-target-cpu-features=sapphirerapids", args)

    def gpu_target_config(self):
        with self.subTest("non default - specific target"):
            config = TirConfig().with_gpu_target("8.0")  # 8.0 == Ampere
            args = config.get_compiler_args()
            self.assertIn("--iree-hal-cuda-llvm-target-arch=8.0", args)

    def args_priority(self):
        config = TirConfig() \
            .register_additional_parameters("--hf-param-low-prio", index=50) \
            .register_additional_parameters("--hf-param-high-prio", 100, index=2) \
            .register_additional_parameters("--hf-param-middle-prio", 1000, index=15)

        args = config.get_compiler_args()
        expected = ["--hf-param-high-prio=100", "--hf-param-middle-prio=1000", "--hf-param-low-prio"]
        self.assertEquals(args, expected)