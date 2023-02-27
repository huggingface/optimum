from unittest import TestCase

from tir import TirTarget


class TirTargetTests(TestCase):
    def test_eq(self):
        # This validates enum are correctly tested for equality because we had issues ...
        # COMPILED_CPU
        self.assertEquals(TirTarget.COMPILED_CPU, TirTarget.COMPILED_CPU)
        self.assertNotEqual(TirTarget.COMPILED_CPU, TirTarget.INTERPRETED_CPU)
        self.assertNotEqual(TirTarget.COMPILED_CPU, TirTarget.COMPILED_GPU)
        self.assertNotEqual(TirTarget.COMPILED_CPU, TirTarget.COMPILED_CUDA)
        self.assertNotEqual(TirTarget.COMPILED_CPU, TirTarget.COMPILED_ROCM)

        # INTERPRETED_CPU
        self.assertEquals(TirTarget.INTERPRETED_CPU, TirTarget.INTERPRETED_CPU)
        self.assertNotEqual(TirTarget.INTERPRETED_CPU, TirTarget.COMPILED_CPU)
        self.assertNotEqual(TirTarget.INTERPRETED_CPU, TirTarget.COMPILED_GPU)
        self.assertNotEqual(TirTarget.INTERPRETED_CPU, TirTarget.COMPILED_CUDA)
        self.assertNotEqual(TirTarget.INTERPRETED_CPU, TirTarget.COMPILED_ROCM)

        # COMPILED_CPU
        self.assertEquals(TirTarget.COMPILED_GPU, TirTarget.COMPILED_GPU)
        self.assertNotEqual(TirTarget.COMPILED_GPU, TirTarget.INTERPRETED_CPU)
        self.assertNotEqual(TirTarget.COMPILED_GPU, TirTarget.COMPILED_CPU)
        self.assertNotEqual(TirTarget.COMPILED_GPU, TirTarget.COMPILED_CUDA)
        self.assertNotEqual(TirTarget.COMPILED_GPU, TirTarget.COMPILED_ROCM)

        # COMPILED_CPU
        self.assertEquals(TirTarget.COMPILED_CUDA, TirTarget.COMPILED_CUDA)
        self.assertNotEqual(TirTarget.COMPILED_CUDA, TirTarget.INTERPRETED_CPU)
        self.assertNotEqual(TirTarget.COMPILED_CUDA, TirTarget.COMPILED_GPU)
        self.assertNotEqual(TirTarget.COMPILED_CUDA, TirTarget.COMPILED_CPU)
        self.assertNotEqual(TirTarget.COMPILED_CUDA, TirTarget.COMPILED_ROCM)

        # COMPILED_CPU
        self.assertEquals(TirTarget.COMPILED_ROCM, TirTarget.COMPILED_ROCM)
        self.assertNotEqual(TirTarget.COMPILED_ROCM, TirTarget.INTERPRETED_CPU)
        self.assertNotEqual(TirTarget.COMPILED_ROCM, TirTarget.COMPILED_GPU)
        self.assertNotEqual(TirTarget.COMPILED_ROCM, TirTarget.COMPILED_CUDA)
        self.assertNotEqual(TirTarget.COMPILED_ROCM, TirTarget.COMPILED_CPU)
