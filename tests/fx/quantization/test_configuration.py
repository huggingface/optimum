# coding=utf-8
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import itertools
import random
from unittest import TestCase

import torch
from packaging import version

from optimum.fx.quantization import CalibrationMethod, QConfig, QConfigUnit, QuantizationConfig


def _generate_random_values(size):
    return torch.rand((size, 3, 4, 5))


class QConfigUnitTester(TestCase):
    def test_can_create_any_qconfig_unit(self):
        dtype_values = [torch.qint8, torch.quint8]
        symmetric_values = [False, True]
        per_channel_values = [False, True]
        ch_axis_values = [-1, 3]
        quant_min_values = [random.randint(-127, 0)]
        quant_max_values = [random.randint(0, 127)]
        calibration_method_values = [method.value for method in CalibrationMethod]
        average_constant_values = [random.random() for _ in range(2)]
        bins_values = [random.randint(512, 4096) for _ in range(2)]
        upsample_rate_values = [random.random() * 200 for _ in range(2)]

        product = itertools.product(
            dtype_values,
            symmetric_values,
            per_channel_values,
            ch_axis_values,
            quant_min_values,
            quant_max_values,
            calibration_method_values,
            average_constant_values,
            bins_values,
            upsample_rate_values,
        )

        _ = [QConfigUnit(*values) for values in product]

    def _get_observer_class_and_observer_kwargs(self):
        classes_and_kwargs = []
        refs = []

        # MinMax
        classes_and_kwargs.append((torch.ao.quantization.MinMaxObserver, {"qscheme": torch.per_tensor_affine}))
        refs.append(QConfigUnit(dtype=torch.quint8, calibration_method=CalibrationMethod.MinMax))

        classes_and_kwargs.append(
            (
                torch.ao.quantization.MinMaxObserver,
                {"qscheme": torch.per_tensor_symmetric, "dtype": torch.qint8, "quant_min": -100, "quant_max": 112},
            )
        )
        refs.append(
            QConfigUnit(
                dtype=torch.qint8,
                quant_min=-100,
                quant_max=112,
                symmetric=True,
                calibration_method=CalibrationMethod.MinMax,
            )
        )

        # MovingAverageMinMax
        classes_and_kwargs.append(
            (
                torch.ao.quantization.MovingAverageMinMaxObserver,
                {"qscheme": torch.per_tensor_affine, "averaging_constant": 0.5},
            )
        )
        refs.append(
            QConfigUnit(dtype=torch.quint8, averaging_constant=0.5, calibration_method=CalibrationMethod.MovingAverage)
        )

        classes_and_kwargs.append(
            (
                torch.ao.quantization.MovingAverageMinMaxObserver,
                {
                    "qscheme": torch.per_tensor_affine,
                    "dtype": torch.qint8,
                    "quant_min": -5,
                    "quant_max": 15,
                    "averaging_constant": 0.15,
                },
            )
        )
        refs.append(
            QConfigUnit(
                dtype=torch.qint8,
                quant_min=-5,
                quant_max=15,
                averaging_constant=0.15,
                calibration_method=CalibrationMethod.MovingAverage,
            )
        )

        # PerChannelMinMax
        classes_and_kwargs.append(
            (
                torch.ao.quantization.PerChannelMinMaxObserver,
                {"dtype": torch.quint8, "ch_axis": 2, "quant_min": 0, "quant_max": 234},
            )
        )
        refs.append(
            QConfigUnit(
                dtype=torch.quint8,
                per_channel=True,
                ch_axis=2,
                quant_min=0,
                quant_max=234,
                calibration_method=CalibrationMethod.MinMax,
            )
        )

        classes_and_kwargs.append(
            (
                torch.ao.quantization.PerChannelMinMaxObserver,
                {"qscheme": torch.per_channel_symmetric, "dtype": torch.qint8, "quant_min": -20, "quant_max": 20},
            )
        )
        refs.append(
            QConfigUnit(
                dtype=torch.qint8,
                quant_min=-20,
                quant_max=20,
                per_channel=True,
                symmetric=True,
                calibration_method=CalibrationMethod.MinMax,
            )
        )

        # MovingAveragePerChannelMinMax
        classes_and_kwargs.append(
            (
                torch.ao.quantization.MovingAveragePerChannelMinMaxObserver,
                {"dtype": torch.quint8, "ch_axis": 1, "quant_min": 0, "quant_max": 135, "averaging_constant": 0.18},
            )
        )
        refs.append(
            QConfigUnit(
                dtype=torch.quint8,
                per_channel=True,
                ch_axis=1,
                quant_min=0,
                quant_max=135,
                averaging_constant=0.18,
                calibration_method=CalibrationMethod.MovingAverage,
            )
        )

        classes_and_kwargs.append(
            (
                torch.ao.quantization.MovingAveragePerChannelMinMaxObserver,
                {"qscheme": torch.per_channel_symmetric, "dtype": torch.qint8, "quant_min": -10, "quant_max": 10},
            )
        )
        refs.append(
            QConfigUnit(
                dtype=torch.qint8,
                quant_min=-10,
                quant_max=10,
                per_channel=True,
                symmetric=True,
                calibration_method=CalibrationMethod.MovingAverage,
            )
        )

        # Histogram
        classes_and_kwargs.append(
            (
                torch.ao.quantization.HistogramObserver,
                {"dtype": torch.quint8, "quant_min": 0, "quant_max": 64, "upsample_rate": 18, "bins": 512},
            )
        )
        refs.append(
            QConfigUnit(
                dtype=torch.quint8,
                quant_min=0,
                quant_max=64,
                upsample_rate=18,
                bins=512,
                calibration_method=CalibrationMethod.Histogram,
            )
        )

        classes_and_kwargs.append(
            (
                torch.ao.quantization.HistogramObserver,
                {"dtype": torch.qint8, "quant_min": -5, "quant_max": 5, "bins": 1024},
            )
        )
        refs.append(
            QConfigUnit(
                dtype=torch.qint8, quant_min=-5, quant_max=5, bins=1024, calibration_method=CalibrationMethod.Histogram
            )
        )

        return classes_and_kwargs, refs

    def _test_as_observer_and_fake_quantize(self, as_observer):

        samples = _generate_random_values(100)
        classes_and_kwargs, refs = self._get_observer_class_and_observer_kwargs()

        outputs = []

        for t, ref in zip(classes_and_kwargs, refs):
            class_, kwargs = t
            if as_observer:
                qconfig_unit = ref.as_observer(as_factory=False)
                real = class_(**kwargs)
            else:
                qconfig_unit = ref.as_fake_quantize(as_factory=False)
                real = torch.ao.quantization.FakeQuantize(class_, **kwargs)

            for x in samples:
                qconfig_unit(x)
                real(x)

            outputs.append((qconfig_unit.calculate_qparams(), real.calculate_qparams()))

        for idx, output in enumerate(outputs):
            for t1, t2 in zip(*output):
                self.assertTrue(
                    torch.allclose(t1, t2),
                    f"The example do not match between the PyTorch ref and the restored to PyTorch one, max diff: {(t1 - t2).abs().max()}, classes_and_kwargs = {classes_and_kwargs[idx]}, ref = {refs[idx]}",
                )

    def test_as_observer(self):
        self._test_as_observer_and_fake_quantize(True)

    def test_as_fake_quantize(self):
        self._test_as_observer_and_fake_quantize(False)

    def test_from_pytorch_back_to_pytorch(self):
        samples = _generate_random_values(100)
        classes_and_kwargs, _ = self._get_observer_class_and_observer_kwargs()

        def test_fn(real_fn, back_fn):
            outputs = []

            for t in classes_and_kwargs:
                class_, kwargs = t
                real = real_fn(class_, kwargs)
                back = back_fn(real)
                for x in samples:
                    real(x)
                    back(x)

                outputs.append((real.calculate_qparams(), back.calculate_qparams()))

            for idx, output in enumerate(outputs):
                for t1, t2 in zip(*output):
                    self.assertTrue(
                        torch.allclose(t1, t2),
                        f"The example do not match between the PyTorch ref and the restored to PyTorch one, max diff: {(t1 - t2).abs().max()}, classes_and_kwargs = {classes_and_kwargs[idx]}",
                    )

        # From observer back to observer
        test_fn(lambda class_, kwargs: class_(**kwargs), lambda x: QConfigUnit.from_pytorch(x).as_observer(False))

        torch_version = version.parse(torch.__version__)
        if (torch_version.major, torch_version.minor) >= (1, 12):
            # From observer back to fake quantize
            test_fn(
                lambda class_, kwargs: class_(**kwargs), lambda x: QConfigUnit.from_pytorch(x).as_fake_quantize(False)
            )

            # From fake quantize back to fake quantize
            test_fn(
                lambda class_, kwargs: torch.ao.quantization.FakeQuantize(class_, **kwargs),
                lambda x: QConfigUnit.from_pytorch(x).as_fake_quantize(False),
            )

        # From fake quantize back to observer
        test_fn(
            lambda class_, kwargs: torch.ao.quantization.FakeQuantize(class_, **kwargs),
            lambda x: QConfigUnit.from_pytorch(x).as_observer(False),
        )


class QConfigTest(TestCase):
    def _get_examples(self):
        pytorch_qconfig = torch.ao.quantization.QConfig(
            activation=torch.ao.quantization.MovingAverageMinMaxObserver.with_args(
                quant_min=0,
                quant_max=234,
                averaging_constant=0.8,
            ),
            weight=torch.ao.quantization.HistogramObserver.with_args(
                quant_min=0,
                quant_max=189,
                upsample_rate=76,
            ),
        )
        pytorch_qconfig_with_fake_quantize = torch.ao.quantization.QConfig(
            activation=torch.ao.quantization.FakeQuantize.with_args(
                observer=torch.ao.quantization.MovingAverageMinMaxObserver,
                quant_min=0,
                quant_max=234,
                averaging_constant=0.8,
            ),
            weight=torch.ao.quantization.FakeQuantize.with_args(
                observer=torch.ao.quantization.HistogramObserver,
                quant_min=0,
                quant_max=189,
                upsample_rate=76,
            ),
        )
        qconfig = QConfig(
            activation=QConfigUnit(
                quant_min=0,
                quant_max=234,
                averaging_constant=0.8,
                calibration_method="moving_average",
            ),
            weight=QConfigUnit(
                quant_min=0,
                quant_max=189,
                upsample_rate=76,
                calibration_method="histogram",
            ),
        )
        return pytorch_qconfig, pytorch_qconfig_with_fake_quantize, qconfig

    def test_from_pytorch(self):
        pytorch_qconfig, pytorch_qconfig_with_fake_quantize, qconfig = self._get_examples()
        self.assertEqual(QConfig.from_pytorch(pytorch_qconfig), qconfig)
        self.assertEqual(QConfig.from_pytorch(pytorch_qconfig_with_fake_quantize), qconfig)

    def test_to_pytorch(self):
        pytorch_qconfig, pytorch_qconfig_with_fake_quantize, qconfig = self._get_examples()

        samples = _generate_random_values(100)
        outputs = []

        # Static (obsevers)
        static_qconfig = qconfig.to_pytorch("static")

        # Activation
        static_qconfig_activation = static_qconfig.activation()
        pytorch_qconfig_activation = pytorch_qconfig.activation()
        for x in samples:
            static_qconfig_activation(x)
            pytorch_qconfig_activation(x)
        outputs.append((static_qconfig_activation.calculate_qparams(), pytorch_qconfig_activation.calculate_qparams()))

        # Weight
        static_qconfig_weight = static_qconfig.weight()
        pytorch_qconfig_weight = pytorch_qconfig.weight()
        for x in samples:
            static_qconfig_weight(x)
            pytorch_qconfig_weight(x)
        outputs.append((static_qconfig_weight.calculate_qparams(), pytorch_qconfig_weight.calculate_qparams()))

        # QAT (fake quantize)
        static_qconfig = qconfig.to_pytorch("qat")

        # Activation
        static_qconfig_activation = static_qconfig.activation()
        pytorch_qconfig_with_fake_quantize_activation = pytorch_qconfig_with_fake_quantize.activation()
        for x in samples:
            static_qconfig_activation(x)
            pytorch_qconfig_with_fake_quantize_activation(x)
        outputs.append(
            (
                static_qconfig_activation.calculate_qparams(),
                pytorch_qconfig_with_fake_quantize_activation.calculate_qparams(),
            )
        )

        # Weight
        static_qconfig_weight = static_qconfig.weight()
        pytorch_qconfig_with_fake_quantize_weight = pytorch_qconfig_with_fake_quantize.weight()
        for x in samples:
            static_qconfig_weight(x)
            pytorch_qconfig_with_fake_quantize_weight(x)

        outputs.append(
            (static_qconfig_weight.calculate_qparams(), pytorch_qconfig_with_fake_quantize_weight.calculate_qparams())
        )

        for output in outputs:
            for t1, t2 in zip(*output):
                self.assertTrue(
                    torch.allclose(t1, t2),
                    f"The example do not match between the PyTorch ref the QConfig one, max diff: {(t1 - t2).abs().max()}",
                )
