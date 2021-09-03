class LpotQuantizationMode:

    DYNAMIC = "post_training_dynamic_quant"
    STATIC = "post_training_static_quant"
    AWARE_TRAINING = "quant_aware_training"


class LpotQuantizer:

    def __init__(self, config_path, model, eval_func, train_func=None, calib_dataloader=None):
        from lpot.conf.config import Conf
        from lpot.conf.dotdict import deep_get

        self.config_path = config_path
        self.config = Conf(config_path)
        self.model = model
        self.approach = deep_get(self.config.usr_cfg, "quantization.approach")
        self._eval_func = eval_func
        self._train_func = train_func
        self._calib_dataloader = calib_dataloader

    @property
    def eval_func(self):
        return self._eval_func

    @property
    def train_func(self):
        return self._train_func

    @property
    def calib_dataloader(self):
        return self._calib_dataloader

    @train_func.setter
    def train_func(self, func):
        self._train_func = func

    @calib_dataloader.setter
    def calib_dataloader(self, dataloader):
        self._calib_dataloader = dataloader

    def init_quantizer(self):
        from lpot.experimental import Quantization, common

        quantizer = Quantization(self.config_path)
        quantizer.model = common.Model(self.model)
        quantizer.eval_func = self._eval_func
        return quantizer

    @staticmethod
    def adaptor_calib():
        from lpot.adaptor.pytorch import PyTorchAdaptor
        from .utils import model_calibration
        PyTorchAdaptor.model_calibration = model_calibration

    def fit_dynamic(self):
        quantizer = self.init_quantizer()
        model = quantizer()
        return model.model

    def fit_static(self):
        self.adaptor_calib()
        quantizer = self.init_quantizer()

        if self._calib_dataloader is None:
            raise ValueError("calib_dataloader must be provided for post-training quantization.")

        quantizer.calib_dataloader = self._calib_dataloader

        model = quantizer()
        return model.model

    def fit_aware_training(self):
        self.adaptor_calib()
        quantizer = self.init_quantizer()

        if self._train_func is None:
            raise ValueError("train_func must be provided for quantization aware training .")

        quantizer.q_func = self._train_func

        model = quantizer()
        return model.model


SUPPORTED_QUANT_APPROACH = {
    LpotQuantizationMode.STATIC,
    LpotQuantizationMode.DYNAMIC,
    LpotQuantizationMode.AWARE_TRAINING
}


def quantization_approach(config):
    """
    Extract quantization approach from YAML configuration file.
    Args:
        config: YAML configuration file used to control the tuning behavior.
    Returns:
        approach: Name of the quantization approach.
    """
    from lpot.conf.config import Conf
    from lpot.conf.dotdict import deep_get

    conf = Conf(config)
    approach = deep_get(conf.usr_cfg, "quantization.approach")

    if approach not in SUPPORTED_QUANT_APPROACH:
        raise ValueError("Unknown quantization approach. Supported approach are " + ", ".join(SUPPORTED_QUANT_APPROACH))

    return approach


def quantize_dynamic(model, config,  eval_func):
    """
    Apply LPOT dynamic quantization.
    Args:
        model: FP32 model specified for low precision tuning.
        config: YAML configuration file used to control the tuning behavior.
        eval_func: Evaluation function provided by user.
    Returns:
        model: Quantized model.
    """
    from lpot.experimental import Quantization, common

    quantizer = Quantization(config)
    quantizer.model = common.Model(model)

    quantizer.eval_func = eval_func

    model = quantizer()

    return model.model


def quantize_static(model, config, eval_func, calib_dataloader):
    """
    Apply LPOT post-training quantization.
    Args:
        model: FP32 model specified for low precision tuning.
        config: YAML configuration file used to control the tuning behavior.
        eval_func: Evaluation function provided by user.
        calib_dataloader: DataLoader for calibration.
    Returns:
        model: Quantized model.
    """
    from lpot.experimental import Quantization, common
    from lpot.adaptor.pytorch import PyTorchAdaptor
    from .utils import model_calibration

    PyTorchAdaptor.model_calibration = model_calibration

    quantizer = Quantization(config)
    quantizer.model = common.Model(model)

    quantizer.calib_dataloader = calib_dataloader
    quantizer.eval_func = eval_func

    model = quantizer()

    return model.model


def quantize_aware_training(model, config, eval_func, train_func):
    """
    Apply LPOT quantization aware training.
    Args:
        model: FP32 model specified for low precision tuning.
        config: YAML configuration file used to control the entire tuning behavior.
        eval_func: Evaluation function provided by user.
        train_func: Training function provided by user.
    Returns:
        model: Quantized model.
    """
    from lpot.experimental import Quantization, common
    from lpot.adaptor.pytorch import PyTorchAdaptor
    from .utils import model_calibration

    PyTorchAdaptor.model_calibration = model_calibration

    quantizer = Quantization(config)
    quantizer.model = common.Model(model)

    quantizer.q_func = train_func
    quantizer.eval_func = eval_func

    model = quantizer()

    return model.model

