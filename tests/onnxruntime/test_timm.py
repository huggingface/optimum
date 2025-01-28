import gc

import onnxruntime
import pytest
import requests
import timm
import torch
from parameterized import parameterized
from PIL import Image
from testing_utils import ORTModelTestMixin
from transformers import PretrainedConfig
from transformers.testing_utils import slow

from optimum.onnxruntime import ORTModelForImageClassification


class ORTModelForImageClassificationIntegrationTest(ORTModelTestMixin):
    TIMM_SUPPORTED_MODELS = [
        "timm/inception_v3.tf_adv_in1k",
        # This is too much for the CI
        # "timm/tf_efficientnet_b0.in1k",
        # "timm/cspdarknet53.ra_in1k",
        # "timm/cspresnet50.ra_in1k",
        # "timm/cspresnext50.ra_in1k",
        # "timm/densenet121.ra_in1k",
        # "timm/dla102.in1k",
        # "timm/dpn107.mx_in1k",
        # "timm/ecaresnet101d.miil_in1k",
        # "timm/efficientnet_b1_pruned.in1k",
        # "timm/inception_resnet_v2.tf_ens_adv_in1k",
        # "timm/fbnetc_100.rmsp_in1k",
        # "timm/xception41.tf_in1k",
        # "timm/senet154.gluon_in1k",
        # "timm/seresnext26d_32x4d.bt_in1k",
        # "timm/hrnet_w18.ms_aug_in1k",
        # "timm/inception_v3.gluon_in1k",
        # "timm/inception_v4.tf_in1k",
        # "timm/mixnet_s.ft_in1k",
        # "timm/mnasnet_100.rmsp_in1k",
        # "timm/mobilenetv2_100.ra_in1k",
        # "timm/mobilenetv3_small_050.lamb_in1k",
        # "timm/nasnetalarge.tf_in1k",
        # "timm/tf_efficientnet_b0.ns_jft_in1k",
        # "timm/pnasnet5large.tf_in1k",
        # "timm/regnetx_002.pycls_in1k",
        # "timm/regnety_002.pycls_in1k",
        # "timm/res2net101_26w_4s.in1k",
        # "timm/res2next50.in1k",
        # "timm/resnest101e.in1k",
        # "timm/spnasnet_100.rmsp_in1k",
        # "timm/resnet18.fb_swsl_ig1b_ft_in1k",
        # "timm/tresnet_l.miil_in1k",
    ]

    @parameterized.expand(TIMM_SUPPORTED_MODELS)
    @pytest.mark.run_slow
    @slow
    def test_compare_to_timm(self, model_id):
        onnx_model = ORTModelForImageClassification.from_pretrained(model_id, export=True)
        self.assertIsInstance(onnx_model.model, onnxruntime.InferenceSession)
        self.assertIsInstance(onnx_model.config, PretrainedConfig)

        timm_model = timm.create_model(model_id, pretrained=True)
        timm_model = timm_model.eval()

        # get model specific transforms (normalization, resize)
        data_config = timm.data.resolve_model_data_config(timm_model)
        transforms = timm.data.create_transform(**data_config, is_training=False)

        url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
        image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        inputs = transforms(image).unsqueeze(0)

        with torch.no_grad():
            timm_outputs = timm_model(inputs)

        for input_type in ["pt", "np"]:
            if input_type == "np":
                inputs = inputs.cpu().detach().numpy()

            onnx_outputs = onnx_model(inputs)

            self.assertIn("logits", onnx_outputs)
            self.assertIsInstance(onnx_outputs.logits, self.TENSOR_ALIAS_TO_TYPE[input_type])

            # compare tensor outputs
            torch.testing.assert_close(torch.Tensor(onnx_outputs.logits), timm_outputs, atol=self.ATOL, rtol=self.RTOL)

        gc.collect()
