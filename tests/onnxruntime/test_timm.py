import gc

import onnxruntime
import requests
import timm
import torch
from parameterized import parameterized
from PIL import Image
from testing_utils import ORTModelTestMixin
from transformers import PretrainedConfig

from optimum.onnxruntime import ORTModelForImageClassification


class ORTModelForImageClassificationIntegrationTest(ORTModelTestMixin):
    TIMM_SUPPORTED_MODELS = ["timm/inception_v3.tf_adv_in1k"]  # only one is required for testing

    @parameterized.expand(TIMM_SUPPORTED_MODELS)
    def test_compare_to_timm(self, model_id):
        onnx_model = ORTModelForImageClassification.from_pretrained(model_id)
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
