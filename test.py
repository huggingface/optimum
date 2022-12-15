import shutil
from pathlib import Path

from transformers import AutoTokenizer, AutoConfig

from huggingface_hub import HfApi
from optimum.onnxruntime import ORTModelForCausalLM, ORTModelForSequenceClassification
from transformers import BertForSequenceClassification, AutoModelForSeq2SeqLM, MBartForConditionalGeneration
from optimum.onnxruntime import ORTModelForSeq2SeqLM


model_ckpt = "hf-internal-testing/tiny-bert"
model_ckpt = "facebook/mbart-large-en-ro"
# model_ckpt = "sshleifer/tiny-mbart"
save_path = Path(f"saved_model/{model_ckpt}")
save_path.mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(model_ckpt, use_auth_token=True)

from transformers.modeling_utils import no_init_weights

config = AutoConfig.from_pretrained(model_ckpt, use_auth_token=True)
with no_init_weights():
    model = MBartForConditionalGeneration(config)

# clean save_path
shutil.rmtree(save_path)

# save to local folder
model.save_pretrained(save_path)

model = ORTModelForSeq2SeqLM.from_pretrained(save_path, from_transformers=True)

# save to local folder
model.save_pretrained(save_path / "onnx")
# onnx.save_model(onnx_model, "path/to/save/the/model.onnx", save_as_external_data=True, all_tensors_to_one_file=True, location="filename", size_threshold=1024, convert_attribute=False)
# import onnx
# onnx_model = onnx.load(str(model.decoders))

# from onnx.external_data_helper import convert_model_to_external_data
# convert_model_to_external_data(onnx_model, all_tensors_to_one_file=False, 
#                     size_threshold=0, convert_attribute=True)
# import os
# os.makedirs(str(model.model_path.parent / "external_data"), exist_ok=True)
# onnx.save_model(onnx_model, str(model.model_path.parent / "external_data" / "model.onnx"), save_as_external_data=True, all_tensors_to_one_file=False, size_threshold=8, convert_attribute=False)


# # shutil.move(save_path / "onnx" / "config.json", save_path / "config.json")
# tokenizer.save_pretrained(save_path)


# open saved model
# model = ORTModelForSequenceClassification.from_pretrained(save_path / "onnx")

# push to hub
# repo_id = "nouamanetazi/bloom-350m-onnx-folder-test"
# api = HfApi()
# api.create_repo(repo_id=repo_id, exist_ok=True)
# api.upload_folder(folder_path=save_path, repo_id=repo_id, path_in_repo=".", repo_type="model")
