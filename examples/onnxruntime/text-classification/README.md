<!---
Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Text classification 

## GLUE tasks

### Fine-tuning

By runing the script [`run_glue.py`](https://github.com/huggingface/optimum/blob/main/examples/onnxruntime/text-classification/run_glue.py),
we will be able to leverage [`ONNX Runtime`](https://github.com/microsoft/onnxruntime) accelerator to fine-tune the models from 
[HuggingFace hub](https://huggingface.co/models) for sequence classification on the [GLUE benchmark](https://gluebenchmark.com/).

To run fine-tuning a model on one of the tasks, here is an example of `bert-base-uncased` model on the sst-2 task:

```bash
python run_glue.py \
    --model_name_or_path bert-base-uncased \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --output_dir /tmp/$TASK_NAME/
```
GLUE has 9 different tasks: cola, sst2, mrpc, stsb, qqp, mnli, qnli, rte, wnli. Here are the results and performances 
of fine-tuning with `bert-base-uncased` by ONNX Runtime compared with PyTorch.


| Task  | Metric                       | Result      | Training time |
|-------|------------------------------|-------------|---------------|
| CoLA  | Matthews corr                | 56.53       | 3:17          |
| SST-2 | Accuracy                     | 92.32       | 26:06         |
| MRPC  | F1/Accuracy                  | 88.85/84.07 | 2:21          |
| STS-B | Pearson/Spearman corr.       | 88.64/88.48 | 2:13          |
| QQP   | Accuracy/F1                  | 90.71/87.49 | 2:22:26       |
| MNLI  | Matched acc./Mismatched acc. | 83.91/84.10 | 2:35:23       |
| QNLI  | Accuracy                     | 90.66       | 40:57         |
| RTE   | Accuracy                     | 65.70       | 57            |
| WNLI  | Accuracy                     | 56.34       | 24            |

### Inference

For the time being, the inference of ONNX runtime trained models can only be done within PyTorch(not with ONNX Runtime). But if you want
to process only the inference with ONNX Runtime, you can export your fine-tuned PyTorch model to ONNX by `torch.export`, and then use ONNX 
Runtime just for inference. Here is an example on the previous task:

```bash
python run_glue.py \
    --model_name_or_path bert-base-uncased \
    --task_name $TASK_NAME \
    --do_eval \
    --output_dir /tmp/$TASK_NAME/
```
