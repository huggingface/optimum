# Generation-step-wise ONNX graph gen
## 용어
- gen-step: "generation step" 의 줄임말로서, text generation 에서 한 단위의 토큰을 생성하는 각 단계를 의미한다. pre-fill phase 의 경우는 `gen-step=0` 이며, 이후 decode phase 의 경우는 `gen-step=1, 2, 3...` 이다. 참고로 `max(generation_step) = max_new_tokens` 이다. 
  - max_new_tokens: https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig.max_new_tokens
  - generation step 이라는 표현은 huggingface 의 문서에 등장한다. https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.TFGenerationMixin.compute_transition_scores.scores 등.
- decode_model_merged.onnx: huggingface 의 `text-generation-with-past` task 로 모델 export 를 하게되면 생성되는 출력 모델이다. 이 모델은 pre-fill phase 의 decode_model.onnx 와 decode phase 의 decode_model_with_past.onnx 를 sub-graph 형태로 병합한  그래프로 구성된다. 각 sub-graph 는 공통된 초기값을 가지므로 두 그래프를 병합함으로써 모델 바이너리 사이즈가 증가하지는 않는다. 또한, dynamic axis 를 사용하므로 variable input length 및 variable gen-step 의 text-generation 과정을 하나의 단일한 ONNX 모델로 표현 가능하다.
- decoder_model-opt_gen_step={}.onnx
  - opt: `Shape` 오퍼레이터를 포함하여 constant folding 가능한 모든 graph 패턴이 단순화되어 있다. 또한 ONNX opset_version=13 하에서 elimination 과 fusing 도 적용되어 있다. 마지막으로 모든 tensor 의 입출력 값이 fixed shape 으로 추론되어 있는 그래프를 의미한다.
  - `gen_step={}` 은 각 generation step 을 의미한다. "gen-step" 항목의 설명 참조.

## 동작 방식
- huggingface `transformers` 와 `optimum` 을 베이스로 동작한다.
  - `optimum` 에서는 pt -> onnx export 기능을, `transformers` 에서는 onnx export 대상이 되는 모델을 가져온다.
- `optimum.litmus` 경로에는 onnx export, onnx simplification/optimization, compilation 관련 일반적인 기능이 구현되어 있다.
- `optimum.litmus.nlp` 경로에는 task-specific 한 기능이 구현되어 있다. 예를 들어, text-generation task 의 경우 `simplify_text_generation_onnx` 함수에서 task-specific 한 모델 전처리를 거친 뒤 `optimum.litmus.simplify_onnx` 함수를 내부적으로 호출하여 onnx 모델 단순화를 수행하는 식이다.
- "generation-step-wise ONNX graph generation": text-generation 은 pre-fill phase 뿐만 아니라 decode phase 를 포함하고 있기 때문에 단순 한 번의 추론으로 끝나지 않고, 동적으로 recursive 하게 다음 토큰을 생성하는 task 이다. 다시 말해 각 generation step 별로 입출력의 크기가 다르기 때문에 정적 tensor shape 을 추론하기 위하여는 각 step 별로 주어진 입력 크기에 따라 onnx graph 를 생성하는 방식으로 동작해야한다.

## 모델
아래 링크의 google drive "decoder_model-opt_gen_step={}.onnx" ONNX 파일
- GPT2: https://furiosa-ai.atlassian.net/jira/software/c/projects/SWPROJ/boards/2/timeline?selectedIssue=SWPROJ-81
- GPT-Neo: https://furiosa-ai.atlassian.net/jira/software/c/projects/SWPROJ/boards/2/timeline?selectedIssue=SWPROJ-80
- LLaMA: https://furiosa-ai.atlassian.net/jira/software/c/projects/SWPROJ/boards/2/timeline?selectedIssue=SWPROJ-85
- OPT: TBA
