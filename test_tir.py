from optimum.tir import TirCompiler, TirTarget
from transformers import AutoModelForSequenceClassification, AutoTokenizer

if __name__ == '__main__':
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

    with TirCompiler(model, dynamic_axes=None) as compiled:
        encodings = tokenizer("My name is Morgan and I live in Paris")
        output = compiled(encodings)

        print(output)