import logging
import os
import unittest
import pytest
import onnxruntime_genai as og


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TDD: ")


class TestCase(unittest.TestCase):

    @unittest.skip("There is no issue with this module")
    def test_genai(self):
        logger.info("Start to Load the model")
        model_path = os.path.abspath("./models/phi2")
        model = og.Model(model_path)
        logger.info("Finish to Load the model")
        tokenizer = og.Tokenizer(model)

        prompt = '''def print_prime(n):
            """
            Print all primes between 1 and n
            """'''

        tokens = tokenizer.encode(prompt)
        params = og.GeneratorParams(model)
        params.set_search_options({"max_length":200})
        params.input_ids = tokens

        output_tokens = model.generate(params)
        text = tokenizer.decode(output_tokens)
        print("Output:")
        print(text)


    def test_optimum(self):
        pass


    def test_build_model(self):
        
        from .modeling_genai import OGModel
        # create the model
        og_model = OGModel
        model_path = og_model.build_model(model_name="microsoft/phi-2",
                         input_path="",
                         output_dir="./test/microsoft/phi2",
                         precision="int4",
                         execution_provider="cpu",
                         cache_dir=os.path.join(".", "cache"),
                         extra_options="",
        )
        logging.info("Successfully tested the build_model!")
        return model_path


    def test_load_model(self):

        from .modeling_genai import OGModel
        # create the model
        og_model = OGModel

        path = self.test_build_model()
        logging.info(f"{path} the path of the model.")
        
        # load model
        model = og_model.load_model(path=path)
        logging.info("Successfully load the model")
        return model


    def test_text_generation(self):

        from .modeling_genai import OGModelForTextGeneration

        model = self.test_load_model()
        # text generation
        genai = OGModelForTextGeneration(
            model=model,
            **{
            "max_length": 200, 
             "top_p": 0.9,
             "top_k": 50,
             "temperature": 1.0,
             "repetition_penalty": 1.0
            },
        )

        genai.forward(prompt='''def print_prime(n): \n""" Print all primes between 1 and n \n"""''')
        logging.info("Successfully tested the text generation module")

