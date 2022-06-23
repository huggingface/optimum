import subprocess
import unittest

class TestTextClassification(unittest.TestCase):
    def test_eval_transformers_examples(self):
        cpu_info = subprocess.check_output(["sysctl -a | grep machdep.cpu"]).decode("utf-8")
        print(cpu_info)