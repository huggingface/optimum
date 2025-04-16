#!/bin/bash
# we use 4.48 for compatibility with bettertransformer
pip install .[tests] transformers==4.48.* optuna
python tests/utils/prepare_for_doc_test.py optimum docs
pytest --verbose -s --doctest-modules $(cat tests/utils/documentation_tests.txt) --doctest-continue-on-failure --doctest-glob='*.mdx'
python tests/utils/prepare_for_doc_test.py optimum docs --remove_new_line