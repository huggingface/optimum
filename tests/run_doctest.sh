#!/bin/bash
pip install accelerate
pip install .[benchmark]
touch optimum/__init__.py
python tests/utils/prepare_for_doc_test.py optimum docs
pytest --verbose -s --doctest-modules $(cat tests/utils/documentation_tests.txt) --doctest-continue-on-failure --doctest-glob='*.mdx'
