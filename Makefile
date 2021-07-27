.PHONY:	style test

# Run code quality checks
style_check:
	black --check .
	isort --check .

style:
	black .
	isort .

# Run tests for the library
test:
	python -m pytest tests

build_dist_install_tools:
	pip install build
	pip install twine

build_dist:
	rm -fr build
	rm -fr dist
	python -m build

pypi_upload: build_dist
	python -m twine upload dist/*
