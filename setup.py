import re

from setuptools import find_namespace_packages, setup


# Ensure we match the version set in src/optimum/version.py
try:
    filepath = "optimum/version.py"
    with open(filepath) as version_file:
        (__version__,) = re.findall('__version__ = "(.*)"', version_file.read())
except Exception as error:
    assert False, "Error: Could not open '%s' due %s\n" % (filepath, error)


REQUIRED_PKGS = [
    "coloredlogs",
    "sympy",
    "transformers[sentencepiece]>=4.29",
    "torch>=1.11",
    "packaging",
    "numpy",
    "huggingface_hub>=0.8.0",
    "datasets",
]

# TODO: unpin pytest once https://github.com/huggingface/transformers/pull/29154 is merged & released
# pytest>=8.0.0 also fails with the transformers version pinned for exporters-tf
TESTS_REQUIRE = [
    "accelerate",
    "pytest<=8.0.0",
    "requests",
    "parameterized",
    "pytest-xdist",
    "Pillow",
    "sacremoses",
    "torchvision",
    "diffusers>=0.17.0",
    "torchaudio",
    "einops",
    "invisible-watermark",
    "timm",
    "scikit-learn",
    "rjieba",
]

QUALITY_REQUIRE = ["black~=23.1", "ruff==0.1.5"]

BENCHMARK_REQUIRE = ["optuna", "tqdm", "scikit-learn", "seqeval", "torchvision", "evaluate>=0.2.0"]

EXTRAS_REQUIRE = {
    "onnxruntime": [
        "onnx",
        "onnxruntime>=1.11.0",
        "datasets>=1.2.1",
        "evaluate",
        "protobuf>=3.20.1",
        "transformers<4.46.0",
    ],
    "onnxruntime-gpu": [
        "onnx",
        "onnxruntime-gpu>=1.11.0",
        "datasets>=1.2.1",
        "evaluate",
        "protobuf>=3.20.1",
        "accelerate",  # ORTTrainer requires it.
        "transformers<4.46.0",
    ],
    "exporters": ["onnx", "onnxruntime", "timm", "transformers<4.46.0"],
    "exporters-gpu": ["onnx", "onnxruntime-gpu", "timm", "transformers<4.46.0"],
    "exporters-tf": [
        "tensorflow>=2.4,<=2.12.1",
        "tf2onnx",
        "onnx",
        "onnxruntime",
        "timm",
        "h5py",
        "numpy<1.24.0",
        "datasets<=2.16",
        "transformers[sentencepiece]>=4.26,<4.38",
    ],
    "diffusers": ["diffusers"],
    "intel": "optimum-intel>=1.18.0",
    "openvino": "optimum-intel[openvino]>=1.18.0",
    "nncf": "optimum-intel[nncf]>=1.18.0",
    "neural-compressor": "optimum-intel[neural-compressor]>=1.18.0",
    "ipex": "optimum-intel[ipex]>=1.18.0",
    "habana": ["optimum-habana", "transformers>=4.43.0,<4.44.0"],
    "neuron": ["optimum-neuron[neuron]>=0.0.20", "transformers>=4.36.2,<4.42.0"],
    "neuronx": ["optimum-neuron[neuronx]>=0.0.20", "transformers>=4.36.2,<4.42.0"],
    "graphcore": "optimum-graphcore",
    "furiosa": "optimum-furiosa",
    "amd": "optimum-amd",
    "quanto": ["optimum-quanto>=0.2.4"],
    "dev": TESTS_REQUIRE + QUALITY_REQUIRE,
    "tests": TESTS_REQUIRE,
    "quality": QUALITY_REQUIRE,
    "benchmark": BENCHMARK_REQUIRE,
    "doc-build": ["accelerate"],
}

setup(
    name="optimum",
    version=__version__,
    description="Optimum Library is an extension of the Hugging Face Transformers library, providing a framework to "
    "integrate third-party libraries from Hardware Partners and interface with their specific "
    "functionality.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="transformers, quantization, pruning, optimization, training, inference, onnx, onnx runtime, intel, "
    "habana, graphcore, neural compressor, ipu, hpu",
    url="https://github.com/huggingface/optimum",
    author="HuggingFace Inc. Special Ops Team",
    author_email="hardware@huggingface.co",
    license="Apache",
    packages=find_namespace_packages(include=["optimum*"]),
    install_requires=REQUIRED_PKGS,
    extras_require=EXTRAS_REQUIRE,
    python_requires=">=3.7.0",
    include_package_data=True,
    zip_safe=False,
    entry_points={"console_scripts": ["optimum-cli=optimum.commands.optimum_cli:main"]},
)
