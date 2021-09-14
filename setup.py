import re
from setuptools import find_packages, setup

# Ensure we match the version set in optimum/version.py
try:
    filepath = 'src/optimum/version.py'
    with open( filepath ) as version_file:
        __version__, = re.findall('__version__ = "(.*)"', version_file.read())
except Exception as error:
    assert False,  "Error: Could not open '%s' due %s\n" % (filepath, error)


install_requires = [
    "coloredlogs",
    "sympy",
    "transformers>=4.9.2",
    "torch>=1.8"
]

extras = {
    "onnxruntime": ["onnx", "onnxruntime"],
    "intel": [
        "pycocotools",
        "lpot @ git+https://github.com/intel/lpot.git",
        "huggingface_hub",
        "datasets >= 1.2.1",
    ]
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
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="transformers, quantization, pruning, training, ipu",
    url="hf.co/hardware",
    author="HuggingFace Inc. Special Ops Team",
    author_email="hardware@huggingface.co",
    license="Apache",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=install_requires,
    extras_require=extras,
    entry_points={
        "console_scripts": [
            "optimum_export=optimum.onnxruntime.convert:main",
            "optimum_optimize=optimum.onnxruntime.optimize_model:main",
            "optimum_export_optimize=optimum.onnxruntime.convert_and_optimize:main"
        ],
    },
    include_package_data=True,
    zip_safe=False,
)