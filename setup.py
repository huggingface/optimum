from setuptools import find_packages, setup

setup(
    name="optimus",
    version="0.1",
    description="optimus is a python package for optimizing and exporting machine learning models to ONNX.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="",
    url="",
    author="",
    author_email="",
    license="MIT",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=[],
    test_suite="nose.collector",
    tests_require=["nose", "nose-cover3"],
    entry_points={
        "console_scripts": [
            "optimus_convert=optimus.convert_to_onnx:main",
            "optimus_optimize=optimus.optimize_model:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
