import argparse
import shutil
from pathlib import Path
from typing import Dict, List

import yaml


SUBPACKAGE_TOC_INSERT_INDEX = 2


parser = argparse.ArgumentParser(
    description="Script to combine doc builds from subpackages with base doc build of Optimum. "
    "Assumes all subpackage doc builds are present in the root of the `optimum` repo."
)
parser.add_argument(
    "--subpackages",
    nargs="+",
    help="Subpackages to integrate docs with Optimum. Use hardware partner names like `habana`, `graphcore`, or `intel`",
)
parser.add_argument("--version", type=str, default="main", help="The version of the Optimum docs")


def rename_subpackage_toc(subpackage: str, toc: Dict):
    """
    Extends table of contents sections with the subpackage name as the parent folder.

    Args:
        subpackage (str): subpackage name.
        toc (Dict): table of contents.
    """
    for item in toc:
        for file in item["sections"]:
            if "local" in file:
                file["local"] = f"{subpackage}/" + file["local"]
            else:
                # if "local" is not in file, it means we have a subsection, hence the recursive call
                rename_subpackage_toc(subpackage, [file])


def rename_copy_subpackage_html_paths(subpackage: str, subpackage_path: Path, optimum_path: Path, version: str):
    """
    Copy and rename the files from the given subpackage's documentation to Optimum's documentation.
    In Optimum's documentation, the subpackage files are organized as follows:

    optimum_doc
        ├──────── subpackage_1
        │               ├────── folder_1
        │               │          └───── ...
        │               ├────── ...
        │               └────── folder_x
        │                          └───── ...
        ├──────── ...
        ├──────── subpackage_n
        │               ├────── folder_1
        │               │          └───── ...
        │               ├────── ...
        │               └────── folder_y
        │                          └───── ...
        └──────── usual_optimum_doc

    Args:
        subpackage (str): subpackage name
        subpackage_path (Path): path to the subpackage's documentation
        optimum_path (Path): path to Optimum's documentation
        version (str): Optimum's version
    """
    subpackage_html_paths = list(subpackage_path.rglob("*.html"))
    # The language folder is the 4th folder in the subpackage HTML paths
    language_folder_level = 3

    for html_path in subpackage_html_paths:
        language_folder = html_path.parts[language_folder_level]
        # Build the relative path from the language folder
        relative_path_from_language_folder = Path(*html_path.parts[language_folder_level + 1 :])
        # New path in Optimum's doc
        new_path_in_optimum = Path(
            f"{optimum_path}/optimum/{version}/{language_folder}/{subpackage}/{relative_path_from_language_folder}"
        )
        # Build the parent folders if necessary
        new_path_in_optimum.parent.mkdir(parents=True, exist_ok=True)

        shutil.copyfile(html_path, new_path_in_optimum)


def add_neuron_doc(base_toc: List):
    """
    Extends the table of content with a section about Optimum Neuron.

    Args:
        base_toc (List): table of content for the doc of Optimum.
    """
    # Update optimum table of contents
    base_toc.insert(
        SUBPACKAGE_TOC_INSERT_INDEX,
        {
            "sections": [
                {
                    # Ideally this should directly point at https://huggingface.co/docs/optimum-neuron/index
                    # Current hacky solution is to have a redirection in _redirects.yml
                    "local": "docs/optimum-neuron/index",
                    "title": "🤗 Optimum Neuron",
                }
            ],
            "title": "AWS Trainium/Inferentia",
            "isExpanded": False,
        },
    )


def add_tpu_doc(base_toc: List):
    """
    Extends the table of content with a section about Optimum TPU.

    Args:
        base_toc (List): table of content for the doc of Optimum.
    """
    # Update optimum table of contents
    base_toc.insert(
        SUBPACKAGE_TOC_INSERT_INDEX,
        {
            "sections": [
                {
                    # Ideally this should directly point at https://huggingface.co/docs/optimum-tpu/index
                    # Current hacky solution is to have a redirection in _redirects.yml
                    "local": "docs/optimum-tpu/index",
                    "title": "🤗 Optimum-TPU",
                }
            ],
            "title": "Google TPUs",
            "isExpanded": False,
        },
    )


def add_executorch_doc(base_toc: List):
    """
    Extends the table of content with a section about Optimum ExecuTorch.

    Args:
        base_toc (List): table of content for the doc of Optimum.
    """
    # Update optimum table of contents
    base_toc.insert(
        SUBPACKAGE_TOC_INSERT_INDEX,
        {
            "sections": [
                {
                    # Ideally this should directly point at https://huggingface.co/docs/optimum-executorch/index
                    # Current hacky solution is to have a redirection in _redirects.yml
                    "local": "docs/optimum-executorch/index",
                    "title": "🤗 Optimum ExecuTorch",
                }
            ],
            "title": "ExecuTorch",
            "isExpanded": False,
        },
    )


def add_furiosa_doc(base_toc: List):
    """
    Extends the table of content with a section about Optimum Furiosa.

    Args:
        base_toc (List): table of content for the doc of Optimum.
    """
    # Update optimum table of contents
    base_toc.insert(
        SUBPACKAGE_TOC_INSERT_INDEX,
        {
            "sections": [
                {
                    # Ideally this should directly point at https://huggingface.co/docs/optimum-furiosa/index
                    # Current hacky solution is to have a redirection in _redirects.yml
                    "local": "docs/optimum-furiosa/index",
                    "title": "🤗 Optimum Furiosa",
                }
            ],
            "title": "Furiosa",
            "isExpanded": False,
        },
    )


def add_onnx_doc(base_toc: List):
    """
    Extends the table of content with a section about Optimum ONNX.
    Args:
        base_toc (List): table of content for the doc of Optimum.
    """
    # Update optimum table of contents
    base_toc.insert(
        SUBPACKAGE_TOC_INSERT_INDEX,
        {
            "sections": [
                {
                    # Ideally this should directly point at https://huggingface.co/docs/optimum-onnx/index
                    # Current hacky solution is to have a redirection in _redirects.yml
                    "local": "docs/optimum-onnx/index",
                    "title": "🤗 Optimum ONNX",
                }
            ],
            "title": "ONNX",
            "isExpanded": False,
        },
    )


def main():
    args = parser.parse_args()
    optimum_path = Path("optimum-doc-build")
    # Load optimum table of contents
    base_toc_path = next(optimum_path.rglob("_toctree.yml"))
    with open(base_toc_path, "r") as f:
        base_toc = yaml.safe_load(f)

    # Copy and rename all files from subpackages' docs to Optimum doc
    for subpackage in args.subpackages[::-1]:
        if subpackage == "neuron":
            # Neuron has its own doc so it is managed differently
            add_neuron_doc(base_toc)
        elif subpackage == "tpu":
            # Optimum TPU has its own doc so it is managed differently
            add_tpu_doc(base_toc)
        elif subpackage == "nvidia":
            # At the moment, Optimum Nvidia's doc is the README of the GitHub repo
            # It is linked to in optimum/docs/source/nvidia_overview.mdx
            continue
        elif subpackage == "executorch":
            # Optimum ExecuTorch has its own doc so it is managed differently
            add_executorch_doc(base_toc)
        elif subpackage == "onnx":
            add_onnx_doc(base_toc)
        elif subpackage == "furiosa":
            # TODO: add furiosa doc when available
            # add_furiosa_doc(base_toc)
            continue
        else:
            subpackage_path = Path(f"{subpackage}-doc-build")

            # Copy all HTML files from subpackage into optimum
            rename_copy_subpackage_html_paths(
                subpackage,
                subpackage_path,
                optimum_path,
                args.version,
            )

            # Load subpackage table of contents
            subpackage_toc_path = next(subpackage_path.rglob("_toctree.yml"))
            with open(subpackage_toc_path, "r") as f:
                subpackage_toc = yaml.safe_load(f)
            # Extend table of contents sections with the subpackage name as the parent folder
            rename_subpackage_toc(subpackage, subpackage_toc)
            # Just keep the name of the partner in the TOC title
            if subpackage == "amd":
                subpackage_toc[0]["title"] = subpackage_toc[0]["title"].split("Optimum-")[-1]
            else:
                subpackage_toc[0]["title"] = subpackage_toc[0]["title"].split("Optimum ")[-1]
            if subpackage != "graphcore":
                # Update optimum table of contents
                base_toc.insert(SUBPACKAGE_TOC_INSERT_INDEX, subpackage_toc[0])

    # Write final table of contents
    with open(base_toc_path, "w") as f:
        yaml.safe_dump(base_toc, f, allow_unicode=True)


if __name__ == "__main__":
    main()
