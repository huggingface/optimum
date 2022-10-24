import argparse
import shutil
from pathlib import Path
from typing import Dict

import yaml


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
    Extend table of contents sections with the subpackage name as the parent folder.

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


def main():
    args = parser.parse_args()
    optimum_path = Path("test_optimum_doc")

    # Copy and rename all files from subpackages' docs to Optimum doc
    for subpackage in args.subpackages:
        subpackage_path = Path(f"test_{subpackage}_doc")

        # Copy all HTML files from subpackage into optimum
        rename_copy_subpackage_html_paths(
            subpackage,
            subpackage_path,
            optimum_path,
            args.version,
        )

        # Load optimum table of contents
        base_toc_path = next(optimum_path.rglob("_toctree.yml"))
        with open(base_toc_path, "r") as f:
            base_toc = yaml.safe_load(f)
        # Load subpackage table of contents
        subpackage_toc_path = next(subpackage_path.rglob("_toctree.yml"))
        with open(subpackage_toc_path, "r") as f:
            subpackage_toc = yaml.safe_load(f)
        # Extend table of contents sections with the subpackage name as the parent folder
        rename_subpackage_toc(subpackage, subpackage_toc)
        # Update optimum table of contents
        base_toc.extend(subpackage_toc)
        with open(base_toc_path, "w") as f:
            yaml.safe_dump(base_toc, f, allow_unicode=True)


if __name__ == "__main__":
    main()
