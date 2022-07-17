import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from transformers import HfArgumentParser

import yaml


@dataclass
class DocArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    subpackages: List[str] = field(metadata={"help": "Subpackages to integrate docs with Optimum"})
    version: str = field(
        default="main",
        metadata={"help": "The version of the docs"},
    )


def main():
    parser = HfArgumentParser(DocArguments)
    args = parser.parse_args()
    base_path = Path("base-doc-build")
    for subpackage in args.subpackages:
        subpackage_path = Path(f"{subpackage}-doc-build")
        # Copy all HTML files from subpackage into optimum
        subpackage_html_paths = list(subpackage_path.rglob("*.html"))
        for html_path in subpackage_html_paths:
            shutil.copyfile(html_path, f"base-doc-build/optimum/{args.version}/en/{subpackage}_{html_path.name}")
        # Load optimum table of contents
        base_toc_path = next(base_path.rglob("_toctree.yml"))
        with open(base_toc_path, "r") as f:
            base_toc = yaml.safe_load(f)
        # Load subpackage table of contents
        subpackage_toc_path = next(subpackage_path.rglob("_toctree.yml"))
        with open(subpackage_toc_path, "r") as f:
            subpackage_toc = yaml.safe_load(f)
        # Extend table of contents sections with subpackage name
        for item in subpackage_toc:
            for file in item["sections"]:
                file["local"] = f"{subpackage}_" + file["local"]
        # Udpate optimum table of contents
        base_toc.extend(subpackage_toc)
        with open(base_toc_path, "w") as f:
            yaml.safe_dump(base_toc, f, allow_unicode=True)


if __name__ == "__main__":
    main()
