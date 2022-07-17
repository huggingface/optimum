import argparse
import shutil
from pathlib import Path

import yaml


parser = argparse.ArgumentParser()
parser.add_argument("--subpackages", nargs="+", help="Subpackages to integrate docs with Optimum")
parser.add_argument("--version", type=str, default="main", help="The version of the Optimum docs")


def main():
    args = parser.parse_args()
    optimum_path = Path("optimum-doc-build")
    for subpackage in args.subpackages:
        subpackage_path = Path(f"{subpackage}-doc-build")
        # Copy all HTML files from subpackage into optimum
        subpackage_html_paths = list(subpackage_path.rglob("*.html"))
        for html_path in subpackage_html_paths:
            shutil.copyfile(html_path, f"optimum-doc-build/optimum/{args.version}/en/{subpackage}_{html_path.name}")
        # Load optimum table of contents
        base_toc_path = next(optimum_path.rglob("_toctree.yml"))
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
        # Update optimum table of contents
        base_toc.extend(subpackage_toc)
        with open(base_toc_path, "w") as f:
            yaml.safe_dump(base_toc, f, allow_unicode=True)


if __name__ == "__main__":
    main()
