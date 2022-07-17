import shutil
from pathlib import Path

import yaml


def main():
    base_path = Path("base-doc-build")
    subpackage_path = Path("habana-doc-build")
    subpackage_html = list(subpackage_path.rglob("*.html"))
    for html in subpackage_html:
        shutil.copyfile(html, f"base-doc-build/optimum/main/en/habana_{html.name}")

    base_toc_path = next(base_path.rglob("_toctree.yml"))
    with open(base_toc_path, "r") as f:
        base_toc = yaml.safe_load(f)

    subpackage_toc_path = next(subpackage_path.rglob("_toctree.yml"))
    with open(subpackage_toc_path, "r") as f:
        subpackage_toc = yaml.safe_load(f)

    for item in subpackage_toc:
        for file in item["sections"]:
            file["local"] = "habana_" + file["local"]

    base_toc.extend(subpackage_toc)
    with open(base_toc_path, "w") as f:
        yaml.safe_dump(base_toc, f, allow_unicode=True)


if __name__ == "__main__":
    main()
