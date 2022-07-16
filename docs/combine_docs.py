import shutil
from pathlib import Path

import yaml


def main():
    subpackage_path = Path("habana-doc-build")
    subpackage_html = list(subpackage_path.rglob("*.html"))
    for html in subpackage_html:
        shutil.copyfile(html, f"base-doc-build/optimum/main/en/habana_{html.name}")


if __name__ == "__main__":
    main()
