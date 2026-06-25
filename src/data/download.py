"""Download and extract the MovieLens dataset. Run inside Codespaces (has internet access)."""
import argparse
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

DATASET_URLS = {
    "small": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
    "full": "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
}
RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def download_movielens(variant: str = "small") -> Path:
    if variant not in DATASET_URLS:
        raise ValueError(f"variant must be one of {list(DATASET_URLS)}, got {variant!r}")
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    url = DATASET_URLS[variant]
    zip_path = RAW_DATA_DIR / f"{variant}.zip"
    print(f"Downloading MovieLens ({variant}) from {url} ...")
    urlretrieve(url, zip_path)
    print("Extracting ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(RAW_DATA_DIR)
    extracted_dirs = [p for p in RAW_DATA_DIR.iterdir() if p.is_dir() and p.name.startswith("ml-")]
    if extracted_dirs:
        src_dir = extracted_dirs[0]
        for f in src_dir.glob("*.csv"):
            shutil.move(str(f), RAW_DATA_DIR / f.name)
        shutil.rmtree(src_dir)
    zip_path.unlink()
    print(f"Done. Files available in: {RAW_DATA_DIR}")
    return RAW_DATA_DIR


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download the MovieLens dataset.")
    parser.add_argument("--variant", choices=["small", "full"], default="small")
    args = parser.parse_args()
    download_movielens(args.variant)
