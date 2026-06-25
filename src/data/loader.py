"""Load and lightly validate the raw MovieLens CSVs."""
from pathlib import Path
from typing import NamedTuple
import pandas as pd

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

_EXPECTED_COLUMNS = {
    "movies.csv": {"movieId", "title", "genres"},
    "ratings.csv": {"userId", "movieId", "rating", "timestamp"},
    "tags.csv": {"userId", "movieId", "tag", "timestamp"},
    "links.csv": {"movieId", "imdbId", "tmdbId"},
}


class MovieLensData(NamedTuple):
    movies: pd.DataFrame
    ratings: pd.DataFrame
    tags: pd.DataFrame
    links: pd.DataFrame


def _load_csv(filename: str, data_dir: Path) -> pd.DataFrame:
    path = data_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.data.download` first.")
    df = pd.read_csv(path)
    missing = _EXPECTED_COLUMNS[filename] - set(df.columns)
    if missing:
        raise ValueError(f"{filename} is missing expected columns {missing}. Found: {list(df.columns)}.")
    return df


def load_movies(data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    return _load_csv("movies.csv", data_dir)


def load_ratings(data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    return _load_csv("ratings.csv", data_dir)


def load_tags(data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    return _load_csv("tags.csv", data_dir)


def load_links(data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    return _load_csv("links.csv", data_dir)


def load_all(data_dir: Path = RAW_DATA_DIR) -> MovieLensData:
    return MovieLensData(
        movies=load_movies(data_dir), ratings=load_ratings(data_dir),
        tags=load_tags(data_dir), links=load_links(data_dir),
    )


def extract_year(movies: pd.DataFrame) -> pd.DataFrame:
    out = movies.copy()
    year_extracted = out["title"].str.extract(r"\((\d{4})\)\s*$")[0]
    out["year"] = pd.to_numeric(year_extracted, errors="coerce")
    out["title_clean"] = out["title"].str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True)
    return out
