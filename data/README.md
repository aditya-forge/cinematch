# Data Directory

This directory is the storage location for raw TMDB CSV files and dataset documentation.

## Downloading the Raw Data

1. Visit the [Kaggle TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata).
2. Download the two source files:
   - `tmdb_5000_movies.csv`
   - `tmdb_5000_credits.csv`
3. Place both files in this directory (`data/`).

## Generated Model Artifacts

Executing `notebooks/analysis.ipynb` generates two serialized artifacts saved in the project root:

| File | Description | Approximate Size |
|---|---|---|
| `movie.pkl` | Processed movie DataFrame with consolidated tag strings | ~2.4 MB |
| `similarity.pkl` | Pairwise cosine similarity matrix (4806 x 4806 float values) | ~184 MB |

Note: Due to GitHub file size constraints, `similarity.pkl` is excluded via `.gitignore` and must be generated locally by executing the analysis notebook.
