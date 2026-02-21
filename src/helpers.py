"""
Helper functions for data access and general utilities used across the
cutting blade condition monitoring project.

This module provides small, reusable building blocks that are not specific
to preprocessing or feature engineering, but are needed by both.

Typical responsibilities include:
- Downloading or locating the dataset on disk.
- Listing and loading all raw CSV files into DataFrames.
- Generic path and filesystem helpers.
- Lightweight convenience wrappers used throughout the notebooks and modules.

The aim is to centralize shared utility code here to avoid duplication and
keep domain-specific modules focused on their core responsibilities.
"""

import subprocess
from pathlib import Path
import pandas as pd

DATASET_NAME = "one-year-industrial-component-degradation"


def download_dataset(data_dir: str = "data") -> Path:
    """
    Ensure that the Kaggle dataset is downloaded and extracted into data_dir.

    - If data_dir does not exist: create and download.
    - If data_dir exists but is (almost) empty: download.
    - If data_dir contains files: assume dataset is present.

    Args:
        data_dir (str): data directory to store the dataset in.

    Returns:
        Path: Path to the extracted dataset.
    """
    data_path = Path(data_dir)

    dir_exists = data_path.exists()
    has_files = dir_exists and any(data_path.iterdir())

    if not dir_exists or not has_files:
        data_path.mkdir(parents=True, exist_ok=True)
        print("Downloading dataset from Kaggle into:", data_path.resolve())

        result = subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                f"init-owl/{DATASET_NAME}",
                "-p",
                data_dir,
                "--unzip",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print("Download completed.")
        else:
            print("Kaggle download failed.")
            print(result.stderr)
    else:
        print("Dataset directory already contains files – no download necessary.")

    print(f"Using data directory: {data_path}")
    return data_path


def load_all_raw_files(data_dir: Path) -> list[pd.DataFrame]:
    """
    Read all CSV files from the one-year dataset and return a list of DataFrames,
    each with an added 'filename' column.

    Args:
        data_dir (Path): Path to load the raw CSV files from.

    Returns:
        df_list (list[pd.DataFrame]): List of DataFrames containing the raw data.
    """
    oneyear_dir = data_dir / "oneyeardata"
    data_files = oneyear_dir.glob("*.csv")

    df_list: list[pd.DataFrame] = []
    for f in data_files:
        df = pd.read_csv(f, encoding="utf-8-sig")
        df["filename"] = f.name
        df_list.append(df)

    return df_list


def load_aggregated_table(
    data_dir: str = "data",
    filename: str = "cutting_blade_aggregated.csv",
) -> pd.DataFrame:
    """
    Load the aggregated feature table used for modeling.

    The function expects that a preprocessing step (implemented in `features.py`)
    has already created a single CSV file containing one row per recording /
    file and all engineered features (e.g. means, standard deviations, min/max,
    outlier counts) as well as target columns such as `mode`, `degradation`
    and `file_anomaly`.

    Args:
        data_dir : str, optional
            Directory where the aggregated CSV file is stored. Defaults to "data".
        filename : str, optional
            Name of the aggregated CSV file. Defaults to "cutting_blade_aggregated.csv".

    Returns:
        pd.DataFrame: DataFrame containing the aggregated feature table that is used
        as input for model training and evaluation.
    """
    data_path = Path(data_dir) / filename
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} not found. Adjust filename or create/export the "
            "aggregated table first."
        )
    return pd.read_csv(data_path)
