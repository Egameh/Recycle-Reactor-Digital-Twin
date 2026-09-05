"""
Tennessee Eastman Process (TEP) data loader.

Merges all training files (d00.dat ... d21.dat) into one labeled dataframe,
and all testing files (d00_te.dat ... d21_te.dat) into another.

Expects the classic whitespace-delimited .dat format, no header row,
52 process variable columns per row. If your files use a different
delimiter or have a header, tweak `read_one_file` below.

Usage:
    python tep_loader.py /path/to/data/folder
"""

import sys
import re
from pathlib import Path

import pandas as pd

# Standard TEP variable names: 41 measured (XMEAS) + 11 manipulated (XMV) = 52
COLUMN_NAMES = [f"xmeas_{i}" for i in range(1, 42)] + [f"xmv_{i}" for i in range(1, 12)]


def read_one_file(path: Path) -> pd.DataFrame:
    """
    Read a single .dat file into a dataframe with standard column names.

    Raw files are stored transposed: 52 rows (one per variable) x N columns
    (one per time sample). We transpose so rows = time samples, columns = variables.
    """
    raw = pd.read_csv(path, sep=r"\s+", header=None)

    if raw.shape[0] == len(COLUMN_NAMES):
        # variables x samples -> transpose to samples x variables
        df = raw.transpose().reset_index(drop=True)
    elif raw.shape[1] == len(COLUMN_NAMES):
        # already samples x variables
        df = raw
    else:
        raise ValueError(
            f"{path.name}: shape {raw.shape} doesn't match {len(COLUMN_NAMES)} "
            "variables on either axis. Check the file format."
        )

    df.columns = COLUMN_NAMES
    return df


def parse_fault_and_split(filename: str):
    """
    Parse fault number and split (train/test) from filename.
    Handles patterns like: d00.dat, d01.dat, d00_te.dat, d21_te.dat
    """
    match = re.match(r"d(\d{2})(_te)?\.dat", filename, re.IGNORECASE)
    if not match:
        return None, None
    fault_number = int(match.group(1))
    split = "test" if match.group(2) else "train"
    return fault_number, split


def load_all(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and merge all TEP files in data_dir.
    Returns (train_df, test_df), each tagged with fault_number, split, run_id, sample_index.
    """
    data_dir = Path(data_dir)
    train_frames, test_frames = [], []

    files = sorted(data_dir.glob("d*.dat"))
    if not files:
        raise FileNotFoundError(f"No d*.dat files found in {data_dir}")

    for path in files:
        fault_number, split = parse_fault_and_split(path.name)
        if fault_number is None:
            print(f"Skipping unrecognized file: {path.name}")
            continue

        df = read_one_file(path)
        df.insert(0, "sample_index", range(len(df)))
        df.insert(0, "run_id", path.stem)  # e.g. "d04_te"
        df.insert(0, "split", split)
        df.insert(0, "fault_number", fault_number)

        if split == "train":
            train_frames.append(df)
        else:
            test_frames.append(df)

        print(f"Loaded {path.name}: fault={fault_number}, split={split}, rows={len(df)}")

    train_df = pd.concat(train_frames, ignore_index=True)
    test_df = pd.concat(test_frames, ignore_index=True)

    print(f"\nTotal train rows: {len(train_df)} ({train_df['fault_number'].nunique()} fault classes)")
    print(f"Total test rows: {len(test_df)} ({test_df['fault_number'].nunique()} fault classes)")

    return train_df, test_df


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python tep_loader.py /path/to/data/folder")
        sys.exit(1)

    data_dir = sys.argv[1]
    train_df, test_df = load_all(data_dir)

    out_dir = Path(data_dir)
    train_df.to_parquet(out_dir / "tep_train_merged.parquet", index=False)
    test_df.to_parquet(out_dir / "tep_test_merged.parquet", index=False)

    print(f"\nSaved merged files to {out_dir}")
