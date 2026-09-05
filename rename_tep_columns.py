"""
Renames TEP merged parquet columns from generic xmeas_*/xmv_* labels
to real plant variable names, per Downs & Vogel (1993) / teprob.f documentation.

Reads tep_train_merged.parquet and tep_test_merged.parquet,
writes tep_train_renamed.parquet and tep_test_renamed.parquet
in the same folder. Originals are left untouched.

Usage:
    python rename_tep_columns.py /path/to/data/folder
"""

import sys
from pathlib import Path

import pandas as pd

# Official XMEAS(1-41) names, Downs & Vogel (1993) / teprob.f documentation
XMEAS_NAMES = {
    1: "a_feed_flow",
    2: "d_feed_flow",
    3: "e_feed_flow",
    4: "a_and_c_feed_flow",
    5: "recycle_flow",
    6: "reactor_feed_rate",
    7: "reactor_pressure",
    8: "reactor_level",
    9: "reactor_temperature",
    10: "purge_rate",
    11: "product_sep_temp",
    12: "product_sep_level",
    13: "product_sep_pressure",
    14: "product_sep_underflow",
    15: "stripper_level",
    16: "stripper_pressure",
    17: "stripper_underflow",
    18: "stripper_temperature",
    19: "stripper_steam_flow",
    20: "compressor_work",
    21: "reactor_cw_outlet_temp",
    22: "separator_cw_outlet_temp",
    # Reactor feed analysis (stream 6), mole %
    23: "reactor_feed_component_a",
    24: "reactor_feed_component_b",
    25: "reactor_feed_component_c",
    26: "reactor_feed_component_d",
    27: "reactor_feed_component_e",
    28: "reactor_feed_component_f",
    # Purge gas analysis (stream 9), mole %
    29: "purge_gas_component_a",
    30: "purge_gas_component_b",
    31: "purge_gas_component_c",
    32: "purge_gas_component_d",
    33: "purge_gas_component_e",
    34: "purge_gas_component_f",
    35: "purge_gas_component_g",
    36: "purge_gas_component_h",
    # Product analysis (stream 11), mole %
    37: "product_component_d",
    38: "product_component_e",
    39: "product_component_f",
    40: "product_component_g",
    41: "product_component_h",
}

# Official XMV(1-11) names (12th, agitator speed, isn't in the 52-column dataset)
XMV_NAMES = {
    1: "d_feed_flow_valve",
    2: "e_feed_flow_valve",
    3: "a_feed_flow_valve",
    4: "a_and_c_feed_flow_valve",
    5: "compressor_recycle_valve",
    6: "purge_valve",
    7: "separator_pot_liquid_flow_valve",
    8: "stripper_liquid_product_flow_valve",
    9: "stripper_steam_valve",
    10: "reactor_cw_flow_valve",
    11: "condenser_cw_flow_valve",
}


def build_rename_map() -> dict:
    rename_map = {}
    for i, name in XMEAS_NAMES.items():
        rename_map[f"xmeas_{i}"] = name
    for i, name in XMV_NAMES.items():
        rename_map[f"xmv_{i}"] = name
    return rename_map


def rename_file(in_path: Path, out_path: Path, rename_map: dict) -> None:
    df = pd.read_parquet(in_path)
    missing = [c for c in rename_map if c not in df.columns]
    if missing:
        print(f"Warning: {in_path.name} is missing expected columns: {missing}")
    df = df.rename(columns=rename_map)
    df.to_parquet(out_path, index=False)
    print(f"Saved {out_path.name}: {df.shape[0]} rows, {df.shape[1]} columns")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python rename_tep_columns.py /path/to/data/folder")
        sys.exit(1)

    data_dir = Path(sys.argv[1])
    rename_map = build_rename_map()

    rename_file(
        data_dir / "tep_train_merged.parquet",
        data_dir / "tep_train_renamed.parquet",
        rename_map,
    )
    rename_file(
        data_dir / "tep_test_merged.parquet",
        data_dir / "tep_test_renamed.parquet",
        rename_map,
    )
