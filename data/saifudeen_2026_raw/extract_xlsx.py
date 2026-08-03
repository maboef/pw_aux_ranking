"""
Extract specific sheets, columns, and rows from a multi-sheet .xlsx file
into separate CSV files.

Fill in the CONFIG section below, then run:
    python extract_sheets_to_csv.py
"""

import pandas as pd


SOURCE_FILE = "Saifudeen_2026_raw.xlsx"

# One entry per sheet you want to extract.
# - sheet:       sheet name (or integer index, e.g. 0 for the first sheet)
# - header_row:  0-based row number containing column headers (None if no header)
# - columns:     list of column names (or letters, e.g. "A", "C:E") to keep;
#                use None to keep all columns
# - row_filter:  optional function(df) -> boolean mask, for filtering rows;
#                use None to keep all rows
# - row_range:   optional (start, end) tuple of 0-based row indices (after
#                header) to slice, e.g. (0, 100); use None to keep all rows
# - output_csv:  output filename

EXTRACTIONS = [
    {
        "sheet": "Table S3",
        "header_row": 11,
        "columns": None,
        "row_filter": None,          # e.g. lambda df: df["Col B"] > 10
        "row_range": None,           # e.g. (0, 50)
        "output_csv": "raw_dose_curves_full_set.csv",
    }
]


# One entry per output CSV you want built by joining two sheets, SQL-style.
# - left / right:   each is a sheet-spec dict, same format as EXTRACTIONS
#                    entries above (minus "output_csv", which isn't needed here)
# - left_on:        key column name in the left sheet
# - right_on:       key column name in the right sheet (same as left_on if equal)
# - how:            "inner", "left", "right", or "outer" (pandas merge semantics)
# - suffixes:       tuple to disambiguate overlapping non-key column names,
#                    e.g. ("_x", "_y")
# - output_csv:     output filename

MERGES = [
    {
        "left": {
            "sheet": "Table S4",
            "header_row": 8,
            "columns": None,
            "row_filter": None,
            "row_range": None,
        },
        "right": {
            "sheet": "Table S13",
            "header_row": 7,
            "columns": None,
            "row_filter": None,
            "row_range": None,
        },
        "left_on": "Compound",
        "right_on": "Compound",
        "how": "outer",
        "suffixes": ("_left", "_right"),
        "output_csv": "raw_percent_inhibition_full_set.csv",
    },
]

# -------------------------------------------------------- #


def extract(source_file: str, spec: dict) -> pd.DataFrame:
    columns = spec.get("columns")
    is_excel_range = isinstance(columns, str)  # e.g. "A:NH" or "A,C,F:H"

    df = pd.read_excel(
        source_file,
        sheet_name=spec["sheet"],
        header=spec.get("header_row", 0),
        usecols=columns if is_excel_range else None,
    )

    if columns and not is_excel_range:
        df = df[columns]  # list of header names

    if spec.get("row_range"):
        start, end = spec["row_range"]
        df = df.iloc[start:end]

    if spec.get("row_filter"):
        df = df[spec["row_filter"](df)]

    return df


def main():
    for spec in EXTRACTIONS:
        df = extract(SOURCE_FILE, spec)
        df.to_csv(spec["output_csv"], index=False)
        print(f"Wrote {len(df)} rows x {len(df.columns)} cols -> {spec['output_csv']}")

    for merge_spec in MERGES:
        left_df = extract(SOURCE_FILE, merge_spec["left"])
        right_df = extract(SOURCE_FILE, merge_spec["right"])

        merged = pd.merge(
            left_df,
            right_df,
            left_on=merge_spec["left_on"],
            right_on=merge_spec["right_on"],
            how=merge_spec.get("how", "inner"),
            suffixes=merge_spec.get("suffixes", ("_left", "_right")),
        )

        merged.to_csv(merge_spec["output_csv"], index=False)
        print(
            f"Merged -> {len(merged)} rows x {len(merged.columns)} cols "
            f"-> {merge_spec['output_csv']}"
        )


if __name__ == "__main__":
    main()