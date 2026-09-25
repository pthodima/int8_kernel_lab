#!/usr/bin/env python3
"""Emit one compile-time ConvShape specialization from a workload CSV."""

import csv
import sys
from pathlib import Path


FIELDS = (
    "n", "h", "w", "c", "k", "r", "s", "pad_h", "pad_w",
    "stride_h", "stride_w", "dilation_h", "dilation_w",
)


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: gen_case.py WORKLOADS.csv CASE OUTPUT.cuh")
    configs, name, output = map(Path, (sys.argv[1], sys.argv[2], sys.argv[3]))
    name = str(name)
    with configs.open(newline="") as stream:
        row = next((item for item in csv.DictReader(stream)
                    if item["name"] == name), None)
    if row is None:
        raise SystemExit(f"unknown case {name!r} in {configs}")
    missing = [field for field in FIELDS if field not in row]
    if missing:
        raise SystemExit(f"{configs} is missing columns: {', '.join(missing)}")
    values = ", ".join(row[field] for field in FIELDS)
    text = (
        "#pragma once\n"
        '#include "conv_shape.cuh"\n'
        f"using TestShape = ConvShape<{values}>;\n"
        f'inline constexpr const char* kCaseName = "{name}";\n'
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)


if __name__ == "__main__":
    main()
