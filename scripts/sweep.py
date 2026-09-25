#!/usr/bin/env python3
"""Build and measure a set of compile-time convolution specializations."""

import argparse
import csv
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default="workloads.csv", type=Path)
    parser.add_argument("--cases", help="comma-separated case names; default all")
    parser.add_argument("--arch", default="sm_75")
    parser.add_argument("--output", default="build/dispatch_measurements.csv",
                        type=Path)
    parser.add_argument("--iterations", type=int, default=101)
    parser.add_argument("--with-cudnn", action="store_true")
    parser.add_argument("--cudnn-root", default="/opt/cudnn-linux-x86_64-9.25.0.15_cuda13-archive")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    configs = args.configs if args.configs.is_absolute() else root / args.configs
    with configs.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    requested = args.cases.split(",") if args.cases else [row["name"] for row in rows]
    known = {row["name"] for row in rows}
    unknown = sorted(set(requested) - known)
    if unknown:
        raise SystemExit(f"unknown cases: {', '.join(unknown)}")

    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    measurements = []
    for case in requested:
        make_args = ["make", f"CASE={case}", f"CONFIG={configs}", f"ARCH={args.arch}"]
        if args.with_cudnn:
            make_args += ["WITH_CUDNN=1", f"CUDNN_ROOT={args.cudnn_root}"]
        subprocess.run(make_args, cwd=root, check=True)
        result = subprocess.run(
            ["./build/conv_lab", "--strategy", "both", "--iterations",
             str(args.iterations), "--csv"],
            cwd=root, check=True, text=True, capture_output=True)
        measurement = next(csv.DictReader(result.stdout.splitlines()))
        if args.with_cudnn and int(next(r for r in rows if r["name"] == case)["c"]) % 32 == 0 and int(next(r for r in rows if r["name"] == case)["k"]) % 32 == 0:
            cudnn = subprocess.run(["./build/cudnn_lab"], cwd=root, check=True,
                                   text=True, capture_output=True)
            cudnn_row = next(csv.DictReader(cudnn.stdout.splitlines()))
            dispatch_ms = float(measurement["strip_ms"] if measurement["winner"] == "strip" else measurement["generic_ms"])
            cudnn_ms = float(cudnn_row["cudnn_ms"])
            measurement["dispatch_ms"] = dispatch_ms
            measurement["cudnn_ms"] = cudnn_ms
            measurement["cudnn_over_dispatch"] = cudnn_ms / dispatch_ms
            measurement["comparison_contract"] = "dispatch:int32_vs_cudnn:int8x32"
        elif args.with_cudnn:
            measurement.update(dispatch_ms="", cudnn_ms="", cudnn_over_dispatch="",
                               comparison_contract="cudnn_int8x32_ineligible")
        measurements.append(measurement)
        with output.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=measurements[0].keys())
            writer.writeheader()
            writer.writerows(measurements)
        print(f"{case}: wrote {output}")


if __name__ == "__main__":
    main()
