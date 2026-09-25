#!/usr/bin/env python3
"""Print the machine-description flags the dispatch policy needs for a device.

The policy in `include/dispatch_policy.cuh` is a compile-time function, so the
machine it targets has to be described by macros rather than queried at run
time.  This emits those macros for a device, either the one installed here or
one named on the command line.

    ./scripts/target_flags.py                 # this machine
    ./scripts/target_flags.py --arch sm_75    # a known architecture
    make CASE=smoke ARCH=sm_75 \\
        NVCCFLAGS_EXTRA="$(./scripts/target_flags.py --arch sm_75)"

Shared memory per SM is the carveout an ordinary kernel can rely on, not the
physical size: the two differ on most architectures.  The CTAs-per-SM target is
the occupancy the kernels are tuned for and also bounds the register budget --
four 128-thread CTAs at 64K registers per SM is 128 registers per thread, which
is what the spatial kernels need for their 64 accumulators.
"""

import argparse
import subprocess
import sys

# shared-memory carveout usable by one CTA, and the register file, per SM
ARCHITECTURES = {
    "sm_70": dict(shared=96 * 1024, registers=65536, name="Volta"),
    "sm_75": dict(shared=64 * 1024, registers=65536, name="Turing"),
    "sm_80": dict(shared=163 * 1024, registers=65536, name="A100"),
    "sm_86": dict(shared=99 * 1024, registers=65536, name="Ampere consumer"),
    "sm_89": dict(shared=99 * 1024, registers=65536, name="Ada"),
    "sm_90": dict(shared=227 * 1024, registers=65536, name="Hopper"),
    "sm_100": dict(shared=227 * 1024, registers=65536, name="Blackwell datacentre"),
    "sm_120": dict(shared=100 * 1024, registers=65536, name="Blackwell consumer"),
}


def query_device():
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,compute_cap,count",
             "--format=csv,noheader"],
            check=True, text=True, capture_output=True).stdout.strip().splitlines()[0]
    except Exception:
        return None
    name, cap, _ = (field.strip() for field in out.split(","))
    arch = "sm_" + cap.replace(".", "")
    sms = None
    try:
        probe = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True, text=True, capture_output=True)
        del probe
    except Exception:
        pass
    return name, arch, sms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", help="sm_XX; default is the installed device")
    parser.add_argument("--sms", type=int, help="SM count; required with --arch")
    parser.add_argument("--ctas-per-sm", type=int, default=4,
                        help="occupancy the kernels are tuned for (default 4)")
    parser.add_argument("--waves", type=int, default=2,
                        help="CTA waves before the grid counts as full (default 2)")
    parser.add_argument("--comment", action="store_true",
                        help="annotate the output instead of emitting bare flags")
    args = parser.parse_args()

    arch, sms, device = args.arch, args.sms, None
    if arch is None:
        found = query_device()
        if found is None:
            sys.exit("no device found; pass --arch and --sms")
        device, arch, _ = found
    if arch not in ARCHITECTURES:
        sys.exit(f"unknown architecture {arch}; add it to ARCHITECTURES")
    if sms is None:
        try:
            import ctypes
            cuda = ctypes.CDLL("libcudart.so")
            count = ctypes.c_int()
            cuda.cudaDeviceGetAttribute(ctypes.byref(count), 16, 0)  # MultiProcessorCount
            sms = count.value or None
        except Exception:
            sms = None
    if not sms:
        sys.exit("could not determine the SM count; pass --sms")

    spec = ARCHITECTURES[arch]
    flags = [
        f"-DINT8_LAB_TARGET_SMS={sms}",
        f"-DINT8_LAB_TARGET_SHARED_PER_SM={spec['shared']}",
        f"-DINT8_LAB_TARGET_CTAS_PER_SM={args.ctas_per_sm}",
        f"-DINT8_LAB_TARGET_WAVES={args.waves}",
    ]
    if args.comment:
        label = device or spec["name"]
        print(f"# {label} ({arch}), {sms} SMs, "
              f"{spec['shared'] // 1024} KB shared per SM")
        print(f"# budget per CTA: {spec['shared'] // args.ctas_per_sm} B shared, "
              f"{spec['registers'] // (args.ctas_per_sm * 128)} registers per thread")
        for flag in flags:
            print(flag)
    else:
        print(" ".join(flags))


if __name__ == "__main__":
    main()
