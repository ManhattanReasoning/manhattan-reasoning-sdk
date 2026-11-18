"""Human-readable cloud feedback; older servers may not send a report."""

import math


def _number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def print_build_report(report):
    if not isinstance(report, dict):
        return
    if report.get("schema_version") != 1:
        print("  Build report: unsupported version; use mrg job --json to inspect.")
        return
    print("  Build report (whole FPGA design):")
    clock = report.get("sys_clk_freq_hz")
    if _number(clock):
        print(f"    Configured user clock: {clock / 1e6:g} MHz")
    fmax = report.get("fmax_mhz")
    target = report.get("timing_target_mhz")
    if _number(fmax):
        print(f"    User-clock Fmax: {fmax:g} MHz")
    if _number(target):
        print(f"    User timing target: {target:g} MHz")
    timing = report.get("timing_met")
    if timing is True:
        print("    Timing: MET for all reported clock domains")
    elif timing is False:
        print("    WARNING: timing NOT MET; the design may not run reliably.")
    else:
        print("    Timing: UNKNOWN (not verified)")
    resources = report.get("utilization")
    if isinstance(resources, dict):
        for key, label in [
            ("lut", "LUT"),
            ("ff", "Registers"),
            ("bram", "BRAM"),
            ("dsp", "DSP"),
        ]:
            value = resources.get(key)
            if not isinstance(value, dict):
                continue
            used, available, percent = (
                value.get(k) for k in ("used", "available", "percent")
            )
            if all(_number(n) for n in (used, available, percent)):
                print(f"    {label}: {used:g}/{available:g} ({percent:g}%)")
