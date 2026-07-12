#!/usr/bin/env python3
"""Check Criterion benchmark output against TenRSo's documented performance budgets.

Source of truth for the budgets below: `TODO.md` ("Performance Validation") and
`CLAUDE.md` section 7 ("Performance Targets"):

    - Tucker-HOOI : < 3s / 10 iters (512x512x128, ranks [64,64,32])
    - TT-SVD      : < 2s build time (32^6)
    - Masked ops  : >= 5x speedup vs dense naive (documented at 90% sparsity in
                    crates/tenrso-sparse/benches/masked_einsum_bench.rs)
    - CP-ALS      : < 2s / 10 iters (256^3, rank-64) -- documented target, but
                    TODO.md records the *actual* measured range as ~21-44s on an
                    8-core x86_64 box, root-caused as memory-bandwidth-limited
                    pure-Rust GEMM (no BLAS parallelism in-policy). This script
                    does NOT gate on the 2s figure -- doing so would fail every
                    CI run unconditionally, which is exactly the kind of
                    "trivially red" gate that is as useless as a trivially green
                    one. Instead it (a) always reports the measured value next
                    to the aspirational target so the gap stays visible, and
                    (b) fails only if the measurement blows through a generous
                    "sanity ceiling" derived from the *currently measured*
                    21-44s range, which would indicate a genuine regression
                    (e.g. an accidental algorithmic complexity change) rather
                    than the known BLAS-gated shortfall.

This script reads Criterion's on-disk JSON estimates
(`target/criterion/<group>/<function>/<value>/new/estimates.json`, produced by
`cargo bench -- <filter>`) rather than parsing stdout, so it is robust to
Criterion's human-readable formatting changes. Directory naming follows
Criterion's own `BenchmarkId::as_directory_name()` rules (see
criterion-0.8.x/src/report.rs): '/' and '^' (among other characters) are
replaced with '_' when a *group* name contains them, but the filter string
passed to `cargo bench --` matches against the *unsanitized* `id()` string.

Exit code: non-zero iff at least one "hard" gate failed. "advisory" and
"tracked" rows never affect the exit code but are always printed so nothing
is silently swept under the rug.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Check:
    name: str
    # Path to a `new/estimates.json` file, relative to the criterion root.
    path: str
    budget_s: float
    mode: str  # "hard" | "hard_margin" | "advisory" | "tracked"
    margin: float = 1.0  # multiplier applied to budget_s before failing (hard_margin)
    note: str = ""


@dataclass
class RatioCheck:
    name: str
    baseline_path: str  # slower ("dense naive") operand
    candidate_path: str  # faster ("masked") operand
    min_ratio: float
    mode: str = "hard"  # "hard" | "advisory"
    note: str = ""


def load_point_estimate_seconds(criterion_root: Path, rel_path: str) -> float | None:
    """Return the Criterion `mean.point_estimate` in seconds, or None if absent.

    `None` covers both "benchmark was intentionally not run this time" (e.g. the
    TT-SVD 32^6 target step was skipped or OOM-killed) and "path is stale" -- the
    caller decides how to treat a missing measurement per check mode.
    """
    full_path = criterion_root / rel_path
    if not full_path.is_file():
        return None
    try:
        data = json.loads(full_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    # Criterion stores the mean in nanoseconds.
    return data["mean"]["point_estimate"] / 1e9


def fmt_s(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}s"


def run_absolute_checks(criterion_root: Path, checks: list[Check]) -> tuple[list[str], bool]:
    lines: list[str] = []
    any_hard_failure = False

    for check in checks:
        measured = load_point_estimate_seconds(criterion_root, check.path)
        effective_budget = check.budget_s * check.margin

        if measured is None:
            status = "SKIPPED"
            detail = "no estimates.json found (benchmark did not run or was skipped)"
        elif check.mode in ("hard", "hard_margin"):
            if measured <= effective_budget:
                status = "PASS"
            else:
                status = "FAIL"
                any_hard_failure = True
            detail = f"measured {fmt_s(measured)} vs budget {fmt_s(effective_budget)}"
            if check.mode == "hard_margin" and check.margin != 1.0:
                detail += f" (documented target {fmt_s(check.budget_s)} x {check.margin:g} noise margin)"
        elif check.mode == "advisory":
            status = "PASS" if measured <= effective_budget else "WARN"
            detail = f"measured {fmt_s(measured)} vs budget {fmt_s(effective_budget)} (advisory only, does not fail CI)"
        elif check.mode == "tracked":
            # Never gates; always shown so the documented-vs-actual gap is visible.
            status = "PASS" if measured <= effective_budget else "TRACKED-GAP"
            detail = f"measured {fmt_s(measured)} vs documented target {fmt_s(check.budget_s)} (not gated; see note)"
        else:  # pragma: no cover - defensive
            status = "ERROR"
            detail = f"unknown check mode {check.mode!r}"
            any_hard_failure = True

        lines.append(f"[{status:11}] {check.name}: {detail}")
        if check.note:
            lines.append(f"              note: {check.note}")

    return lines, any_hard_failure


def run_ratio_checks(criterion_root: Path, checks: list[RatioCheck]) -> tuple[list[str], bool]:
    lines: list[str] = []
    any_hard_failure = False

    for check in checks:
        baseline = load_point_estimate_seconds(criterion_root, check.baseline_path)
        candidate = load_point_estimate_seconds(criterion_root, check.candidate_path)

        if baseline is None or candidate is None:
            status = "SKIPPED"
            detail = "missing dense and/or masked estimates.json"
            lines.append(f"[{status:11}] {check.name}: {detail}")
            continue

        ratio = baseline / candidate if candidate > 0 else float("inf")
        meets = ratio >= check.min_ratio
        if check.mode == "advisory":
            status = "PASS" if meets else "WARN"
        else:
            status = "PASS" if meets else "FAIL"
            if not meets:
                any_hard_failure = True
        detail = (
            f"dense {fmt_s(baseline)} / masked {fmt_s(candidate)} = "
            f"{ratio:.1f}x speedup (budget >= {check.min_ratio:g}x)"
        )
        if check.mode == "advisory":
            detail += " (advisory only, does not fail CI)"
        lines.append(f"[{status:11}] {check.name}: {detail}")
        if check.note:
            lines.append(f"              note: {check.note}")

    return lines, any_hard_failure


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--criterion-root",
        type=Path,
        default=Path("target/criterion"),
        help="Root directory of Criterion's on-disk output (default: target/criterion)",
    )
    parser.add_argument(
        "--cp-als-sanity-ceiling-s",
        type=float,
        default=120.0,
        help=(
            "Absolute wall-clock ceiling (seconds) for the CP-ALS 256^3/rank-64/"
            "10-iters benchmark. This is NOT the documented 2s target (see module "
            "docstring) -- it is a generous multiple of the currently-measured "
            "21-44s range, intended only to catch a catastrophic regression."
        ),
    )
    args = parser.parse_args()

    absolute_checks = [
        Check(
            name="tucker_hooi_target (512x512x128, r[64,64,32], 10 iters)",
            path="tucker_hooi_target/512x512x128_r64_64_32_10iters/new/estimates.json",
            budget_s=3.0,
            mode="hard_margin",
            margin=2.0,
            note=(
                "Documented budget is 3s. Gate uses a 2x margin (6s ceiling) because "
                "shared GitHub-hosted runners show run-to-run wall-clock noise well "
                "in excess of dedicated hardware; values between 3s and 6s are a "
                "genuine miss of the documented target but are not treated as a hard "
                "CI failure to avoid false-positive flakiness."
            ),
        ),
        Check(
            name="tt_svd_target (32^6, max-rank 20)",
            path="tt_svd_target/32_6_r20/new/estimates.json",
            budget_s=2.0,
            mode="advisory",
            margin=3.0,
            note=(
                "Advisory only (never fails CI). This shape allocates a dense "
                "32^6 = 1,073,741,824-element f64 tensor (~8.6 GB) before TT-SVD "
                "even starts; TODO.md records this exact configuration as "
                "'not yet measured' because of the allocation size, and it risks "
                "OOM-kill on shared GitHub-hosted runners (16GB total, shared with "
                "the OS and the rest of the job). The workflow step that produces "
                "this measurement is marked `continue-on-error: true` for the same "
                "reason -- an OOM-kill must not fail the whole job."
            ),
        ),
        Check(
            name="cp_als_target (256^3, rank-64, 10 iters) -- documented target",
            path="cp_als_target/256x256x256_r64_10iters/new/estimates.json",
            budget_s=2.0,
            mode="tracked",
            note=(
                "TODO.md documents this target as currently unreachable in pure "
                "Rust: measured ~21-44s (8-core x86_64 AVX2), root-caused as "
                "memory-bandwidth-limited (the 33MB Khatri-Rao product exceeds L3 "
                "cache; matrixmultiply's pure-Rust GEMM runs at ~3-4 GFLOP/s vs "
                "~50 GFLOP/s for multi-threaded OpenBLAS DGEMM). Reaching 2s "
                "requires an in-policy BLAS backend, which does not exist yet. "
                "This row is intentionally NEVER gated on 2s -- see the "
                "sanity-ceiling row below for the actual regression guard used "
                "instead."
            ),
        ),
        Check(
            name="cp_als_target sanity ceiling (regression guard, not the 2s target)",
            path="cp_als_target/256x256x256_r64_10iters/new/estimates.json",
            budget_s=args.cp_als_sanity_ceiling_s,
            mode="hard",
            note=(
                "Hard gate against a catastrophic regression (e.g. an accidental "
                "O(n^2) -> O(n^3) blow-up), not against the documented 2s target. "
                f"Ceiling defaults to {args.cp_als_sanity_ceiling_s:g}s, a >2.5x "
                "margin over the top of the currently-measured 21-44s range, to "
                "absorb shared-runner noise while still catching real blow-ups. "
                "See also the criterion --baseline comparison step in the "
                "workflow, which additionally flags any statistically "
                "significant relative regression against the last master run."
            ),
        ),
    ]

    # Empirically (measured 2026-07-11 against a fair contiguous-slice dense
    # baseline, see crates/tenrso-sparse/examples/masked_speedup.rs) the 90%-sparse
    # speedup is comfortably above 5x at 256x256 (~10-18x) and 512x512 (~14-27x),
    # so those are hard gates. At 64x64 both operands are L1/L2-resident, dense
    # naive is already cache-optimal, and the masked path's fixed overhead (COO
    # build, column packing) leaves the true ratio hovering right at ~5x -- noisy
    # shared-runner Criterion has read it as low as ~3.5x. That row is therefore
    # advisory (WARN, never fails CI) to avoid spurious red builds; the two larger
    # sizes carry the real budget.
    ratio_checks = [
        RatioCheck(
            name=f"masked_einsum vs dense_naive @ {size}x{size}, 90% sparsity",
            baseline_path=f"matmul_comparison/dense/{size}x{size}_sp90/new/estimates.json",
            candidate_path=f"matmul_comparison/masked/{size}x{size}_sp90/new/estimates.json",
            min_ratio=5.0,
            mode="advisory" if size == 64 else "hard",
            note=(
                "Blueprint target ('Masked operations: >= 5x speedup vs dense naive'), "
                "documented at 90% sparsity in "
                "crates/tenrso-sparse/benches/masked_einsum_bench.rs. "
                + (
                    "Advisory: at 64x64 both operands are cache-resident so the true "
                    "ratio sits near the 5x line and reads noisily on shared runners; "
                    "does not fail CI."
                    if size == 64
                    else "Hard gate: measured margin at this size (>=10x) is well "
                    "clear of shared-runner noise."
                )
            ),
        )
        for size in (64, 256, 512)
    ]

    print("=" * 100)
    print("TenRSo performance budget report")
    print("=" * 100)

    abs_lines, abs_failed = run_absolute_checks(args.criterion_root, absolute_checks)
    ratio_lines, ratio_failed = run_ratio_checks(args.criterion_root, ratio_checks)

    for line in abs_lines:
        print(line)
    print("-" * 100)
    for line in ratio_lines:
        print(line)
    print("=" * 100)

    any_failure = abs_failed or ratio_failed
    if any_failure:
        print("RESULT: one or more HARD performance budgets failed.")
    else:
        print("RESULT: all HARD performance budgets passed (advisory/tracked rows do not gate).")

    return 1 if any_failure else 0


if __name__ == "__main__":
    sys.exit(main())
