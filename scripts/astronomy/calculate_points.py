#!/usr/bin/env python3
"""
Recalculate center frequency and linear polarization points from raw Stokes data.

Inputs:
  <data_dir>/selection_windows.csv
  <data_dir>/raw/*_polarization.h5

Outputs:
  <save_dir>/derived_points.csv
  <save_dir>/derived_points_plot_columns.csv
  <save_dir>/paper_table_s3_comparison.csv
  <save_dir>/derived_vs_reference.png

Index convention in selection_windows.csv:
  Stokes arrays are [frequency, time].
  start/end indices are inclusive endpoints.
"""

from __future__ import annotations

import argparse
import os
import math
import h5py
import numpy as np
import pandas as pd
from typing import Any
from pathlib import Path



REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(REPO_ROOT)

FLOAT_FIELDS = [
    "freq_start_mhz",
    "freq_end_mhz",
    "time_resolution_s",
]

INT_FIELDS = [
    "analysis_order",
    "paper_burst_number",
    "noise_start_idx",
    "noise_end_idx",
    "burst_start_idx",
    "burst_end_idx",
    "freq_start_idx",
    "freq_end_idx",
]


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def background_subtract_per_frequency(
    arr: np.ndarray,
    noise_sl: slice,
) -> np.ndarray:
    baseline = np.nanmean(arr[:, noise_sl], axis=1, keepdims=True)
    return arr - baseline


def debiased_linear_profile(
    q_profile: np.ndarray,
    u_profile: np.ndarray,
    sigma_i: float,
    threshold: float = 1.57,
) -> tuple[np.ndarray, np.ndarray]:
    measured_l = np.sqrt(q_profile * q_profile + u_profile * u_profile)
    if not np.isfinite(sigma_i) or sigma_i <= 0:
        return np.zeros_like(measured_l), measured_l

    snr = measured_l / sigma_i
    debiased_l = np.where(
        snr > threshold,
        sigma_i * np.sqrt(np.maximum(snr * snr - 1.0, 0.0)),
        0.0,
    )
    return debiased_l, measured_l


def snr_weighted_center_ghz(
    frequency_mhz: np.ndarray,
    i_sub: np.ndarray,
    freq_sl: slice,
    noise_sl: slice,
    burst_sl: slice,
    n_burst_samples: int,
) -> float:
    freq = frequency_mhz[freq_sl]
    signal = np.nansum(i_sub[freq_sl, burst_sl], axis=1)
    noise_sigma = np.nanstd(i_sub[freq_sl, noise_sl], axis=1, ddof=1)
    denom = noise_sigma * math.sqrt(max(n_burst_samples, 1))
    snr = np.divide(signal, denom, out=np.zeros_like(signal), where=denom > 0)
    weights = np.where(np.isfinite(snr) & (snr > 0), snr, 0.0)

    if np.nansum(weights) <= 0:
        weights = np.where(np.isfinite(signal) & (signal > 0), signal, 0.0)
    if np.nansum(weights) <= 0:
        return float(np.nanmean(freq) / 1000.0)
    return float(np.nansum(freq * weights) / np.nansum(weights) / 1000.0)


def i_weighted_center_ghz(
    frequency_mhz: np.ndarray,
    i_sub: np.ndarray,
    freq_sl: slice,
    burst_sl: slice,
) -> float:
    freq = frequency_mhz[freq_sl]
    signal = np.nansum(i_sub[freq_sl, burst_sl], axis=1)
    if np.nansum(signal) == 0:
        return float(np.nanmean(freq) / 1000.0)
    return float(np.nansum(freq * signal) / np.nansum(signal) / 1000.0)


def compute_one(row: pd.Series, data_dir: Path) -> dict[str, Any]:
    raw_path = data_dir.parent / str(row["raw_h5_path"])
    burst_name = str(row["burst_name"])

    with h5py.File(raw_path, "r") as handle:
        frequency_mhz = handle["frequency"][:]
        group = handle[burst_name]
        i_arr = group["I"][:]
        q_arr = group["Q"][:]
        u_arr = group["U"][:]
        v_arr = group["V"][:]
        rm_value = as_float(group.attrs.get("rm_value"))

    ns, ne = row["noise_start_idx"], row["noise_end_idx"]
    bs, be = row["burst_start_idx"], row["burst_end_idx"]
    fs, fe = row["freq_start_idx"], row["freq_end_idx"]
    noise_sl = slice(ns, ne + 1)
    burst_sl = slice(bs, be + 1)
    freq_sl = slice(fs, fe + 1)
    n_burst = be - bs + 1

    i_sub = background_subtract_per_frequency(i_arr, noise_sl)
    q_sub = background_subtract_per_frequency(q_arr, noise_sl)
    u_sub = background_subtract_per_frequency(u_arr, noise_sl)
    v_sub = background_subtract_per_frequency(v_arr, noise_sl)

    i_profile = np.nanmean(i_sub[freq_sl, :], axis=0)
    q_profile = np.nanmean(q_sub[freq_sl, :], axis=0)
    u_profile = np.nanmean(u_sub[freq_sl, :], axis=0)
    v_profile = np.nanmean(v_sub[freq_sl, :], axis=0)

    sigma_i = float(np.nanstd(i_profile[noise_sl], ddof=1))
    l_debiased, l_measured = debiased_linear_profile(q_profile, u_profile, sigma_i)

    i_sum = float(np.nansum(i_profile[burst_sl]))
    l_debiased_sum = float(np.nansum(l_debiased[burst_sl]))
    l_measured_sum = float(np.nansum(l_measured[burst_sl]))
    v_sum = float(np.nansum(v_profile[burst_sl]))

    linear = l_debiased_sum / i_sum if i_sum != 0 else math.nan
    linear_naive = l_measured_sum / i_sum if i_sum != 0 else math.nan
    circular = v_sum / i_sum if i_sum != 0 else math.nan

    linear_error = (
        math.sqrt(n_burst + n_burst * linear * linear) * sigma_i / i_sum
        if i_sum != 0 and np.isfinite(linear)
        else math.nan
    )
    circular_error = (
        math.sqrt(n_burst + n_burst * circular * circular) * sigma_i / i_sum
        if i_sum != 0 and np.isfinite(circular)
        else math.nan
    )

    center_snr = snr_weighted_center_ghz(
        frequency_mhz, i_sub, freq_sl, noise_sl, burst_sl, n_burst
    )
    center_i = i_weighted_center_ghz(frequency_mhz, i_sub, freq_sl, burst_sl)
    center_mid = float((frequency_mhz[fs] + frequency_mhz[fe]) / 2000.0)
    total_snr = i_sum / (sigma_i * math.sqrt(n_burst)) if sigma_i > 0 else math.nan

    return {
        "analysis_order": row["analysis_order"],
        "group_name": row["group_name"],
        "FRB": row["FRB"],
        "telescope": row["telescope"],
        "paper_burst_number": row["paper_burst_number"],
        "source_file": row["source_file"],
        "burst_name": burst_name,
        "rm_value_rad_m2": rm_value,
        "center_frequency_ghz": center_snr,
        "center_frequency_mhz": center_snr * 1000.0,
        "center_frequency_ghz_i_weighted": center_i,
        "center_frequency_ghz_freq_midpoint": center_mid,
        "linear_polarization": linear,
        "linear_polarization_percent": linear * 100.0,
        "linear_polarization_error": linear_error,
        "linear_polarization_error_percent": linear_error * 100.0,
        "linear_polarization_naive": linear_naive,
        "circular_polarization": circular,
        "circular_polarization_error": circular_error,
        "total_snr": total_snr,
        "sigma_i": sigma_i,
        "i_sum": i_sum,
        "noise_start_idx": ns,
        "noise_end_idx": ne,
        "burst_start_idx": bs,
        "burst_end_idx": be,
        "freq_start_idx": fs,
        "freq_end_idx": fe,
        "freq_start_mhz": row["freq_start_mhz"],
        "freq_end_mhz": row["freq_end_mhz"],
        "time_resolution_s": row["time_resolution_s"],
    }


def build_comparison(df_derived: pd.DataFrame, df_reference: pd.DataFrame) -> pd.DataFrame:
    if df_reference.empty:
        return pd.DataFrame()

    paper_columns = [
        "group_name",
        "paper_frequency_ghz",
        "paper_linear_polarization",
        "paper_linear_polarization_error",
        "paper_circular_polarization",
        "paper_circular_polarization_error",
    ]
    df = df_derived.merge(df_reference[paper_columns], on="group_name", how="inner")
    for column in paper_columns[1:]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    delta_linear = df["linear_polarization"] - df["paper_linear_polarization"]
    combined_linear_error = np.sqrt(
        df["linear_polarization_error"] ** 2
        + df["paper_linear_polarization_error"] ** 2
    )

    return pd.DataFrame(
        {
            "analysis_order": df["analysis_order"],
            "group_name": df["group_name"],
            "FRB": df["FRB"],
            "telescope": df["telescope"],
            "burst_name": df["burst_name"],
            "derived_frequency_ghz": df["center_frequency_ghz"],
            "paper_frequency_ghz": df["paper_frequency_ghz"],
            "delta_frequency_ghz": df["center_frequency_ghz"] - df["paper_frequency_ghz"],
            "derived_linear_polarization": df["linear_polarization"],
            "paper_linear_polarization": df["paper_linear_polarization"],
            "delta_linear_polarization": delta_linear,
            "derived_linear_polarization_error": df["linear_polarization_error"],
            "paper_linear_polarization_error": df["paper_linear_polarization_error"],
            "delta_linear_sigma": np.divide(
                delta_linear,
                combined_linear_error,
                out=np.full(len(df), np.nan),
                where=combined_linear_error > 0,
            ),
            "derived_circular_polarization": df["circular_polarization"],
            "paper_circular_polarization": df["paper_circular_polarization"],
            "delta_circular_polarization": (
                df["circular_polarization"] - df["paper_circular_polarization"]
            ),
            "derived_circular_polarization_error": df["circular_polarization_error"],
            "paper_circular_polarization_error": df[
                "paper_circular_polarization_error"
            ],
        }
    )


def make_plot(
    derived_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    colors = {
        "FRB20190303A": "#fee090",
        "FRB20190417A": "#ffffbf",
        "FRB20190520B": "#313695",
        "FRB20201124A": "#fdae61",
    }
    markers = {"FAST": "o", "GBT": "D"}

    fig, ax = plt.subplots(figsize=(8, 5.2))
    for row in derived_rows:
        ax.errorbar(
            row["center_frequency_ghz"],
            row["linear_polarization"],
            yerr=row["linear_polarization_error"],
            fmt=markers.get(row["telescope"], "o"),
            ms=7,
            color=colors.get(row["FRB"], "0.4"),
            markeredgecolor="k",
            ecolor="0.55",
            capsize=2,
            alpha=0.85,
        )

    for row in comparison_rows:
        ax.scatter(
            row["paper_frequency_ghz"],
            row["paper_linear_polarization"],
            marker="x",
            s=42,
            color=colors.get(row["FRB"], "0.4"),
            linewidths=1.4,
            alpha=0.85,
        )

    ax.set_xscale("log")
    ax.set_xlim(0.7, 7.0)
    ax.set_ylim(-0.05, 1.35)
    ax.set_xlabel("Center frequency (GHz)")
    ax.set_ylabel("Degree of linear polarization")
    ax.set_title("Raw-derived points (filled) vs Table S3 reference (x)")
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir",
        type=str,
        default=REPO_ROOT / "data" / "handoff_sigmaRM_ai_reproduction" / "data",
        help="Input data directory containing selection_windows.csv, raw/, and paper_table_s3_local_reference.csv.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=REPO_ROOT / "logs" / "scripts" / "astronomy" / "calculate_points",
        help="Output directory for derived CSV files and comparison plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    # read table rows
    df_selection = pd.read_csv(data_dir / "selection_windows.csv")
    assert all(c in df_selection.columns for c in INT_FIELDS + FLOAT_FIELDS)
    df_selection[INT_FIELDS] = df_selection[INT_FIELDS].astype("int64")
    df_selection[FLOAT_FIELDS] = df_selection[FLOAT_FIELDS].astype("float64")

    df_list = []
    for idx, row in df_selection.iterrows():
        derived_row = compute_one(row, data_dir)
        df_list.append(derived_row)
    df_derived = pd.DataFrame(df_list, index=df_selection.index)
    df_derived = df_derived.sort_values("analysis_order")
    df_derived.to_csv(save_dir / "derived_points.csv", index=False)
    print(f"Wrote {save_dir / 'derived_points.csv'} ({len(df_derived)} rows)")

    df_plot = df_derived.rename(columns={
        "telescope": "Telescope",
        "center_frequency_ghz": "Frequency",
        "linear_polarization": "Linear_Polarization",
        "linear_polarization_error": "Linear_Polarization_Error",
    })[[
        "FRB", "Telescope", "Frequency",
        "Linear_Polarization", "Linear_Polarization_Error",
        "group_name", "burst_name",
    ]]
    df_plot.to_csv(save_dir / "derived_points_plot_columns.csv", index=False)

    derived_rows = df_derived.to_dict(orient="records")
    if (reference_path := data_dir / "paper_table_s3_local_reference.csv").exists():
        df_reference = pd.read_csv(reference_path)
        df_comparison = build_comparison(df_derived, df_reference)
        df_comparison.to_csv(save_dir / "paper_table_s3_comparison.csv", index=False)
        comparison_rows = df_comparison.to_dict(orient="records")
        make_plot(derived_rows, comparison_rows, save_dir / "derived_vs_reference.png")

        max_df = df_comparison["delta_frequency_ghz"].abs().max()
        max_dl = df_comparison["delta_linear_polarization"].abs().max()
        print(f"Wrote {save_dir / 'paper_table_s3_comparison.csv'}")
        print(f"Max abs frequency delta vs Table S3: {max_df:.4f} GHz")
        print(f"Max abs linear-polarization delta vs Table S3: {max_dl:.4f}")


if __name__ == "__main__":
    main()
