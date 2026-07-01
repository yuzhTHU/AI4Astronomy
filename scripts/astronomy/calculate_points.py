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
import os
import math
import h5py
import logging
import argparse
import numpy as np
import pandas as pd
from typing import Any
from pathlib import Path

SCRIPT_NAME = Path(__file__).stem
_logger = logging.getLogger(f'sr_agent.{__name__}')

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


def background_subtract_per_frequency(
    arr: np.ndarray,
    noise_sl: slice,
) -> np.ndarray:
    """按频率通道扣除噪声窗口中的基线均值。"""
    baseline = np.nanmean(arr[:, noise_sl], axis=1, keepdims=True)
    return arr - baseline


def debiased_linear_profile(
    q_profile: np.ndarray,
    u_profile: np.ndarray,
    sigma_i: float,
    threshold: float = 1.57,
) -> tuple[np.ndarray, np.ndarray]:
    """根据 Q/U 时间序列计算去噪声偏置和未去偏置的线偏振强度。"""
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
    """用 Stokes-I 信噪比作为权重计算中心频率。"""
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
    """用 Stokes-I 积分强度作为权重计算中心频率。"""
    freq = frequency_mhz[freq_sl]
    signal = np.nansum(i_sub[freq_sl, burst_sl], axis=1)
    if np.nansum(signal) == 0:
        return float(np.nanmean(freq) / 1000.0)
    return float(np.nansum(freq * signal) / np.nansum(signal) / 1000.0)


def compute_one(row: pd.Series, data_dir: Path) -> dict[str, Any]:
    with h5py.File(data_dir.parent / str(row["raw_h5_path"]), "r") as handle:
        frequency_mhz = handle["frequency"][:]
        group = handle[row["burst_name"]]
        i_arr = group["I"][:]
        q_arr = group["Q"][:]
        u_arr = group["U"][:]
        v_arr = group["V"][:]
        rm_value = float(group.attrs.get("rm_value"))

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
        "analysis_order": row["analysis_order"],                    # 分析顺序
        "group_name": row["group_name"],                            # 唯一标识
        "FRB": row["FRB"],                                          # FRB 名称
        "telescope": row["telescope"],                              # 望远镜
        "paper_burst_number": row["paper_burst_number"],            # 论文 burst 编号
        "source_file": row["source_file"],                          # 原始文件名
        "burst_name": row["burst_name"],                            # HDF5 burst 名
        "rm_value_rad_m2": rm_value,                                # 旋转量
        "center_frequency_ghz": center_snr,                         # S/N 加权中心频率
        "center_frequency_mhz": center_snr * 1000.0,                # 中心频率 MHz
        "center_frequency_ghz_i_weighted": center_i,                # I 加权中心频率
        "center_frequency_ghz_freq_midpoint": center_mid,           # 频窗中点
        "linear_polarization": linear,                              # 去偏置线偏振度
        "linear_polarization_percent": linear * 100.0,              # 线偏振百分比
        "linear_polarization_error": linear_error,                  # 线偏振误差
        "linear_polarization_error_percent": linear_error * 100.0,  # 线偏振误差百分比
        "linear_polarization_naive": linear_naive,                  # 未去偏置线偏振度
        "circular_polarization": circular,                          # 圆偏振度
        "circular_polarization_error": circular_error,              # 圆偏振误差
        "total_snr": total_snr,                                     # 总信噪比
        "sigma_i": sigma_i,                                         # I 噪声标准差
        "i_sum": i_sum,                                             # I 积分强度
        "noise_start_idx": ns,                                      # 噪声起点
        "noise_end_idx": ne,                                        # 噪声终点
        "burst_start_idx": bs,                                      # burst 起点
        "burst_end_idx": be,                                        # burst 终点
        "freq_start_idx": fs,                                       # 频率起点
        "freq_end_idx": fe,                                         # 频率终点
        "freq_start_mhz": row["freq_start_mhz"],                    # 起始频率 MHz
        "freq_end_mhz": row["freq_end_mhz"],                        # 结束频率 MHz
        "time_resolution_s": row["time_resolution_s"],              # 时间分辨率
    }


def plot_selection_window(row: pd.Series, data_dir: Path, output_dir: Path) -> Path:
    """绘制单个 burst 的 Stokes-I 时频窗口审查图。"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    raw_path = data_dir.parent / str(row["raw_h5_path"])
    burst_name = str(row["burst_name"])
    with h5py.File(raw_path, "r") as handle:
        freq = handle["frequency"][:]
        stokes_i = handle[f"{burst_name}/I"][:]
        time_resolution = float(handle.attrs.get("time_resolution", row["time_resolution_s"]))

    n_time = stokes_i.shape[1]
    time_ms = np.arange(n_time) * time_resolution * 1000.0
    i_profile = np.nanmean(stokes_i, axis=0)
    vmin, vmax = np.nanpercentile(stokes_i, [5, 95])

    ns, ne = row["noise_start_idx"], row["noise_end_idx"]
    bs, be = row["burst_start_idx"], row["burst_end_idx"]
    fs, fe = row["freq_start_idx"], row["freq_end_idx"]


    fig, axes = plt.subplots(2, 1, figsize=(11, 6.4), height_ratios=[1, 2], gridspec_kw={"hspace": 0.05}, sharex=True)
    fig.suptitle(f"{row['analysis_order']:02d} {row['group_name']} / {burst_name}", fontsize=11, fontweight="bold")

    axes[0].plot(time_ms, i_profile, color="0.15", lw=0.8)
    axes[0].axvspan(time_ms[ns], time_ms[ne], color="#7aa6ff", alpha=0.22, label="noise")
    axes[0].axvspan(time_ms[bs], time_ms[be], color="#ff6b6b", alpha=0.24, label="burst")
    axes[0].set_ylabel("mean I")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].imshow(
        stokes_i,
        aspect="auto", origin="lower", cmap="viridis",
        extent=[time_ms[0], time_ms[-1], freq[0], freq[-1]],
        vmin=vmin, vmax=vmax,
    )
    for idx, color in [(ns, "#2f66ff"), (ne, "#2f66ff"), (bs, "#d62728"), (be, "#d62728")]:
        axes[1].axvline(time_ms[idx], color=color, ls="--", lw=1.1, alpha=0.8)
    axes[1].axhline(freq[fs], color="#66ff66", ls="--", lw=1.1)
    axes[1].axhline(freq[fe], color="#66ff66", ls="--", lw=1.1)
    axes[1].axhspan(freq[fs], freq[fe], color="#66ff66", alpha=0.12)
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("Frequency (MHz)")

    text = (
        f"time burst [{bs}, {be}], noise [{ns}, {ne}], "
        f"freq [{fs}, {fe}] = {freq[fs]:.1f}-{freq[fe]:.1f} MHz"
    )
    axes[1].text(
        0.01, 0.02, text,
        transform=axes[1].transAxes, fontsize=8, color="white",
        bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none", "pad": 3},
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{row['analysis_order']:02d}_{row['group_name']}_selection.png"
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


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
    combined_linear_error = np.sqrt(df["linear_polarization_error"] ** 2 + df["paper_linear_polarization_error"] ** 2)
    df['delta_linear_sigma'] = np.divide(
        delta_linear, combined_linear_error,
        out=np.full(len(df), np.nan),
        where=combined_linear_error > 0,
    )

    return pd.DataFrame({
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
        "delta_linear_sigma": df['delta_linear_sigma'],

        "derived_circular_polarization": df["circular_polarization"],
        "paper_circular_polarization": df["paper_circular_polarization"],
        "delta_circular_polarization": df["circular_polarization"] - df["paper_circular_polarization"],

        "derived_circular_polarization_error": df["circular_polarization_error"],
        "paper_circular_polarization_error": df["paper_circular_polarization_error"],
    })


def make_reference_plot(
    df_derived: pd.DataFrame,
    df_comparison: pd.DataFrame,
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
    for _, row in df_derived.iterrows():
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

    for _, row in df_comparison.iterrows():
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
    parser.add_argument("--data_dir", type=str, default="./data/handoff_sigmaRM_ai_reproduction/data", help="Input data directory containing selection_windows.csv, raw/, and paper_table_s3_local_reference.csv.")
    parser.add_argument("--save_dir", type=str, default=f"./logs/scripts/astronomy/{SCRIPT_NAME}", help="Output directory for derived CSV files and comparison plot.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    # Derived points
    df_selection = pd.read_csv(data_dir / "selection_windows.csv")
    assert all(c in df_selection.columns for c in INT_FIELDS + FLOAT_FIELDS)
    df_selection[INT_FIELDS] = df_selection[INT_FIELDS].astype("int64")
    df_selection[FLOAT_FIELDS] = df_selection[FLOAT_FIELDS].astype("float64")
    df_list = []
    for _, row in df_selection.iterrows():
        derived_row = compute_one(row, data_dir)
        df_list.append(derived_row)
    df_derived = pd.DataFrame(df_list, index=df_selection.index)
    df_derived = df_derived.sort_values("analysis_order")
    df_derived.to_csv(save_dir / "derived_points.csv", index=False)
    _logger.info(f"Wrote {save_dir / 'derived_points.csv'} ({len(df_derived)} rows)")

    # Plot
    for _, row in df_selection.sort_values("analysis_order").iterrows():
        plot_path = plot_selection_window(row, data_dir, save_dir / "selection_plots")
        _logger.info(f"Plotted selection window for {row['group_name']} / {row['burst_name']} to {plot_path}")

    # Compare with reference table (if available)
    if (reference_path := data_dir / "paper_table_s3_local_reference.csv").exists():
        df_reference = pd.read_csv(reference_path)
        df_comparison = build_comparison(df_derived, df_reference)
        df_comparison.to_csv(save_dir / "paper_table_s3_comparison.csv", index=False)
        make_reference_plot(df_derived, df_comparison, save_dir / "derived_vs_reference.png")

        max_df = df_comparison["delta_frequency_ghz"].abs().max()
        max_dl = df_comparison["delta_linear_polarization"].abs().max()
        _logger.info(f"Wrote {save_dir / 'paper_table_s3_comparison.csv'}")
        _logger.info(f"Max abs frequency delta vs Table S3: {max_df:.4f} GHz")
        _logger.info(f"Max abs linear-polarization delta vs Table S3: {max_dl:.4f}")


if __name__ == "__main__":
    main()
