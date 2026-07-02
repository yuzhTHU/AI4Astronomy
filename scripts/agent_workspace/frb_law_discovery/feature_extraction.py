#!/usr/bin/env python3
"""Extract FRB polarization features from local HDF5 data.

Inputs:
  ./data/selection_windows.csv
  ./data/raw/*_polarization.h5

Output:
  ./saved/<exp_name>/output.csv
  ./saved/<exp_name>/description.json
  ./saved/<exp_name>/feature_extraction.py
  ./saved/<exp_name>/info.log
  ./saved/<exp_name>/args.json

Index convention in selection_windows.csv:
  Stokes arrays are [frequency, time].
  start/end indices are inclusive endpoints.
"""

from __future__ import annotations
import sys
import h5py
import math
import json
import shlex
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Any
from pathlib import Path
from datetime import datetime

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

##############################
# 以下内容可修改
##############################

def build_argparser() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment task name.")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed. Default -1 means using current system time.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose agent logging.")
    parser.add_argument("--data-dir", type=Path, default=script_dir / "data", help="Input directory containing selection_windows.csv and raw/*.h5.")
    parser.add_argument("--save-dir", type=Path, default=script_dir / "saved", help="Directory to save experiment results.")
    return parser


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


def compute_one(args, row: pd.Series) -> dict[str, Any]:
    with h5py.File(Path(args.data_dir) / row["raw_h5_path"].removeprefix('data/'), "r") as handle:
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
        "analysis_order": row["analysis_order"],
        "group_name": row["group_name"],
        "FRB": row["FRB"],
        "telescope": row["telescope"],
        "paper_burst_number": row["paper_burst_number"],
        "source_file": row["source_file"],
        "burst_name": row["burst_name"],
        "freq_start_mhz": row["freq_start_mhz"],
        "freq_end_mhz": row["freq_end_mhz"],
        "time_resolution_s": row["time_resolution_s"],
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
    }

def post_process(args, df: pd.DataFrame) -> pd.DataFrame:
    return df


FEATURE_DESCRIPTION: dict[str, str] = {
    "analysis_order": "Row order used in the analysis selection table.",
    "group_name": "Unique identifier for the selected FRB burst/window group.",
    "FRB": "FRB source name.",
    "telescope": "Telescope used for the observation.",
    "paper_burst_number": "Burst number used by the reference paper/table.",
    "source_file": "Name of the raw HDF5 file containing the burst.",
    "burst_name": "HDF5 group name for the burst.",
    "freq_start_mhz": "Lower frequency edge of the selected analysis window, in MHz.",
    "freq_end_mhz": "Upper frequency edge of the selected analysis window, in MHz.",
    "time_resolution_s": "Time resolution of the Stokes dynamic spectrum, in seconds.",
    "rm_value_rad_m2": "Rotation measure stored in the raw HDF5 burst group attributes, in rad m^-2.",
    "center_frequency_ghz": "S/N-weighted center frequency across the selected frequency window, in GHz.",
    "center_frequency_mhz": "S/N-weighted center frequency across the selected frequency window, in MHz.",
    "center_frequency_ghz_i_weighted": "Stokes-I-integrated-intensity-weighted center frequency, in GHz.",
    "center_frequency_ghz_freq_midpoint": "Arithmetic midpoint of the selected frequency window, in GHz.",
    "linear_polarization": "Debiased linear polarization fraction integrated over the burst window.",
    "linear_polarization_percent": "Debiased linear polarization percentage integrated over the burst window.",
    "linear_polarization_error": "Estimated uncertainty of the debiased linear polarization fraction.",
    "linear_polarization_error_percent": "Estimated uncertainty of the debiased linear polarization percentage.",
    "linear_polarization_naive": "Non-debiased linear polarization fraction integrated over the burst window.",
    "circular_polarization": "Circular polarization fraction integrated over the burst window.",
    "circular_polarization_error": "Estimated uncertainty of the circular polarization fraction.",
    "total_snr": "Integrated burst signal-to-noise ratio from Stokes I.",
    "sigma_i": "Noise-window standard deviation of the frequency-averaged Stokes-I profile.",
    "i_sum": "Integrated background-subtracted Stokes-I signal over the burst window.",
    "noise_start_idx": "Inclusive start time index of the noise window.",
    "noise_end_idx": "Inclusive end time index of the noise window.",
    "burst_start_idx": "Inclusive start time index of the burst window.",
    "burst_end_idx": "Inclusive end time index of the burst window.",
    "freq_start_idx": "Inclusive start frequency-channel index of the analysis window.",
    "freq_end_idx": "Inclusive end frequency-channel index of the analysis window.",
}

##############################
# 以上内容可修改
##############################

def main(args) -> None:
    # Load Data
    data_dir = args.data_dir.resolve()
    selection_path = data_dir / "selection_windows.csv"
    df_selection = pd.read_csv(selection_path)
    if missing := [c for c in INT_FIELDS + FLOAT_FIELDS if c not in df_selection.columns]:
        raise ValueError(f"Missing required columns in {selection_path}: {missing}")
    else:
        df_selection[INT_FIELDS] = df_selection[INT_FIELDS].astype("int64")
        df_selection[FLOAT_FIELDS] = df_selection[FLOAT_FIELDS].astype("float64")

    # Extract Features
    rows = [compute_one(args, row) for _, row in tqdm(df_selection.iterrows(), total=len(df_selection))]
    df_data = pd.DataFrame(rows, index=df_selection.index)
    df_data = post_process(args, df_data)

    # Save Output
    output_path = Path(args.save_path) / "output.csv"
    df_data.to_csv(output_path, index=False)
    _logger.info(f"Saved output with {len(df_data)} rows and {len(df_data.columns)} columns to {output_path}")

    description = {name: FEATURE_DESCRIPTION.get(name, "(No Description)") for name in df_data.columns}
    description_path = Path(args.save_path) / "description.json"
    with description_path.open("w", encoding="utf-8") as f:
        json.dump(description, f, ensure_ascii=False, indent=2)
        f.write("\n")
    if missing := [name for name in df_data.columns if name not in FEATURE_DESCRIPTION]:
        missing_text = f"missing descriptions for columns: {missing}"
    else:
        missing_text = "all columns have descriptions."
    _logger.info(f"Saved feature description with {len(description)} entries to {description_path}, {missing_text}")

    # Save this script
    this_script = Path(__file__).read_text()
    script_path = Path(args.save_path) / Path(__file__).name
    script_path.write_text(this_script)
    _logger.info(f"Saved script with {script_path.stat().st_size:,} bytes to {script_path}")


if __name__ == "__main__":
    from sr_agent.utils import setup_logging, save_args, seed_all, tag2ansi

    parser = build_argparser()
    args, unknown = parser.parse_known_args()
    if args.exp_name is None:
        now = datetime.now()
        args.exp_name = f"{now:%Y%m%d}_{SCRIPT_NAME}_{now:%H%M%S}"
    if args.seed == -1:
        args.seed = int(datetime.now().timestamp() * 1000) % (2**32 - 1)
    args.save_path = Path(args.save_dir) / args.exp_name
    if args.save_path.exists():
        raise FileExistsError(f"Save path already exists: {args.save_path}, please choose a different name to prevent overwriting previous results.")
    else:
        args.save_path.mkdir(parents=True, exist_ok=True)
    args.command = " ".join(map(shlex.quote, [sys.executable, *sys.argv]))
    setup_logging(
        info_level="debug" if args.verbose else "info",
        exp_name=args.exp_name,
        save_path=args.save_path / "info.log",
        force=True,
    )
    if unknown:
        _logger.warning(f"Unknown args: {unknown}")
    _logger.note(f"Args: {args}")
    save_args(args, args.save_path / "args.json")
    seed_all(args.seed)
    main(args)
    _logger.note(tag2ansi(f"Experiment completed. Re-run the script with [green bold]{args.command}[reset]"))
