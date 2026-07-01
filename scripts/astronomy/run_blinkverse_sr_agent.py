#!/usr/bin/env python3
"""Prepare Blinkverse symbolic-regression tasks and optionally run SRAgent.

The Blinkverse catalog is observational astronomy data, not a synthetic
``y = f(x)`` benchmark.  This adapter turns a few paper-motivated questions
into the ``SRAgent.fit(X, y, problem_description)`` interface.
"""

from __future__ import annotations
import re
import sys
import json
import shlex
import dotenv
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sr_agent import SRAgent
from datetime import datetime
from socket import gethostname
from sr_agent.tools import BaseTool
from sr_agent.utils import setup_logging, add_minus_flags, add_negation_flags, seed_all, log_exception, tag2ansi, sanitize_filename, save_args


SCRIPT_NAME = Path(__file__).stem
_logger = logging.getLogger(f"sr_agent.{SCRIPT_NAME}")
dotenv.load_dotenv()


TASKS = {
    "nature2021_energy_distribution": "Fit the FRB20121102A burst-energy density as a function of log10 energy.",
    "nature2021_waiting_time": "Fit the FRB20121102A waiting-time density as a function of log10 waiting time.",
    "nature2021_dm_evolution": "Fit FRB20121102A dispersion measure as a function of observation time.",
    "science2022_polarization_frequency": "Explore how linear polarization fraction depends on observing frequency and propagation tracers.",
}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SRAgent on BlinkVerse catalog-derived tasks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--name", default=f"{SCRIPT_NAME}", help="Experiment task name used when auto-generating exp_name.")
    parser.add_argument("--exp_name", default=None, help="Experiment name. Defaults to a timestamped name.")
    parser.add_argument("--save_dir", default=f"./logs/{SCRIPT_NAME}", help="Root directory for logs and run artifacts.")
    parser.add_argument("--task", choices=sorted(TASKS), default="nature2021_energy_distribution")
    parser.add_argument("--data_dir", type=str, default="data/blinkverse")
    parser.add_argument("--bins", type=int, default=40, help="Histogram bins for distribution tasks.")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed. Default -1 means using current system time.")
    parser.add_argument("--llm_provider", default="openrouter", help="LLM provider name.")
    parser.add_argument("--llm_model", default="deepseek/deepseek-v4-flash", help="LLM model name.")
    parser.add_argument("--tools", default=BaseTool.all_registered_names, type=str, nargs='+', help="Optional list of tools to use. Default is all built-in tools.")
    parser.add_argument("-K", "--local_sample_size", type=int, default=2, help="Number of LLM samples to generate for each branch.")
    parser.add_argument("-L", "--max_refinement_depth", type=int, default=5, help="Maximum agent refinement depth.")
    parser.add_argument("-C", "--global_width", type=int, default=2, help="Number of independent branches per restart loop.")
    parser.add_argument("-R", "--max_restart_loop", type=int, default=2, help="Maximum number of best-solution restart loops.")
    parser.add_argument("--restart_top_k", type=int, default=1, help="Number of previous best formulas to inject into the next restart prompt.")
    parser.add_argument("--tool_parser", default="openai", choices=["openai", "text", "json", "xml"], help="Tool response parser type.")
    parser.add_argument("--save_path", default=None, help="Path to save agent logs and artifacts. Default is auto-generated from --save_dir and --exp_name.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose agent logging.")
    parser.add_argument("--debug", action="store_true", default=True, help="Enable debug mode (verbose + raise caught exceptions).")
    parser.add_argument("--max_workers", type=int, default=0, help="Maximum number of parallel workers for tool execution. 0 means no parallel execution.")
    parser = add_minus_flags(parser)
    parser = add_negation_flags(parser)
    return parser


def as_numeric(series: pd.Series) -> pd.Series:
    """Convert Blinkverse scalar-like fields to numeric values."""
    return pd.to_numeric(series.replace({"null": np.nan, "None": np.nan, "": np.nan}), errors="coerce")


def band_midpoint_mhz(series: pd.Series) -> pd.Series:
    """Parse values such as '1000-1500' into their midpoint in MHz."""
    values = []
    for item in series.fillna("").astype(str):
        match = re.search(r"([0-9.]+)\s*-\s*([0-9.]+)", item)
        if match:
            low, high = float(match.group(1)), float(match.group(2))
            values.append((low + high) / 2.0)
        else:
            values.append(np.nan)
    return pd.Series(values, index=series.index)


def finite_dict(data: dict[str, np.ndarray], target: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    mask = np.ones(len(next(iter(data.values()))), dtype=bool)
    for values in data.values():
        mask &= np.isfinite(values)
    cleaned = {name: np.asarray(values, dtype=float)[mask] for name, values in data.items()}
    if len(cleaned[target]) == 0:
        raise ValueError(f"No finite rows remain for target {target!r}.")
    X = {name: values for name, values in cleaned.items() if name != target}
    y = {target: cleaned[target]}
    return X, y


def histogram_task(values: np.ndarray, x_name: str, y_name: str, bins: int, log_values=True):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if log_values:
        values = np.log10(values)
    if len(values) < max(10, bins):
        raise ValueError(f"Not enough positive finite values for histogram task: {len(values)}")
    counts, edges = np.histogram(values, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mask = np.isfinite(counts) & (counts > 0)
    return {x_name: centers[mask]}, {y_name: counts[mask]}


def energy_rate_task(values: np.ndarray, bins: int, observing_hours: float = 59.5):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) < max(10, bins):
        raise ValueError(f"Not enough positive finite energy values for Figure 2 task: {len(values)}")
    log_edges = np.linspace(np.log10(values.min()), np.log10(values.max()), bins + 1)
    edges = 10 ** log_edges
    counts, _ = np.histogram(values, bins=edges)
    centers = np.sqrt(edges[:-1] * edges[1:])
    rate = counts / observing_hours
    mask = np.isfinite(rate) & (rate > 0)
    return {"energy_erg": centers[mask]}, {"burst_rate_h_per_energy_bin": rate[mask]}


def build_task(args):
    path = Path(args.data_dir) / "analysis_single.flattened.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing Blinkverse burst table: {path}")
    bursts = pd.read_csv(path, low_memory=False)

    if args.task == "nature2021_energy_distribution":
        df = bursts[(bursts["source"] == "FRB20121102A") & (bursts["telescope"] == "FAST")].copy()
        energy = as_numeric(df["energy"]).to_numpy(dtype=float)
        X, y = energy_rate_task(energy, bins=args.bins)
        problem = (
            "Nature 2021 reported that FAST bursts from repeating source FRB20121102A "
            "have a bimodal isotropic-equivalent energy distribution.  The data here "
            "follows the paper's Figure 2 bottom-panel setup: the feature energy_erg "
            "is the geometric center of a logarithmic energy bin in erg, and the target "
            "burst_rate_h_per_energy_bin is the number of bursts in that bin divided by "
            "the 59.5 h FAST observing time.  Fit a simple, interpretable formula "
            "burst_rate_h_per_energy_bin = f(energy_erg).  A good expression should "
            "reveal whether one component is insufficient.  If a base-10 logarithm is "
            "useful, write it as log(energy_erg) / log(10)."
        )
        return X, y, problem
    elif args.task == "nature2021_waiting_time":
        df = bursts[(bursts["source"] == "FRB20121102A") & (bursts["telescope"] == "FAST")].copy()
        mjd = as_numeric(df["mjd"]).dropna().sort_values().to_numpy(dtype=float)
        wait_s = np.diff(mjd) * 86400.0
        wait_s = wait_s[(wait_s > 0) & (wait_s < 3600.0)]
        X, y = histogram_task(wait_s, x_name="log10_wait_s", y_name="waiting_time_density", bins=args.bins)
        problem = (
            "Nature 2021 studied waiting times between adjacent bursts of FRB20121102A "
            "and found a broad log-normal-like distribution rather than a stable clock. "
            "Fit waiting_time_density = f(log10_wait_s) using the Blinkverse FAST subset. "
            "The 3600 s cut avoids day-scale observing gaps."
        )
        return X, y, problem
    elif args.task == "nature2021_dm_evolution":
        df = bursts[bursts["source"] == "FRB20121102A"].copy()
        dm = as_numeric(df["dm_alig"]).fillna(as_numeric(df["dm_snr"]))
        mjd = as_numeric(df["mjd"]).fillna(as_numeric(df["mjd_inf"]))
        data = {
            "year_since_mjd_57000": (mjd.to_numpy(dtype=float) - 57000.0) / 365.25,
            "dm_pc_cm3": dm.to_numpy(dtype=float),
        }
        X, y = finite_dict(data, target="dm_pc_cm3")
        problem = (
            "Nature 2021 reported a long-term increase of dispersion measure for "
            "FRB20121102A.  Fit dm_pc_cm3 = f(year_since_mjd_57000). "
            "Start with linear trends, then check whether extra complexity is justified."
        )
        return X, y, problem
    elif args.task == "science2022_polarization_frequency":
        df = bursts[bursts["type"].isin(["REPEAT", "REPEAT_FIRST"])].copy()
        freq_mhz = as_numeric(df["freq_peak"]).fillna(band_midpoint_mhz(df["observing_band"]))
        polar_l = as_numeric(df["polar_l"])
        rm = as_numeric(df["rm_syn"]).fillna(as_numeric(df["rm_qufit"]))
        scatt_t = as_numeric(df["scatt_t"])
        data = {
            "freq_ghz": freq_mhz.to_numpy(dtype=float) / 1000.0,
            "wavelength_m": 0.299792458 / (freq_mhz.to_numpy(dtype=float) / 1000.0),
            "abs_rm_rad_m2": np.abs(rm.to_numpy(dtype=float)),
            "scatt_t_ms": scatt_t.to_numpy(dtype=float),
            "linear_polarization_fraction": polar_l.to_numpy(dtype=float) / 100.0,
        }
        X, y = finite_dict(data, target="linear_polarization_fraction")
        problem = (
            "Science 2022 argued that repeating FRBs become less linearly polarized "
            "at lower observing frequencies, consistent with RM-scattering "
            "depolarization.  Explore linear_polarization_fraction = f(freq_ghz, "
            "wavelength_m, abs_rm_rad_m2, scatt_t_ms).  Treat this as a catalog-level "
            "hypothesis-generation task, not a full reproduction of the paper's "
            "source-level sRM fit."
        )
        return X, y, problem
    else:
        raise ValueError(f"Unknown task {args.task!r}. Choose one of: {', '.join(TASKS)}")


def main(args):
    X, y, problem = build_task(args)
    features = list(X.keys())
    target = next(iter(y))
    context_path = Path(args.save_path) / "context.npz"
    np.savez(
        context_path, 
        data=(X | y), 
        target=target, 
        problem_description=problem
    )
    _logger.note(
        f"Prepared BlinkVerse task: "
        f"task={args.task}, "
        f"n={len(y[target])}, "
        f"features={features!r}, "
        f"target={target!r}. "
        f"Context saved to {context_path}. "
        f"Problem description: {problem!r}"
    )

    agent = SRAgent(
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        tools=args.tools,
        local_sample_size=args.local_sample_size,
        max_refinement_depth=args.max_refinement_depth,
        global_width=args.global_width,
        max_restart_loop=args.max_restart_loop,
        restart_top_k=args.restart_top_k,
        verbose=args.verbose,
        tool_parser=args.tool_parser,
        save_path=args.save_path,
        max_workers=args.max_workers,
    )

    result = {"status": "not_started", "task": args.task}
    try:
        result |= agent.fit(X=X, y=y, problem_description=problem)
    except KeyboardInterrupt as exc:
        result |= getattr(exc, "partial_result", {"status": "interrupted"})
    except Exception as exc:
        _logger.error(f"Experiment failed with an exception: {log_exception(exc)}")
        result |= getattr(exc, "partial_result", {"status": "failed"})
        result["error"] = repr(exc)
        if args.debug:
            raise
    finally:
        with (Path(args.save_path) / "result.json").open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            f.write("\n")
    return result


if __name__ == "__main__":
    parser = build_argparser()
    args, unknown = parser.parse_known_args()

    if args.exp_name is None:
        now = datetime.now()
        args.exp_name = sanitize_filename(
            f"{now:%Y%m%d}_{args.name}_{now:%H%M%S}_{gethostname()}"
        )
    else:
        args.exp_name = sanitize_filename(args.exp_name)
    if args.debug:
        args.verbose = True
    if args.seed == -1:
        args.seed = int(datetime.now().timestamp() * 1000) % (2**32 - 1)
    seed_all(args.seed)
    save_path = Path(args.save_dir) / args.exp_name
    save_path.mkdir(parents=True, exist_ok=True)
    args.save_path = str(save_path)
    args.command = " ".join(map(shlex.quote, [sys.executable, *sys.argv]))

    setup_logging(
        info_level="debug" if args.verbose else "info",
        exp_name=args.exp_name,
        save_path=save_path / "info.log",
        force=True,
    )

    if unknown:
        _logger.warning(f"Unknown args: {unknown}")
    _logger.note(f"Args: {args}")

    save_args(args, save_path / "args.json")

    main(args)
    _logger.note(tag2ansi(f"Experiment completed. Re-run the script with [green bold]{args.command}[reset]"))
