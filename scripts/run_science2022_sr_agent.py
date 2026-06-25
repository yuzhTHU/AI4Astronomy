#!/usr/bin/env python3
"""Run SRAgent on the Science 2022 RM-scattering polarization table.

Input table:
    data/science2022/science2022_polarization_sr.csv

Two intended modes:

1. Single source, no fitted parameter supplied:
       y ~= exp(-a * lambda4_m4)

2. All sources with --with-sigma-rm:
       y ~= exp(-2 * sigma_rm**2 * lambda4_m4)

Here ``sigma_rm`` is fitted per source from ``a = 2*sigma_rm**2`` and then
broadcast to each row.  This is deliberately not a grouped-parameter SR
implementation; evaluate_formula only sees ordinary columns.

If ``--with_p0`` is also passed, the per-source intercept is exposed as a
baseline polarization column.  The expected source-normalized form becomes
``p0 * exp(-2 * sigma_rm**2 * lambda_m**4)``.
"""

from __future__ import annotations
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
from sklearn.metrics import r2_score
from sr_agent.utils import setup_logging, add_minus_flags, add_negation_flags, seed_all, log_exception, tag2ansi, sanitize_filename, save_args

SCRIPT_NAME = Path(__file__).stem
_logger = logging.getLogger(f"sr_agent.{SCRIPT_NAME}")
dotenv.load_dotenv()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SRAgent on the Science 2022 RM-scattering polarization table.", 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--name", default=f"{SCRIPT_NAME}", help="Experiment task name used when auto-generating exp_name.")
    parser.add_argument("--exp_name", default=None, help="Experiment name. Defaults to a timestamped name.")
    parser.add_argument("--save_dir", default=f"./logs/{SCRIPT_NAME}", help="Root directory for logs and run artifacts.")
    parser.add_argument("--input_csv", type=str, default="data/science2022/science2022_polarization_sr.csv")
    parser.add_argument("--source", default="FRB20121102A", help="Use a source name, or 'all'.")
    parser.add_argument("--with_sigma_rm", action="store_true", help="Fit correct per-source sigma_rm and expose it as a feature.")
    parser.add_argument("--with_p0", action="store_true", help="With --with_sigma_rm, also expose the fitted per-source baseline polarization p0.")
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


def build_task(args):
    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Missing Science 2022 polarization table: {input_csv}")

    df = pd.read_csv(input_csv)
    required = {"source", "frequency_mhz", "linear_polarization_fraction"}
    if missing := required - set(df.columns):
        raise ValueError(f"{input_csv} is missing required columns: {sorted(missing)}")
    _logger.info(f"Loaded {len(df)} rows from {input_csv}.")

    df0 = df.copy()
    df["frequency_mhz"] = pd.to_numeric(df["frequency_mhz"], errors="coerce")
    df["linear_polarization_fraction"] = pd.to_numeric(df["linear_polarization_fraction"], errors="coerce")
    df = df.dropna(subset=["source", "frequency_mhz", "linear_polarization_fraction"])
    df = df[(df["frequency_mhz"] > 0) & (df["linear_polarization_fraction"] > 0)]
    df = df.sort_values(["source", "frequency_mhz"]).reset_index(drop=True)
    _logger.info(
        f"Loaded {len(df)} usable rows from {input_csv} "
        f"(dropped {len(df0) - len(df)} rows with missing or invalid data)."
    )

    if args.source == "all":
        pass
    elif args.source in (available_sources := sorted(df["source"].unique())):
        df = df[df["source"].eq(args.source)]
        _logger.info(f"Filtered to source={args.source!r}, {len(df)} rows remain.")
    else:
        raise ValueError(f"Source {args.source!r} not found. Available: {', '.join(available_sources)}")

    X = {}
    X['source'] = df['source'].to_numpy(dtype=str)
    X['frequency_mhz'] = df['frequency_mhz'].to_numpy(dtype=float)
    X['lambda_m'] = 299792458.0 / (X['frequency_mhz'] * 1e6)
    y = {"linear_polarization_fraction": df["linear_polarization_fraction"].to_numpy(dtype=float)}

    if args.with_p0 and not args.with_sigma_rm:
        raise ValueError("--with_p0 requires --with_sigma_rm")

    if args.with_sigma_rm:
        df_data = pd.DataFrame(X | y)
        for name, group in df_data.groupby("source"):
            x_group = group["lambda_m"].to_numpy(dtype=float)
            y_group = group["linear_polarization_fraction"].to_numpy(dtype=float)
            # y = p0 * exp(-2 * sigma_rm**2 * x ** 4)
            if len(group) > 1:
                a, b = np.polyfit(x_group**4, -np.log(y_group), 1)
                sigma_rm = float(np.sqrt(a / 2.0)) if a > 0 else np.nan
                p0 = float(np.exp(-b))
                y_pred = p0 * np.exp(-2 * sigma_rm**2 * x_group ** 4)
                r2 = r2_score(y_group, y_pred)
                _logger.info(f"Fitted source parameters for {name!r}: sigma_rm={sigma_rm:.6f}, p0={p0:.6f}, r2={r2:.6f}")
            elif 0 < y_group[0] <= 1 and x_group[0] > 0:
                a = float(-np.log(y_group[0]) / (x_group[0] ** 4))
                sigma_rm = float(np.sqrt(a / 2.0)) if a >= 0 else np.nan
                p0 = 1.0
                r2 = np.nan
                _logger.info(f"Estimated source parameters for single-point source {name!r}: sigma_rm={sigma_rm:.6f}, p0={p0:.6f}")
            else:
                sigma_rm = np.nan
                p0 = np.nan
                r2 = np.nan
                _logger.warning(f"Cannot fit sigma_rm for single-point source {name!r}.")
            df_data.loc[df_data['source'].eq(name), 'sigma_rm'] = sigma_rm
            df_data.loc[df_data['source'].eq(name), 'p0'] = p0
        X["sigma_rm"] = df_data["sigma_rm"].to_numpy(dtype=float)
        if args.with_p0:
            X["p0"] = df_data["p0"].to_numpy(dtype=float)

    problem = (
        f"This dataset is derived from the frequency-dependent polarization study "
        f"of repeating fast radio bursts reported in Science 2022. "
        f"It contains measurements for {"all selected repeating FRB sources" if args.source == "all" else f"the repeating FRB source {args.source}"}. "
        f"Each row gives an observing frequency and the corresponding debiased degree of linear polarization, normalized so that 1 means 100% linearly polarized. "
        f"The feature frequency_mhz is the observing frequency in MHz. "
        f"The feature lambda_m is the observing wavelength in meters, computed from the speed of light divided by the observing frequency. "
        f"{"The feature sigma_rm is a source-level parameter in rad m^-2 that quantifies the scatter of rotation measures along different propagation paths; it is constant for all rows from the same source. " if args.with_sigma_rm else ""}"
        f"{"The feature p0 is a source-level baseline linear-polarization fraction before the wavelength-dependent depolarization is applied; it is constant for all rows from the same source. " if args.with_p0 else ""}"
        f"The scientific goal is to find a compact, interpretable mathematical relationship that predicts linear_polarization_fraction from the provided physical variables. "
        f"The relationship should capture how propagation through a magnetized, inhomogeneous plasma changes "
        f"the observed linear polarization as wavelength and, when provided, the rotation-measure-scatter parameter vary. "
        f"Do not assume a polynomial form; compare simple nonlinear candidates and retain formulas that generalize across the available frequency range."
    )

    return X, y, problem


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
        f"Prepared Science 2022 task: "
        f"source={args.source}, "
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

    result = {"status": "not_started", "source": args.source}
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
