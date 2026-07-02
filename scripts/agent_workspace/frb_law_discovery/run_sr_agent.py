#!/usr/bin/env python3
"""Run SRAgent on a CSV produced inside this workspace.

Example:
  python run_sr_agent.py \
      --csv-file ./saved/feature_extraction/output.csv \
      --target linear_polarization \
      --features center_frequency_ghz rm_value_rad_m2 \
      --problem-description "Find an interpretable relation for linear polarization."
"""

from __future__ import annotations

import os
import sys
import json
import math
import shlex
import random
import dotenv
import argparse
import logging
import numpy as np
import nd2py as nd
import pandas as pd
from pathlib import Path
from datetime import datetime
from socket import gethostname
from sr_agent import SRAgent
from sr_agent.tools import BaseTool
from sr_agent.utils import add_minus_flags, add_negation_flags, format_pareto_front, log_exception, sanitize_filename, save_args, seed_all, setup_logging, tag2ansi

REPO_ROOT = Path(__file__)
while not (REPO_ROOT / '.env').exists():
    REPO_ROOT = REPO_ROOT.parent
dotenv.load_dotenv(REPO_ROOT / '.env', override=True)
SCRIPT_NAME = Path(__file__).stem  # run_sr_agent
_logger = logging.getLogger(f"sr_agent.{SCRIPT_NAME}")


def build_argparser() -> argparse.ArgumentParser:
    default_tools = sorted(set(BaseTool.all_registered_names) - {
        'workspace_code_executor', 'ask_human', 'call_llm', 'workspace_shell', 'create_skill', 'edit_skill'
    })

    parser = argparse.ArgumentParser(
        description="Run SRAgent on selected columns from a CSV file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--name", default=f"{SCRIPT_NAME}", help="Experiment task name used when auto-generating exp_name.")
    parser.add_argument("--exp_name", default=None, help="Experiment name. Defaults to a timestamped name.")
    parser.add_argument("--save_dir", default=f"./logs/{SCRIPT_NAME}", help="Root directory for logs and run artifacts.")
    parser.add_argument("--csv_file", type=str, required=True, help="Input CSV file, for example ./saved/<exp_name>/output.csv.")
    parser.add_argument("--target", type=str, required=True, help="Target column name.")
    parser.add_argument("--features", type=str, nargs="+", default=None, help="Feature column names. Defaults to all columns except the target.")
    parser.add_argument("--problem_description", type=str, default=None, help="Problem description passed to SRAgent. Defaults to a generic description from target/features.")
    parser.add_argument("--description_file", type=str, default=None, help="Optional JSON mapping column names to descriptions. Defaults to description.json next to --csv_file.")
    parser.add_argument("--drop_missing", action="store_true", default=True, help="Drop rows with missing values in selected columns.")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed. Default -1 means using current system time.")
    parser.add_argument("--llm_provider", default="openrouter", help="LLM provider name.")
    parser.add_argument("--llm_model", default="deepseek/deepseek-v4-pro", help="LLM model name.")
    parser.add_argument("--tools", default=default_tools, type=str, nargs='+', help="Optional list of tools to use. Default is all built-in tools.")
    parser.add_argument("--ban_tools", default=[], type=str, nargs='+', help="Optional list of tools to ban. Takes precedence over --tools.")
    parser.add_argument("-K", "--local_sample_size", type=int, default=2, help="Number of LLM samples to generate for each branch.")
    parser.add_argument("-L", "--max_refinement_depth", type=int, default=10, help="Maximum agent refinement depth.")
    parser.add_argument("-C", "--global_width", type=int, default=1, help="Number of independent branches per restart loop.")
    parser.add_argument("-R", "--max_restart_loop", type=int, default=2, help="Maximum number of best-solution restart loops.")
    parser.add_argument("--restart_top_k", type=int, default=1, help="Number of previous best formulas to inject into the next restart prompt.")
    parser.add_argument("--tool_parser", default="openai", choices=["openai", "text", "json", "xml"], help="Tool response parser type.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose agent logging.")
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode (verbose + raise caught exceptions).")
    parser.add_argument("--max_workers", type=int, default=0, help="Maximum number of parallel workers for tool execution. 0 means no parallel execution.")
    parser = add_minus_flags(parser)
    parser = add_negation_flags(parser)
    return parser


def load_task(args):
    df = pd.read_csv(args.csv_file)
    target = args.target
    features = args.features or [column for column in df.columns if column != target]
    problem_description = args.problem_description or (
        f"Find a simple, interpretable symbolic formula for {target} "
        f"as a function of {', '.join(features)} using the selected CSV data."
    )
    if missing := [c for c in [target, *features] if c not in df.columns]:
        raise ValueError(f"Missing requested columns: {missing}")
    if not features:
        raise ValueError("No feature columns selected. Pass --features explicitly or provide numeric columns in the CSV.")

    selected_columns = [target, *features]
    description_path = Path(args.description_file) if args.description_file else Path(args.csv_file).with_name("description.json")
    if not description_path.exists():
        _logger.warning(f"No column description file found at {description_path}. Continuing without column descriptions.")
        description_dict = {}
    else:
        description_dict = json.loads(description_path.read_text(encoding="utf-8"))
    variable_description = []
    for name in selected_columns:
        description = description_dict.get(name, "(No description)")
        var_position = "target" if name == target else "feature"
        var_type = str(df[name].dtype)
        variable_description.append(f"- {name} ({var_position}, {var_type}): {description}")
    
    problem_description = (
        problem_description.rstrip() +
        '\n\nVariable descriptions:\n' +
        '\n'.join(variable_description)
    )
    df = df[selected_columns]

    original_rows = len(df)
    if args.drop_missing:
        df2 = df.replace([np.inf, -np.inf], np.nan).dropna()
        if df2.empty:
            raise ValueError(f"No rows remain after dropping missing values in selected columns: {selected_columns}")
        else:
            _logger.info(f"Dropped {len(df) - len(df2)} rows with missing values in selected columns: {selected_columns}")
        df = df2

    X = {name: df[name].values for name in features}
    y = {target: df[target].values}
    metadata = {
        "target": target,
        "features": features,
        "problem_description": problem_description,
        'original_rows': int(original_rows),
        "used_rows": int(len(df)),
        "dropped_rows": int(original_rows - len(df)),
    }
    return X, y, problem_description, metadata


def main(args: argparse.Namespace) -> dict:
    X, y, problem_description, metadata = load_task(args)
    args.target = next(iter(y))
    args.features = list(X.keys())
    args.problem_description = problem_description
    with (Path(args.save_path) / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
        f.write("\n")
    _logger.note(
        f"Prepared symbolic-regression task.\n"
        f"CSV: {args.csv_file}\n"
        f"Rows: {metadata['original_rows']} original, {metadata['used_rows']} used, {metadata['dropped_rows']} dropped\n"
        f"Target: {args.target}\n"
        f"Features: {args.features}\n"
        f"Problem description: {problem_description!r}"
    )

    context_path = Path(args.save_path) / "context.npz"
    np.savez(context_path, data=(X | y), target=args.target, problem_description=problem_description)
    _logger.note(f"Saved context.npz to {context_path}, start running SRAgent...")

    tools = [tool for tool in args.tools if tool not in args.ban_tools]
    _logger.note(tag2ansi(f"Using tools: [green bold]{', '.join(tools)}[reset]"))
    agent = SRAgent(
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        tools=tools,
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

    result = {
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds": None,
        "csv_file": args.csv_file,
        "target": args.target,
        "features": args.features,
        "n_rows": metadata["used_rows"],
        "problem_description": problem_description,
        "random_seed": args.seed,
        "best_formula": None,
        "best_mse": None,
        "status": "not_started",
        "progress": None,
        "token_usage": None,
        "money_usage": None,
        "tools_usage": None,
        "llm_model": f"{args.llm_model} @ {args.llm_provider}",
    }
    try:
        result |= agent.fit(X=X, y=y, problem_description=problem_description)
    except KeyboardInterrupt as e:
        _logger.note("Experiment interrupted by user.")
        result |= getattr(e, "partial_result", {"status": "interrupted"})
    except Exception as e:
        _logger.error(f"Experiment failed with an exception: {log_exception(e)}")
        result |= getattr(e, "partial_result", {"status": "failed"})
        result["error"] = repr(e)
        if args.debug: raise
    finally:
        result["duration_seconds"] = (datetime.now() - datetime.strptime(result["start_time"], "%Y-%m-%d %H:%M:%S")).total_seconds()
        result["times_usage"] = agent.named_timer.to_str(mode='time', mode_of_detail='pace', mode_of_percent='by_time')
        result["token_usage"] = agent.token_counter.to_str(mode='count', mode_of_detail=None, mode_of_percent=None)
        result["money_usage"] = agent.money_counter.to_str(mode='count', mode_of_detail=None, mode_of_percent=None)
        result["tools_usage"] = agent.tools_counter.to_str(mode='count', mode_of_detail='count', mode_of_percent='by_count')
        # 打印日志
        log = '\n'.join([f"[red]{k.replace("_", " ").title()}[reset]: {v}" for k, v in result.items() if k != 'pareto_front'])
        _logger.note(tag2ansi(
            f'\n[gray]{"=" * 50}[reset]\n'
            "[red bold]Symbolic Regression Result[reset]\n"
            f"{log}\n"
            f"\n[red bold]Pareto Front[reset]\n"
            f"{format_pareto_front(result.get('pareto_front'))}\n"
            f'[gray]{"=" * 50}[reset]'
        ))
        # 保存文件
        result_path = Path(args.save_path) / "result.jsonl"
        with open(result_path, "a", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=True)
            f.write("\n")
        _logger.note(f"Result saved to {result_path}")


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
