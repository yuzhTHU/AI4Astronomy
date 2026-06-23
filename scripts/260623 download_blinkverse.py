#!/usr/bin/env python3
"""Download Blinkverse tables into ./data/blinkverse.

The official Blinkverse front end exposes CSV downloads for some tables, but
the single-burst table currently returns HTTP 500 from its CSV endpoint.  For
that table this script falls back to the paginated JSON API and writes both a
raw JSONL dump and a flattened CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


BASE_URL = "https://blinkverse.zero2x.org"
API_ROOT = f"{BASE_URL}/api/app/adcp-blinkverse"
DEFAULT_OUT_DIR = Path("data/blinkverse")

PARTS = {
    "source": "FRB_SOURCE",
    "host": "FRB_HOST",
    "analysis_single": "FRB_ANALYSIS_SINGLE",
}


def parse_record_ranges(values: list[str]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for value in values:
        if "-" in value:
            start_text, end_text = value.split("-", 1)
            start, end = int(start_text), int(end_text)
        else:
            start = end = int(value)
        if start < 1 or end < start:
            raise ValueError(f"invalid record range: {value}")
        ranges.append((start, end))
    return ranges


def record_in_ranges(record_number: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start <= record_number <= end for start, end in ranges)


def page_fully_in_ranges(start: int, stop_exclusive: int, ranges: list[tuple[int, int]]) -> bool:
    return all(record_in_ranges(record_number, ranges) for record_number in range(start, stop_exclusive))


def request_bytes(url: str, *, timeout: int = 30) -> tuple[bytes, dict[str, str]]:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "AI4Astronomy Blinkverse downloader",
            "Accept": "application/json,text/csv,*/*",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        headers = {k.lower(): v for k, v in resp.headers.items()}
        return resp.read(), headers


def request_json(url: str, *, timeout: int = 30) -> Any:
    payload, _headers = request_bytes(url, timeout=timeout)
    return json.loads(payload.decode("utf-8"))


def api_url(path: str, params: dict[str, Any] | None = None) -> str:
    url = f"{API_ROOT}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params, doseq=True)
    return url


def content_rows(response: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    error = response.get("error", {})
    if error.get("code") != "0":
        raise RuntimeError(f"Blinkverse API error: {error}")
    data = response.get("data", {})
    rows = data.get("list", [])
    total = data.get("total_info", {}).get("total_count", len(rows))
    return rows, int(total)


def flatten_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key in ["data_id", "name", "type", "type_desc", "created_time", "updated_time"]:
        flat[key] = flatten_value(record.get(key))
    content = record.get("content") or {}
    for key, value in content.items():
        flat[key] = flatten_value(value)
    return flat


def write_flat_csv(records: list[dict[str, Any]], path: Path) -> None:
    fieldnames: list[str] = []
    seen: set[str] = set()
    flattened = [flatten_record(record) for record in records]
    for row in flattened:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flattened)


def try_download_official_csv(part: str, data_type: str, out_dir: Path) -> Path | None:
    path = out_dir / f"{part}.csv"
    url = api_url(f"/type/{data_type}/download", {"query": "", "data_sub_types": ""})
    try:
        payload, headers = request_bytes(url)
    except urllib.error.HTTPError as exc:
        print(f"[warn] official CSV failed for {part}: HTTP {exc.code}", file=sys.stderr)
        return None

    if not payload.strip():
        print(f"[warn] official CSV for {part} is empty", file=sys.stderr)
        return None

    path.write_bytes(payload)
    print(f"[ok] wrote {path} ({len(payload):,} bytes)")
    disposition = headers.get("content-disposition")
    if disposition:
        print(f"     server filename: {disposition}")
    return path


def request_json_with_retries(url: str, *, timeout: int, retries: int) -> Any:
    last_error: BaseException | None = None
    for attempt in range(1, retries + 1):
        try:
            return request_json(url, timeout=timeout)
        except (TimeoutError, urllib.error.URLError, urllib.error.HTTPError) as exc:
            last_error = exc
            if isinstance(exc, urllib.error.HTTPError) and exc.code == 500:
                raise
            print(f"[warn] request failed on attempt {attempt}/{retries}: {exc}", file=sys.stderr, flush=True)
            time.sleep(min(2 * attempt, 10))
    raise RuntimeError(f"request failed after {retries} attempts: {last_error}")


def download_paginated(
    part: str,
    data_type: str,
    out_dir: Path,
    page_size: int,
    *,
    timeout: int,
    retries: int,
    max_single_500: int,
    skip_record_ranges: list[tuple[int, int]],
) -> None:
    raw_path = out_dir / f"{part}.jsonl"
    csv_path = out_dir / f"{part}.flattened.csv"
    skipped_path = out_dir / f"{part}.skipped_pages.json"

    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    page = 1
    total = None
    while True:
        start = (page - 1) * page_size + 1
        stop = start + page_size
        if page_fully_in_ranges(start, stop, skip_record_ranges):
            skipped.extend(
                {
                    "page": record_number,
                    "page_size": 1,
                    "reason": "inside user-specified skip record range",
                }
                for record_number in range(start, stop)
            )
            print(f"[skip] {part}: skipped known-bad records {start}-{stop - 1}", flush=True)
            page += 1
            continue

        url = api_url(f"/type/{data_type}", {"page": page, "page_size": page_size})
        try:
            response = request_json_with_retries(url, timeout=timeout, retries=retries)
        except urllib.error.HTTPError as exc:
            if exc.code == 500 and page_size > 50:
                print(
                    f"[warn] {part}: page_size={page_size} failed with HTTP 500; retrying with page_size=50",
                    file=sys.stderr,
                    flush=True,
                )
                page_size = 50
                page = 1
                records.clear()
                skipped.clear()
                continue
            if exc.code == 500 and page_size > 1:
                start = (page - 1) * page_size + 1
                stop = start + page_size
                recovered = 0
                print(
                    f"[warn] {part}: page {page} failed with HTTP 500; trying records {start}-{stop - 1} individually",
                    file=sys.stderr,
                    flush=True,
                )
                consecutive_single_500 = 0
                for single_page in range(start, stop):
                    if record_in_ranges(single_page, skip_record_ranges):
                        skipped.append(
                            {
                                "page": single_page,
                                "page_size": 1,
                                "reason": "inside user-specified skip record range",
                            }
                        )
                        continue
                    single_url = api_url(f"/type/{data_type}", {"page": single_page, "page_size": 1})
                    try:
                        single_response = request_json_with_retries(
                            single_url,
                            timeout=timeout,
                            retries=retries,
                        )
                    except urllib.error.HTTPError as single_exc:
                        skipped.append(
                            {
                                "page": single_page,
                                "page_size": 1,
                                "reason": f"HTTP {single_exc.code}",
                            }
                        )
                        print(f"[skip] {part}: record page {single_page} failed with HTTP {single_exc.code}", flush=True)
                        if single_exc.code == 500:
                            consecutive_single_500 += 1
                            if consecutive_single_500 >= max_single_500:
                                for remaining_page in range(single_page + 1, stop):
                                    skipped.append(
                                        {
                                            "page": remaining_page,
                                            "page_size": 1,
                                            "reason": (
                                                f"not requested after {max_single_500} consecutive "
                                                "HTTP 500 responses in the same parent page"
                                            ),
                                        }
                                    )
                                print(
                                    f"[skip] {part}: skipped records {single_page + 1}-{stop - 1} after "
                                    f"{max_single_500} consecutive HTTP 500 responses",
                                    flush=True,
                                )
                                break
                        continue
                    single_rows, total_count = content_rows(single_response)
                    consecutive_single_500 = 0
                    total = total_count
                    records.extend(single_rows)
                    recovered += len(single_rows)
                print(f"[page] {part}: page {page}, recovered {recovered}/{page_size}", flush=True)
                if total is not None and (page * page_size) >= total:
                    break
                page += 1
                time.sleep(0.1)
                continue
            raise
        rows, total_count = content_rows(response)
        total = total_count
        records.extend(rows)
        print(f"[page] {part}: page {page}, got {len(rows)}, total {total_count}", flush=True)
        if not rows or len(records) >= total_count:
            break
        page += 1
        time.sleep(0.1)

    with raw_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    write_flat_csv(records, csv_path)
    skipped_path.write_text(json.dumps(skipped, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] wrote {raw_path} ({len(records):,}/{total:,} records)")
    print(f"[ok] wrote {csv_path}")
    if skipped:
        print(f"[warn] skipped {len(skipped):,} one-record pages; details in {skipped_path}")
    else:
        print(f"[ok] wrote {skipped_path} (no skipped pages)")


def write_manifest(out_dir: Path) -> None:
    manifest = {
        "source": "Blinkverse official API",
        "base_url": BASE_URL,
        "api_root": API_ROOT,
        "parts": PARTS,
        "notes": [
            "source.csv and host.csv are official CSV exports when available.",
            "analysis_single is downloaded through the paginated JSON API if the official CSV endpoint fails.",
            "Dynamic spectra and external large datasets are not downloaded by default; image/file URLs are preserved in table fields.",
        ],
        "external_dataset_links_seen_in_frontend": [
            "https://fastro.scidb.cn",
            "https://www.scidb.cn/en/detail?dataSetId=3b3cf2f75a74419b89a56cc9626af2a0",
        ],
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] wrote {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parts",
        nargs="+",
        choices=sorted(PARTS),
        default=["source", "host", "analysis_single"],
        help="Tables to download. Default: all lightweight tables.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--page-size", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=30, help="Per-request timeout in seconds.")
    parser.add_argument("--retries", type=int, default=5, help="Retries per paginated JSON request.")
    parser.add_argument(
        "--max-single-500",
        type=int,
        default=5,
        help="Skip the rest of a failed parent page after this many consecutive one-record HTTP 500 responses.",
    )
    parser.add_argument(
        "--skip-record-ranges",
        nargs="*",
        default=[],
        metavar="START-END",
        help="One-based record ranges to mark as skipped without requesting, e.g. 795-6928.",
    )
    parser.add_argument(
        "--no-official-csv",
        action="store_true",
        help="Skip official CSV endpoints and use paginated JSON for every selected part.",
    )
    args = parser.parse_args()
    skip_record_ranges = parse_record_ranges(args.skip_record_ranges)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(args.out_dir)

    for part in args.parts:
        data_type = PARTS[part]
        official = None
        if not args.no_official_csv:
            official = try_download_official_csv(part, data_type, args.out_dir)
        if official is None:
            download_paginated(
                part,
                data_type,
                args.out_dir,
                args.page_size,
                timeout=args.timeout,
                retries=args.retries,
                max_single_500=args.max_single_500,
                skip_record_ranges=skip_record_ranges,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
