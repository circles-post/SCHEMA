"""Standalone bioRxiv API connectivity test.

Usage:
    python test_biorxiv.py                 # test with current env proxies
    HTTPS_PROXY=http://127.0.0.1:10810 python test_biorxiv.py
    python test_biorxiv.py --no-proxy      # force direct (expect timeout on this box)

Exercises three endpoints from the docs you pasted:
  1. /details/biorxiv/10d/0/json           — 10 most recent bioRxiv posts
  2. /details/biorxiv/<date>/<date>/0/json — date range, with category filter
  3. /details/biorxiv/<DOI>/na/json        — single manuscript by DOI
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import requests


BASE = "https://api.biorxiv.org"
TIMEOUT = 15.0


def _show_env() -> None:
    print("=== Proxy environment ===")
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "NO_PROXY", "no_proxy"):
        val = os.environ.get(key)
        if val:
            print(f"  {key}={val}")
    if not any(os.environ.get(k) for k in ("HTTPS_PROXY", "https_proxy")):
        print("  (no HTTPS proxy set)")
    print()


def _call(label: str, url: str) -> dict | None:
    print(f"[{label}]")
    print(f"  GET {url}")
    t0 = time.time()
    try:
        resp = requests.get(url, timeout=TIMEOUT)
    except Exception as exc:
        print(f"  FAILED in {time.time() - t0:.1f}s: {type(exc).__name__}: {exc}")
        print()
        return None
    dt = time.time() - t0
    print(f"  HTTP {resp.status_code} in {dt:.2f}s  ({len(resp.content)} bytes)")
    try:
        payload = resp.json()
    except Exception as exc:
        print(f"  JSON decode failed: {exc}")
        print(f"  body head: {resp.text[:200]!r}")
        print()
        return None
    messages = payload.get("messages") or []
    collection = payload.get("collection") or []
    print(f"  messages: {messages[:1]}")
    print(f"  collection size: {len(collection)}")
    if collection:
        first = collection[0]
        if isinstance(first, dict):
            print(f"  first.doi:      {first.get('doi')}")
            print(f"  first.title:    {str(first.get('title',''))[:90]!r}")
            print(f"  first.date:     {first.get('date')}")
            print(f"  first.category: {first.get('category')}")
    print()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-proxy", action="store_true", help="strip proxy env vars before testing")
    parser.add_argument("--doi", default="10.1101/2020.09.09.283549", help="single-DOI test target")
    parser.add_argument("--keyword", default="", help="optional keyword to grep locally across the first page")
    args = parser.parse_args()

    if args.no_proxy:
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            os.environ.pop(key, None)
        print("(stripped proxy env)\n")

    _show_env()

    # 1) most recent N posts — cheapest sanity check
    _call("recent_10d", f"{BASE}/details/biorxiv/10d/0/json")

    # 2) date range with category filter (docs explicitly support this)
    _call(
        "range_cell_biology",
        f"{BASE}/details/biorxiv/2025-03-21/2025-03-28/0/json?category=cell_biology",
    )

    # 3) single DOI lookup — this is the endpoint you SHOULD use for per-keyword lookups
    _call("single_doi", f"{BASE}/details/biorxiv/{args.doi}/na/json")

    # 4) optional: show what a keyword grep over 10d would return
    if args.keyword:
        payload = _call("keyword_grep_source", f"{BASE}/details/biorxiv/10d/0/json")
        if payload:
            kw = args.keyword.casefold()
            hits = [
                item for item in (payload.get("collection") or [])
                if isinstance(item, dict)
                and (kw in str(item.get("title", "")).casefold()
                     or kw in str(item.get("abstract", "")).casefold())
            ]
            print(f"[keyword_grep] '{args.keyword}' matched {len(hits)} / {len(payload.get('collection') or [])} papers")
            for h in hits[:3]:
                print(f"  - {h.get('doi')}  {str(h.get('title',''))[:80]!r}")
            print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
