from __future__ import annotations

import argparse
import json

from pubmed_graph.pubmed_client import PubMedClient


def main() -> None:
    parser = argparse.ArgumentParser(description="PubMed API demo via ESearch, ESummary, EFetch, ELink, and PMC mapping.")
    parser.add_argument("--query", default="cancer immunotherapy")
    parser.add_argument("--retmax", type=int, default=5)
    args = parser.parse_args()
    client = PubMedClient()
    search = client.esearch(args.query, retmax=args.retmax)
    pmids = search.get("esearchresult", {}).get("idlist", [])
    summary = client.esummary(pmids)
    papers = [paper.__dict__ for paper in client.efetch_pubmed_xml(pmids)]
    pmc_map = client.map_pubmed_to_pmc(pmids)
    related = client.elink_related(pmids[: min(3, len(pmids))]) if pmids else {}
    payload = {
        "search": search,
        "summary_keys": list(summary.keys()),
        "papers": papers,
        "pmc_map": pmc_map,
        "related": related,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
