from __future__ import annotations


def csv_rows(rows: list[tuple[object, ...]]) -> str:
    return "\n".join(",".join(str(item) for item in row) for row in rows)
