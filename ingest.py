#!/usr/bin/env python3
"""(Re)build the StyleSense Qdrant collection from the knowledge sources.

Usage:
    python ingest.py                        # built-in knowledge_base.json only
    python ingest.py --force                # delete and rebuild from scratch
    python ingest.py --xlsx path/to/file.xlsx           # built-in + your Excel
    python ingest.py --xlsx file.xlsx --replace         # only your Excel
    python ingest.py --xlsx file.xlsx --sheet "Sheet2"  # pick a worksheet

Importing stylesense first loads .env so QDRANT_URL / QDRANT_API_KEY are available.
"""

import os
import sys

import stylesense  # noqa: F401  (imported for its .env loading side effect)
import rag


def _arg_value(flag: str) -> str | None:
    """Return the value following ``flag`` in argv, if present."""
    if flag in sys.argv:
        idx = sys.argv.index(flag)
        if idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
    return None


def main() -> int:
    if not rag.rag_enabled():
        print(
            "Qdrant is not configured. Add QDRANT_URL and QDRANT_API_KEY to .env and "
            "install requirements (pip install -r requirements.txt)."
        )
        return 1

    xlsx = _arg_value("--xlsx")
    if xlsx:
        os.environ["KB_XLSX"] = xlsx
    sheet = _arg_value("--sheet")
    if sheet:
        os.environ["KB_XLSX_SHEET"] = sheet
    if "--replace" in sys.argv:
        os.environ["KB_INCLUDE_BUILTIN"] = "0"

    # Providing a source (or --force) rebuilds so new data actually lands.
    force = "--force" in sys.argv or bool(xlsx)

    entries = rag._load_kb()
    sources = []
    if os.getenv("KB_INCLUDE_BUILTIN", "1").strip().lower() not in ("0", "false", "no"):
        sources.append("knowledge_base.json")
    if xlsx:
        sources.append(xlsx + (f" (sheet: {sheet})" if sheet else ""))
    print(f"Sources: {', '.join(sources)}")
    print(f"Entries to index: {len(entries)}")
    print(
        f"Embedding with {rag.embed_model()} (local) and uploading to "
        f"Qdrant collection '{rag.collection_name()}'…"
    )
    ok = rag.ingest(force=force)
    print("Knowledge base ready." if ok else "Ingest failed — check the connection and your source files.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
