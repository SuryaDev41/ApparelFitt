#!/usr/bin/env python3
"""RAG layer for StyleSense AI.

Curated fashion knowledge lives in ``knowledge_base.json``. It is embedded locally
with fastembed (no embedding-API calls) and stored in Qdrant Cloud. At query time the
app searches this store so common questions can be answered directly from the
knowledge base — cutting Grok/LLM calls — and Grok answers are grounded with retrieved
context when a call is still warranted.

The module degrades gracefully: if ``qdrant-client`` is not installed, or
``QDRANT_URL`` / ``QDRANT_API_KEY`` are not configured, ``rag_enabled()`` returns
False and StyleSense keeps working exactly as before.

Knowledge sources (merged):
    knowledge_base.json   Curated built-in fashion entries.
    Your own .xlsx        Any rows-and-columns spreadsheet, via KB_XLSX / ingest.py --xlsx.

Environment variables (read lazily so ``.env`` is loaded first):
    QDRANT_URL          Qdrant Cloud cluster URL
    QDRANT_API_KEY      Qdrant Cloud API key
    QDRANT_COLLECTION   Collection name (default: stylesense_fashion)
    EMBED_MODEL         fastembed model (default: BAAI/bge-small-en-v1.5)
    KB_XLSX             Optional path to your own Excel file to include.
    KB_XLSX_SHEET       Optional sheet name (default: first/active sheet).
    KB_INCLUDE_BUILTIN  Include knowledge_base.json too (default: 1; set 0 to use only the Excel).
"""

import json
import os
from pathlib import Path

KB_PATH = Path(__file__).with_name("knowledge_base.json")

# Header names that hint at which column holds the answer vs the searchable label.
_ANSWER_HINTS = ("answer", "response", "content", "description", "detail", "advice", "recommendation", "text", "note")
_LABEL_HINTS = ("question", "query", "prompt", "topic", "title", "name", "item", "keyword")

_client = None
_ready = False


def collection_name() -> str:
    return os.getenv("QDRANT_COLLECTION", "stylesense_fashion")


def embed_model() -> str:
    return os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")


def _load_json_kb() -> list[dict]:
    if not KB_PATH.is_file():
        return []
    try:
        return json.loads(KB_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def _find_column(headers: list[str], hints: tuple) -> int | None:
    """Return the index of the first header whose name contains any hint word."""
    for idx, header in enumerate(headers):
        low = header.lower()
        if any(hint in low for hint in hints):
            return idx
    return None


def load_xlsx(path: str) -> list[dict]:
    """Turn any rows-and-columns spreadsheet into knowledge entries.

    The first row is treated as headers. Each subsequent row becomes one entry:
      - the searchable text is every non-empty cell in the row (so any field matches);
      - the returned answer is a detected answer/description column if present,
        otherwise a tidy "**Header:** value" list of the whole row;
      - the label/topic is a detected question/name column, else the first cell.
    """
    from openpyxl import load_workbook

    workbook = load_workbook(path, read_only=True, data_only=True)
    sheet_name = os.getenv("KB_XLSX_SHEET")
    worksheet = workbook[sheet_name] if sheet_name else workbook.active

    row_iter = worksheet.iter_rows(values_only=True)
    header_row = next(row_iter, None)
    if not header_row:
        return []
    headers = [str(h).strip() if h is not None else f"column{i + 1}" for i, h in enumerate(header_row)]

    answer_idx = _find_column(headers, _ANSWER_HINTS)
    label_idx = _find_column(headers, _LABEL_HINTS)

    entries: list[dict] = []
    for row in row_iter:
        values = ["" if cell is None else str(cell).strip() for cell in row]
        values += [""] * (len(headers) - len(values))  # pad short rows
        pairs = [(headers[i], values[i]) for i in range(len(headers)) if values[i]]
        if not pairs:
            continue

        document = " ".join(value for _, value in pairs)
        if answer_idx is not None and answer_idx < len(values) and values[answer_idx]:
            answer = values[answer_idx]
        else:
            answer = "\n".join(f"- **{header}:** {value}" for header, value in pairs)
        label = values[label_idx] if (label_idx is not None and label_idx < len(values) and values[label_idx]) else pairs[0][1]

        entries.append({"question": label, "answer": answer, "topic": label[:60], "document": document})
    return entries


def _load_kb() -> list[dict]:
    """Merge the built-in knowledge base with an optional Excel file."""
    entries: list[dict] = []
    if os.getenv("KB_INCLUDE_BUILTIN", "1").strip().lower() not in ("0", "false", "no"):
        entries += _load_json_kb()

    xlsx = os.getenv("KB_XLSX")
    if xlsx:
        if not Path(xlsx).is_file():
            print(f"[rag] KB_XLSX not found: {xlsx}")
        else:
            try:
                entries += load_xlsx(xlsx)
            except ImportError:
                print("[rag] openpyxl is required for Excel files. Run: pip install -r requirements.txt")
            except Exception as error:  # noqa: BLE001 - never let a bad sheet break startup
                print(f"[rag] Could not read Excel file ({error}).")
    return entries


def rag_enabled() -> bool:
    """True only when Qdrant is configured and the client library is importable."""
    if not (os.getenv("QDRANT_URL") and os.getenv("QDRANT_API_KEY")):
        return False
    try:
        import qdrant_client  # noqa: F401
    except ImportError:
        return False
    return True


def get_client():
    """Return a cached Qdrant client with a local fastembed model, or None."""
    global _client
    if _client is not None:
        return _client
    if not rag_enabled():
        return None
    from qdrant_client import QdrantClient

    client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
    client.set_model(embed_model())  # local fastembed — no embedding API calls
    _client = client
    return client


def _collection_ready(client, expected: int) -> bool:
    try:
        info = client.get_collection(collection_name())
        return (info.points_count or 0) >= expected
    except Exception:
        return False


def ingest(force: bool = False) -> bool:
    """Embed the knowledge base locally and upsert it into Qdrant.

    Idempotent: if the collection already holds at least as many points as the
    knowledge base, it is left untouched unless ``force`` is set.
    """
    client = get_client()
    if client is None:
        return False
    kb = _load_kb()
    if not kb:
        return False

    coll = collection_name()
    if not force and _collection_ready(client, len(kb)):
        return True
    if force:
        try:
            client.delete_collection(coll)
        except Exception:
            pass

    documents = [
        entry.get("document") or f"{entry.get('question', '')} {' '.join(entry.get('tags', []))}".strip()
        for entry in kb
    ]
    metadata = [
        {"question": entry.get("question", ""), "answer": entry["answer"], "topic": entry.get("topic", "")}
        for entry in kb
    ]
    ids = list(range(1, len(kb) + 1))
    client.add(collection_name=coll, documents=documents, metadata=metadata, ids=ids)
    return True


def ensure_ready() -> bool:
    """Prepare the collection once per process; never raises."""
    global _ready
    if _ready:
        return True
    try:
        _ready = ingest(force=False)
    except Exception:
        _ready = False
    return _ready


def retrieve(query: str, limit: int = 3) -> list[dict]:
    """Return the top matches as dicts: {score, question, answer, topic}. Never raises."""
    client = get_client()
    if client is None:
        return []
    try:
        hits = client.query(collection_name=collection_name(), query_text=query, limit=limit)
    except Exception:
        return []

    results = []
    for hit in hits:
        meta = getattr(hit, "metadata", None) or {}
        results.append(
            {
                "score": float(getattr(hit, "score", 0.0)),
                "question": meta.get("question", ""),
                "answer": meta.get("answer", ""),
                "topic": meta.get("topic", ""),
            }
        )
    return results
