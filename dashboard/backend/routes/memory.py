"""Memory files viewer."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from memory.memory_io import LESSONS, MARKET_CONTEXT, STRATEGY_NOTES, TRADE_LOG, read_one

router = APIRouter()

_FILES = {
    "trade_log": TRADE_LOG,
    "lessons_learned": LESSONS,
    "market_context": MARKET_CONTEXT,
    "strategy_notes": STRATEGY_NOTES,
}


@router.get("/{name}")
def get_memory(name: str) -> dict:
    path = _FILES.get(name)
    if path is None:
        raise HTTPException(status_code=404, detail=f"Unknown memory file: {name}")
    return {
        "name": name,
        "path": str(path),
        "content": read_one(path),
        "mtime": path.stat().st_mtime if path.exists() else None,
    }


@router.get("/")
def list_memory() -> list[dict]:
    out: list[dict] = []
    for name, path in _FILES.items():
        out.append({
            "name": name,
            "path": str(path),
            "exists": path.exists(),
            "mtime": path.stat().st_mtime if path.exists() else None,
            "size": path.stat().st_size if path.exists() else 0,
        })
    return out
