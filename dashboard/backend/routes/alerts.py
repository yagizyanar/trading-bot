"""GET /api/alerts/ — current system alerts (cached 5 minutes server-side)."""
from __future__ import annotations

from fastapi import APIRouter

from ..alerts import get_alerts

router = APIRouter()


@router.get("/")
def list_alerts() -> dict:
    alerts = get_alerts()
    summary = {
        "red":    sum(1 for a in alerts if a["severity"] == "red"),
        "yellow": sum(1 for a in alerts if a["severity"] == "yellow"),
        "green":  sum(1 for a in alerts if a["severity"] == "green"),
        "total":  len(alerts),
    }
    summary["top_severity"] = (
        "red" if summary["red"] else "yellow" if summary["yellow"] else "green"
    )
    return {"alerts": alerts, "summary": summary}
