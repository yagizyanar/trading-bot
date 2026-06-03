"""Tests for the best-effort Telegram alert helper (network mocked)."""
from __future__ import annotations

import notifications


class _FakeResp:
    def __init__(self, status=200):
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_send_telegram_unconfigured_returns_false(monkeypatch):
    monkeypatch.delenv("TELEGRAM_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    assert notifications.send_telegram("hi") is False


def test_send_telegram_success(monkeypatch):
    monkeypatch.setenv("TELEGRAM_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123")
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return _FakeResp(200)

    monkeypatch.setattr(notifications.requests, "post", fake_post)
    assert notifications.send_telegram("hello") is True
    assert "bottok/sendMessage" in captured["url"]
    assert captured["json"]["chat_id"] == "123"
    assert captured["json"]["text"] == "hello"


def test_send_telegram_swallows_errors(monkeypatch):
    monkeypatch.setenv("TELEGRAM_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123")

    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(notifications.requests, "post", boom)
    assert notifications.send_telegram("hello") is False  # never raises
