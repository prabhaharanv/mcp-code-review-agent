"""Tests for the GitHub webhook endpoint and /metrics endpoint."""

import hashlib
import hmac
import json

import pytest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client without starting MCP servers."""
    with patch("app.api._mcp_client", new=AsyncMock()) as mock_client:
        mock_client.servers = {"github": True}
        mock_client.call_tool = AsyncMock(return_value='{"title": "test"}')
        from app.api import app
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


class TestMetricsEndpoint:
    """Tests for GET /metrics."""

    def test_metrics_returns_prometheus_format(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "agent_review_requests_total" in resp.text
        assert "agent_tool_calls_total" in resp.text


class TestWebhookEndpoint:
    """Tests for POST /webhook."""

    def _pr_payload(self, action="opened"):
        return {
            "action": action,
            "pull_request": {
                "html_url": "https://github.com/owner/repo/pull/42",
                "title": "Test PR",
            },
        }

    def test_webhook_ignores_non_pr_events(self, client):
        resp = client.post(
            "/webhook",
            content=json.dumps({"action": "created"}).encode(),
            headers={"X-GitHub-Event": "issue_comment"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ignored"

    def test_webhook_ignores_closed_action(self, client):
        resp = client.post(
            "/webhook",
            content=json.dumps(self._pr_payload("closed")).encode(),
            headers={"X-GitHub-Event": "pull_request"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ignored"

    def test_webhook_accepts_opened_pr(self, client):
        resp = client.post(
            "/webhook",
            content=json.dumps(self._pr_payload("opened")).encode(),
            headers={"X-GitHub-Event": "pull_request"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "accepted"
        assert resp.json()["pr_url"] == "https://github.com/owner/repo/pull/42"

    def test_webhook_accepts_synchronize(self, client):
        resp = client.post(
            "/webhook",
            content=json.dumps(self._pr_payload("synchronize")).encode(),
            headers={"X-GitHub-Event": "pull_request"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "accepted"

    def test_webhook_validates_signature(self):
        """Test HMAC signature validation when webhook_secret is set."""
        secret = "test-secret-123"
        payload = json.dumps(self._pr_payload()).encode()
        sig = "sha256=" + hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()

        with patch("app.api._mcp_client", new=AsyncMock()) as mock_client, \
             patch("app.api.settings") as mock_settings:
            mock_client.servers = {"github": True}
            mock_settings.webhook_secret = secret
            mock_settings.llm_provider = "anthropic"
            mock_settings.llm_model = "claude-sonnet-4-20250514"

            from app.api import app
            with TestClient(app, raise_server_exceptions=False) as client:
                # Valid signature
                resp = client.post(
                    "/webhook",
                    content=payload,
                    headers={
                        "X-GitHub-Event": "pull_request",
                        "X-Hub-Signature-256": sig,
                    },
                )
                assert resp.status_code == 200

    def test_webhook_rejects_bad_signature(self):
        """Test that invalid HMAC signatures are rejected."""
        secret = "test-secret-123"
        payload = json.dumps(self._pr_payload()).encode()

        with patch("app.api._mcp_client", new=AsyncMock()) as mock_client, \
             patch("app.api.settings") as mock_settings:
            mock_client.servers = {"github": True}
            mock_settings.webhook_secret = secret

            from app.api import app
            with TestClient(app, raise_server_exceptions=False) as client:
                resp = client.post(
                    "/webhook",
                    content=payload,
                    headers={
                        "X-GitHub-Event": "pull_request",
                        "X-Hub-Signature-256": "sha256=invalid",
                    },
                )
                assert resp.status_code == 401

    def test_webhook_missing_pr_url(self, client):
        payload = {"action": "opened", "pull_request": {}}
        resp = client.post(
            "/webhook",
            content=json.dumps(payload).encode(),
            headers={"X-GitHub-Event": "pull_request"},
        )
        assert resp.status_code == 400
