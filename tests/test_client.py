"""Tests for the MCP client."""

from __future__ import annotations

from agent.client import MCPClient, DEFAULT_SERVERS


class TestMCPClientInit:
    def test_default_state(self):
        client = MCPClient()
        assert client.servers == {}
        assert client._tool_to_server == {}

    def test_get_tool_schemas_empty(self):
        client = MCPClient()
        assert client.get_tool_schemas() == []

    def test_get_anthropic_tools_empty(self):
        client = MCPClient()
        assert client.get_anthropic_tools() == []

    def test_get_openai_tools_empty(self):
        client = MCPClient()
        assert client.get_openai_tools() == []


class TestDefaultServers:
    def test_all_servers_defined(self):
        assert "github" in DEFAULT_SERVERS
        assert "code-analysis" in DEFAULT_SERVERS
        assert "test-runner" in DEFAULT_SERVERS
        assert "knowledge-base" in DEFAULT_SERVERS

    def test_server_paths_are_python(self):
        for name, path in DEFAULT_SERVERS.items():
            assert path.endswith(".py"), f"{name} server path should be .py"
