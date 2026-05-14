"""MCP Client — connects to all MCP servers and exposes a unified tool interface.

Manages server lifecycle (start/stop) and provides:
    - Dynamic tool discovery across all servers
    - Tool routing (call the right server for each tool)
    - Tool schema generation for LLM function calling
"""

from __future__ import annotations

import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import structlog

log = structlog.get_logger()

# Map of server name → module path (relative to project root)
DEFAULT_SERVERS = {
    "github": "servers/github_server.py",
    "code-analysis": "servers/analysis_server.py",
    "test-runner": "servers/test_runner_server.py",
    "knowledge-base": "servers/knowledge_base_server.py",
}


@dataclass
class ServerConnection:
    """A live connection to one MCP server."""

    name: str
    session: ClientSession
    tools: list[dict]  # tool schemas


@dataclass
class MCPClient:
    """Manages connections to multiple MCP servers."""

    servers: dict[str, ServerConnection] = field(default_factory=dict)
    _tool_to_server: dict[str, str] = field(default_factory=dict)
    _exit_stack: AsyncExitStack = field(default_factory=AsyncExitStack)

    async def connect(
        self,
        server_configs: dict[str, str] | None = None,
    ) -> None:
        """Start and connect to all MCP servers.

        Args:
            server_configs: Map of server name → Python script path.
                           Defaults to DEFAULT_SERVERS.
        """
        configs = server_configs or DEFAULT_SERVERS
        project_root = str(Path(__file__).parent.parent)

        for name, script_path in configs.items():
            full_path = str(Path(project_root) / script_path)
            server_params = StdioServerParameters(
                command=sys.executable,
                args=[full_path],
            )

            log.info("connecting_to_server", server=name, script=script_path)

            stdio_transport = await self._exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read_stream, write_stream = stdio_transport
            session = await self._exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await session.initialize()

            # Discover tools
            tools_result = await session.list_tools()
            tool_schemas = []
            for tool in tools_result.tools:
                schema = {
                    "name": tool.name,
                    "description": tool.description or "",
                    "input_schema": tool.inputSchema,
                }
                tool_schemas.append(schema)
                self._tool_to_server[tool.name] = name

            self.servers[name] = ServerConnection(
                name=name,
                session=session,
                tools=tool_schemas,
            )
            log.info(
                "server_connected",
                server=name,
                tools=[t["name"] for t in tool_schemas],
            )

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Call a tool by name, routing to the correct server.

        Args:
            tool_name: The tool to call
            arguments: Tool arguments as a dict

        Returns:
            Tool result as a string
        """
        server_name = self._tool_to_server.get(tool_name)
        if not server_name:
            available = list(self._tool_to_server.keys())
            return f"Error: Unknown tool '{tool_name}'. Available tools: {available}"

        server = self.servers[server_name]
        log.info("calling_tool", tool=tool_name, server=server_name)

        result = await server.session.call_tool(tool_name, arguments)

        # MCP returns a list of content blocks; concatenate text blocks
        texts = []
        for block in result.content:
            if hasattr(block, "text"):
                texts.append(block.text)
        return "\n".join(texts)

    def get_tool_schemas(self) -> list[dict]:
        """Get all tool schemas in OpenAI/Anthropic function-calling format."""
        all_tools = []
        for server in self.servers.values():
            all_tools.extend(server.tools)
        return all_tools

    def get_anthropic_tools(self) -> list[dict]:
        """Get tool schemas formatted for the Anthropic API."""
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["input_schema"],
            }
            for t in self.get_tool_schemas()
        ]

    def get_openai_tools(self) -> list[dict]:
        """Get tool schemas formatted for the OpenAI API."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
            for t in self.get_tool_schemas()
        ]

    async def close(self) -> None:
        """Shut down all server connections."""
        await self._exit_stack.aclose()
        self.servers.clear()
        self._tool_to_server.clear()
        log.info("all_servers_disconnected")
