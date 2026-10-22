"""
Cliente MCP: la conexión hacia mcp_server.py.

Este archivo se entrega ya completo — a diferencia del curso de Anthropic,
donde el estudiante lo completa como ejercicio, aquí el objetivo de la clase
es entender qué hace un servidor MCP (tools, resources, prompts), no repetir
la fontanería de gestionar una sesión. Los estudiantes reutilizan este mismo
archivo sin cambios para la tarea de K-Means.
"""

import json
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextResourceContents


class MCPClient:
    def __init__(
        self,
        command: str,
        args: list[str],
        env: dict | None = None,
        cwd: str | None = None,
    ):
        self._command = command
        self._args = args
        self._env = env
        self._cwd = cwd
        self._session: ClientSession | None = None
        self._exit_stack = AsyncExitStack()

    async def connect(self):
        server_params = StdioServerParameters(
            command=self._command, args=self._args, env=self._env, cwd=self._cwd
        )
        stdio_transport = await self._exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        read, write = stdio_transport
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        await self._session.initialize()

    def session(self) -> ClientSession:
        if self._session is None:
            raise ConnectionError(
                "El cliente MCP no está conectado. Llama a connect() primero."
            )
        return self._session

    # --- Tools --------------------------------------------------------
    async def list_tools(self):
        result = await self.session().list_tools()
        return result.tools

    async def call_tool(self, tool_name: str, tool_input: dict):
        return await self.session().call_tool(tool_name, tool_input)

    # --- Resources ------------------------------------------------------
    async def list_resources(self):
        result = await self.session().list_resources()
        return result.resources

    async def read_resource(self, uri: str):
        result = await self.session().read_resource(uri)
        resource = result.contents[0]
        if isinstance(resource, TextResourceContents):
            if resource.mime_type == "application/json":
                return json.loads(resource.text)
            return resource.text
        return resource

    # --- Prompts --------------------------------------------------------
    async def list_prompts(self):
        result = await self.session().list_prompts()
        return result.prompts

    async def get_prompt(self, prompt_name: str, args: dict):
        result = await self.session().get_prompt(prompt_name, args)
        return result.messages

    async def cleanup(self):
        await self._exit_stack.aclose()

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *_exc_info):
        await self.cleanup()
