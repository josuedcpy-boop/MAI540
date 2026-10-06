"""
El lazo de conversación con Claude: envía mensajes, ofrece las tools del
servidor MCP, y ejecuta cada tool que Claude decida usar hasta obtener una
respuesta final en texto.

Este archivo tampoco es parte del ejercicio de la clase — es la misma
fontanería en cualquier aplicación que use la API de Claude con tools, sin
relación específica con MCP. Se entrega completo.
"""

import json
import os

from anthropic import Anthropic
from anthropic.types import MessageParam, ToolParam

from mcp_client import MCPClient

MODEL = "claude-sonnet-4-5"


class Chat:
    def __init__(self, mcp_client: MCPClient):
        self.mcp_client = mcp_client
        self.claude = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.messages: list[MessageParam] = []

    async def _tools_para_claude(self) -> list[ToolParam]:
        tools = await self.mcp_client.list_tools()
        return [
            {
                "name": t.name,
                "description": t.description or "",
                "input_schema": t.inputSchema,
            }
            for t in tools
        ]

    async def run(self, query: str) -> str:
        self.messages.append({"role": "user", "content": query})
        tools = await self._tools_para_claude()

        while True:
            response = self.claude.messages.create(
                model=MODEL,
                max_tokens=1500,
                messages=self.messages,
                tools=tools,
            )

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            text_blocks = [b.text for b in response.content if b.type == "text"]

            self.messages.append({"role": "assistant", "content": response.content})

            if not tool_uses:
                return "\n".join(text_blocks)

            tool_results = []
            for block in tool_uses:
                print(f"  [usando tool: {block.name}({json.dumps(block.input)})]")
                try:
                    result = await self.mcp_client.call_tool(block.name, block.input)
                    contenido = "\n".join(
                        c.text for c in result.content if hasattr(c, "text")
                    )
                except Exception as exc:  # noqa: BLE001
                    contenido = f"Error ejecutando la tool: {exc}"
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": contenido,
                    }
                )

            self.messages.append({"role": "user", "content": tool_results})
