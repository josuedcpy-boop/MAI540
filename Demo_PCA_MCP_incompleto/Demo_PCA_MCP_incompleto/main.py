"""
CLI del laboratorio de la Clase 5.1.

Uso normal: escribe una pregunta y Claude decide si necesita usar una tool
(cargar_dataset, ejecutar_pca).

Dos atajos, en paralelo a lo que hace el curso de Anthropic con documentos:

  @nombre_dataset
      Inserta automáticamente la ficha del dataset (resource
      data://datasets/{nombre}) en tu mensaje, sin que Claude tenga que pedirla
      con una tool. Ejemplo: "¿qué columnas tiene @iris?"

  /interpretar nombre_dataset n_componentes
      Pide el prompt ya evaluado interpretar_componentes al servidor y lo
      envía directamente a Claude. Ejemplo: /interpretar iris 2
"""

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from chat import Chat
from mcp_client import MCPClient

load_dotenv()

PROJECT_DIR = Path(__file__).parent


async def main():
    if "ANTHROPIC_API_KEY" not in os.environ:
        print("Falta ANTHROPIC_API_KEY. Copia .env.example a .env y ponla ahí.")
        sys.exit(1)

    command, args = sys.executable, ["mcp_server.py"]

    async with MCPClient(
        command=command, args=args, cwd=str(PROJECT_DIR)
    ) as mcp_client:
        chat = Chat(mcp_client)

        datasets = await mcp_client.read_resource("data://datasets")
        print("Servidor conectado. Datasets disponibles:", ", ".join(datasets))
        print("Escribe una pregunta, @dataset, /interpretar dataset n, o 'salir'.\n")

        while True:
            try:
                entrada = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not entrada or entrada.lower() in {"salir", "exit", "quit"}:
                break

            if entrada.startswith("/interpretar"):
                partes = entrada.split()
                if len(partes) != 3:
                    print("Uso: /interpretar nombre_dataset n_componentes")
                    continue
                _, nombre, n = partes
                mensajes = await mcp_client.get_prompt(
                    "interpretar_componentes",
                    {"nombre": nombre, "n_componentes": n},
                )
                texto = mensajes[0].content.text
                respuesta = await chat.run(texto)
                print(f"\n{respuesta}\n")
                continue

            if "@" in entrada:
                for palabra in entrada.split():
                    if palabra.startswith("@"):
                        nombre = palabra[1:]
                        try:
                            ficha = await mcp_client.read_resource(
                                f"data://datasets/{nombre}"
                            )
                            entrada += f"\n\n[Ficha de {nombre}: {ficha}]"
                        except Exception as exc:  # noqa: BLE001
                            print(f"  (no se pudo leer {nombre}: {exc})")

            respuesta = await chat.run(entrada)
            print(f"\n{respuesta}\n")


if __name__ == "__main__":
    asyncio.run(main())
