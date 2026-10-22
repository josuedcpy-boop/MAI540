"""
Servidor MCP: dos tools, dos resources, un prompt.

Este es el archivo que el profesor completa en vivo durante la Clase 5.1,
siguiendo el mismo patrón del curso de Anthropic "Introduction to Model
Context Protocol" (M02), adaptado de un chatbot de documentos a un
servidor de análisis de datos.

Tools    -> operaciones que Claude puede decidir ejecutar (cargar un dataset,
            correr PCA).
Resources -> datos que el cliente puede pedir directamente, sin pasar por
            una decisión de Claude (la lista de datasets, la ficha de uno).
Prompt   -> una plantilla ya evaluada para una tarea recurrente: interpretar
            componentes principales en términos del dominio, no solo en
            términos de varianza.

ESTADO: INCOMPLETO A PROPÓSITO.
Cada bloque "# TODO" de abajo es una pieza que se escribe en vivo durante la
clase, siguiendo el patrón de @mcp.tool / @mcp.resource / @mcp.prompt que ya
viste en las diapositivas 10, 11 y 12. La lógica de negocio (pca_utils.py) ya
está completa — aquí solo falta envolverla con el decorador correcto.

Para probar este archivo una vez completado, sin cliente ni CLI:
    mcp dev mcp_server.py
Abre el Inspector en el navegador, conecta, y prueba cada tool/resource/prompt
a mano antes de conectarlo a nada más.
"""

from mcp.server.mcpserver import MCPServer, UserMessage
from pydantic import Field

import kmeans_utils
import pca_utils

mcp = MCPServer("analisis-datos")


# ---------------------------------------------------------------------------
# Tools — algo que Claude DECIDE ejecutar, con los argumentos que el modelo
# elige según lo que pida quien está conversando.
# ---------------------------------------------------------------------------

@mcp.tool(
    name="cargar_dataset",
    description=(
        "Describe un dataset disponible (filas, columnas numéricas y "
        "categóricas). Úsala ANTES de ejecutar_pca, para saber qué columnas "
        "numéricas tiene el dataset y decidir con cuántos componentes correr PCA."
    ),
)
def cargar_dataset(nombre: str = Field(description="Nombre del dataset, sin extensión .csv (ej. 'iris').")) -> dict:
    return pca_utils.describir_dataset(nombre)


@mcp.tool(
    name="ejecutar_pca",
    description=(
        "Ejecuta PCA sobre las columnas numéricas de un dataset. Devuelve la "
        "varianza explicada por componente, la varianza acumulada y las cargas "
        "(loadings) de cada variable original en cada componente."
    ),
)
def ejecutar_pca(
    nombre: str = Field(description="Nombre del dataset, sin extensión .csv (ej. 'iris')."),
    n_componentes: int = Field(description="Número de componentes principales a calcular."),
) -> dict:
    return pca_utils.ejecutar_pca(nombre, n_componentes)


@mcp.tool(
    name="segmentar_kmeans",
    description=(
        "Corre K-Means sobre las columnas numéricas (escaladas) de un dataset, "
        "con el número de grupos k indicado. Devuelve la etiqueta de grupo de "
        "cada fila, el tamaño de cada grupo y el coeficiente de silueta del k "
        "elegido (qué tan bien separados quedaron los grupos entre sí)."
    ),
)
def segmentar_kmeans(
    nombre: str = Field(description="Nombre del dataset, sin extensión .csv (ej. 'iris')."),
    k: int = Field(description="Número de grupos (clusters) a formar."),
) -> dict:
    resultado = kmeans_utils.ejecutar_kmeans(nombre, k)
    etiquetas = resultado["etiquetas"]
    tamano_por_grupo = {
        str(grupo): etiquetas.count(grupo) for grupo in sorted(set(etiquetas))
    }
    return {
        "dataset": resultado["dataset"],
        "k": resultado["k"],
        "etiquetas": etiquetas,
        "tamano_por_grupo": tamano_por_grupo,
        "coeficiente_silueta": resultado["coeficiente_silueta"],
        "inercia": resultado["inercia"],
    }


# ---------------------------------------------------------------------------
# Resources — datos que el CLIENTE pide directamente, sin que Claude decida
# nada. Estático (siempre lo mismo) o con plantilla (un parámetro en la URI).
# ---------------------------------------------------------------------------

@mcp.resource("data://datasets", mime_type="application/json")
def listar_datasets() -> list[str]:
    return pca_utils.listar_datasets()


@mcp.resource("data://datasets/{nombre}", mime_type="application/json")
def ficha_dataset(nombre: str) -> dict:
    return pca_utils.describir_dataset(nombre)


# ---------------------------------------------------------------------------
# Prompt — una plantilla YA EVALUADA para una tarea que se repite, en vez de
# dejar que cada usuario improvise su propia pregunta de interpretación.
# ---------------------------------------------------------------------------

@mcp.prompt(
    name="interpretar_componentes",
    description=(
        "Plantilla ya evaluada para interpretar un resultado de PCA en términos "
        "del dominio de los datos, no solo en términos de varianza."
    ),
)
def interpretar_componentes(
    nombre: str = Field(description="Nombre del dataset, sin extensión .csv (ej. 'iris')."),
    n_componentes: int = Field(description="Número de componentes principales a interpretar."),
) -> list[UserMessage]:
    resultado = pca_utils.ejecutar_pca(nombre, n_componentes)
    prompt = (
        f"Aquí está el resultado de PCA sobre el dataset '{nombre}' "
        f"con {n_componentes} componentes:\n\n"
        f"Varianza explicada por componente: {resultado['varianza_explicada_por_componente']}\n"
        f"Varianza acumulada: {resultado['varianza_acumulada']}\n"
        f"Cargas (loadings) de cada variable original por componente: {resultado['cargas']}\n\n"
        "Con esta información:\n"
        "1. Para cada componente, identifica las 2-3 variables con mayor carga (en valor absoluto).\n"
        "2. Explica qué patrón del dominio podría representar cada componente, a partir de esas variables.\n"
        "3. Indica si el signo de cada carga tiene una lectura razonable (por ejemplo, variables que "
        "suben juntas deberían tener el mismo signo).\n"
        "4. Para terminar, según la varianza acumulada, ¿alcanzan estos "
        f"{n_componentes} componentes para resumir bien el dataset, o haría falta "
        "uno más?"
    )
    return [UserMessage(prompt)]


if __name__ == "__main__":
    mcp.run(transport="stdio")
