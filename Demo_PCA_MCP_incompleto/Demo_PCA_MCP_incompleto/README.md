# Demo Clase 5.1 — Servidor MCP para PCA

Adaptado del curso de Anthropic *Introduction to Model Context Protocol* (M01–M03):
mismo patrón (servidor con tools, resources y un prompt; cliente que se conecta a
él; un chatbot de CLI que los usa), pero en vez de un chatbot de documentos, es un
servidor de análisis de datos que expone PCA.

## Qué construye este proyecto

Un CLI que conversa con Claude y, cuando hace falta, usa un servidor MCP propio
(`mcp_server.py`) para cargar datasets y ejecutar PCA. Servidor y cliente corren
en el mismo proceso de desarrollo — en un proyecto real normalmente se hace solo
uno de los dos (ver la nota del video de *Project Setup* del curso original).

| Pieza | Qué hace | Se completa en clase o ya viene lista |
|---|---|---|
| `pca_utils.py` | La lógica de PCA en sí, sin nada de MCP | Ya viene lista |
| `mcp_server.py` | Envuelve `pca_utils.py` con tools, resources y un prompt | **Se completa en vivo en la Clase 5.1** |
| `mcp_client.py` | La conexión hacia el servidor (gestión de sesión) | Ya viene lista |
| `chat.py` | El lazo de conversación con la API de Claude | Ya viene lista |
| `main.py` | El CLI: entrada de usuario, atajos `@dataset` y `/interpretar` | Ya viene lista |

## Instalación

Requiere Python 3.10+. Esto es lo único que hace falta para la Tarea 5.1 — no
necesitas ninguna clave de API para instalar ni para probar el servidor.

```bash
pip install -r requirements.txt
```

## Probar el servidor solo, sin cliente ni CLI

Esto es lo primero que hay que hacer siempre que se toca `mcp_server.py` — antes
de conectar nada, confirmar que el servidor responde bien. **Esta es la evidencia
que pide la Tarea 5.1, y no requiere `ANTHROPIC_API_KEY` ni ningún pago:**

```bash
mcp dev mcp_server.py
```

Abre la URL que imprime en el navegador, conecta, y prueba a mano:
- **Tools** → `cargar_dataset` con `nombre="iris"`, luego `ejecutar_pca` con
  `nombre="iris"` y `n_componentes=2`.
- **Resources** → `data://datasets` (lista completa), y el template
  `data://datasets/{nombre}` con `nombre="wine"`.
- **Prompts** → `interpretar_componentes` con `nombre="iris"`, `n_componentes=2`.

## Conectar con Claude Desktop (gratis)

Si quieres ver a Claude decidiendo usar tus tools en una conversación real —no
solo probándolas a mano en el Inspector— sin pagar nada: Claude Desktop (la
app, no el navegador) se conecta a servidores MCP locales usando tu cuenta
normal de claude.ai, sin clave de API y sin cobro por token. Esto es opcional,
no se evalúa en la tarea.

1. Instala Claude Desktop desde `claude.ai/download` si no la tienes.
2. Abre Claude Desktop → Settings → Developer → Edit Config. Esto abre (o crea)
   `claude_desktop_config.json`.
3. Agrega una entrada con **rutas absolutas** a este proyecto:

```json
{
  "mcpServers": {
    "analisis-datos": {
      "command": "/ruta/completa/a/este/proyecto/.venv/bin/python",
      "args": ["/ruta/completa/a/este/proyecto/mcp_server.py"]
    }
  }
}
```

   En Windows, usa `.venv/Scripts/python.exe` y rutas con `C:/...` (con
   barras `/`, aunque sea Windows, para que el JSON no se rompa). Obtén la
   ruta completa con `pwd` (Mac) o `cd` sin argumentos (Windows) dentro de la
   terminal del proyecto.
4. Guarda el archivo y reinicia Claude Desktop por completo.
5. Debe aparecer un ícono de herramienta en el cuadro de chat — ábrelo para
   confirmar que ve `cargar_dataset` y `ejecutar_pca`.
6. Chatea normal: "ejecuta PCA sobre iris con 2 componentes y dime qué
   representa cada uno" — Claude decide usar la tool por su cuenta.

## Correr el CLI completo (opcional, requiere clave de pago)

Esta sección es una alternativa al Claude Desktop de arriba, usando el
cliente propio del proyecto en vez de la app. Requiere crear una clave en
`console.anthropic.com` (de pago, aparte de una suscripción de claude.ai) y
pegarla en un archivo `.env`:

```bash
cp .env.example .env
# Edita .env y pon tu ANTHROPIC_API_KEY
python main.py
```

Ejemplos de uso una vez conectado:

```
> ¿qué columnas numéricas tiene wine?
> ejecuta PCA sobre iris con 2 componentes y dime qué representa cada uno
> ¿qué columnas tiene @iris?
> /interpretar wine 3
```

## Los tres conceptos, en este proyecto

- **Tool** — algo que *Claude decide* ejecutar. `ejecutar_pca` es una tool porque
  Claude necesita decidir con cuántos componentes correrlo según lo que pida el
  usuario.
- **Resource** — algo que el *cliente pide directamente*, sin pasar por una
  decisión de Claude. `data://datasets` es estático (siempre la misma lista);
  `data://datasets/{nombre}` es con plantilla (un parámetro en la URI).
- **Prompt** — una plantilla ya redactada y evaluada para una tarea que se repite.
  `interpretar_componentes` existe porque "interpreta esto" a secas da resultados
  mediocres; un prompt específico que pide identificar variables dominantes,
  explicar el patrón y evaluar si el número de componentes alcanza, da resultados
  consistentemente mejores.

## Para la tarea (K-Means)

El patrón es el mismo. En lugar de completar `mcp_server.py`, en la tarea:

1. Agregas la lógica de K-Means a un archivo tipo `kmeans_utils.py` (equivalente
   a `pca_utils.py`): una función que cargue un dataset y otra que corra K-Means
   con un número de grupos dado y devuelva las etiquetas, el criterio usado para
   elegir *k*, y algún resumen por grupo.
2. Envuelves esas funciones con `@mcp.tool(...)` en tu propio servidor.
3. Reutilizas `mcp_client.py`, `chat.py` y `main.py` tal cual están aquí — no
   hace falta tocarlos.
4. Escribes tu propio prompt, por ejemplo `interpretar_grupos`, que le pida a
   Claude describir cada grupo en términos del dominio del dataset que elijas
   para tu sector profesional, no solo en términos del centroide.

## Datasets incluidos

`datasets/iris.csv` (150 filas, 4 numéricas) y `datasets/wine.csv` (178 filas,
13 numéricas) — ambos de scikit-learn, generados con `sklearn.datasets`. Puedes
agregar cualquier otro CSV a esa carpeta; `pca_utils.py` no tiene nada
hardcodeado sobre estos dos en particular.
