"""
Lógica de negocio de K-Means: nada de esto sabe que existe MCP.

Mismo patrón que pca_utils.py -- se separa de mcp_server.py para que quede
claro qué es "hacer K-Means" y qué es "exponer K-Means como una tool de MCP".
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

DATASETS_DIR = Path(__file__).parent / "datasets"


def _ruta_dataset(nombre: str) -> Path:
    ruta = DATASETS_DIR / f"{nombre}.csv"
    if not ruta.exists():
        disponibles = sorted(p.stem for p in DATASETS_DIR.glob("*.csv"))
        raise ValueError(f"No existe el dataset '{nombre}'. Disponibles: {disponibles}")
    return ruta


def cargar_y_escalar(nombre: str) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """
    Carga el dataset y escala sus columnas numéricas (StandardScaler).

    K-Means agrupa por distancia euclidiana, así que es tan sensible a la
    escala de las variables como PCA: sin escalar, una columna con un rango
    mucho mayor que las demás dominaría la distancia y, por tanto, los grupos.
    """
    df = pd.read_csv(_ruta_dataset(nombre))
    numericas = df.select_dtypes(include="number").columns.tolist()
    if not numericas:
        raise ValueError(f"'{nombre}' no tiene columnas numéricas para aplicar K-Means.")

    X = df[numericas].dropna()
    X_escalado = StandardScaler().fit_transform(X)
    return df.loc[X.index], X_escalado, numericas


def ejecutar_kmeans(nombre: str, k: int) -> dict:
    """
    Corre K-Means con k grupos sobre las columnas numéricas escaladas de un dataset.

    Devuelve la etiqueta de grupo de cada fila, la inercia (suma de distancias
    al cuadrado de cada punto a su centroide -- más bajo es más compacto) y el
    coeficiente de silueta (entre -1 y 1; qué tan bien separados quedan los
    grupos entre sí) para el k elegido.
    """
    _, X_escalado, numericas = cargar_y_escalar(nombre)

    if k < 2 or k > len(X_escalado) - 1:
        raise ValueError(
            f"k debe estar entre 2 y {len(X_escalado) - 1} para '{nombre}' "
            f"({len(X_escalado)} filas disponibles)."
        )

    modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
    etiquetas = modelo.fit_predict(X_escalado)
    silueta = silhouette_score(X_escalado, etiquetas)

    return {
        "dataset": nombre,
        "filas_usadas": len(X_escalado),
        "variables_usadas": numericas,
        "k": k,
        "etiquetas": etiquetas.tolist(),
        "inercia": round(float(modelo.inertia_), 4),
        "coeficiente_silueta": round(float(silueta), 4),
    }
