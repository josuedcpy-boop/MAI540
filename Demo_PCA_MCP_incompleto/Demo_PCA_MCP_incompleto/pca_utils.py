"""
Lógica de negocio del servidor: nada de esto sabe que existe MCP.

Se separa a propósito de mcp_server.py para que quede claro qué es "hacer PCA"
y qué es "exponer PCA como una tool de MCP" — son dos cosas distintas. Esta
separación es también la que un estudiante debe replicar al adaptar este mismo
servidor para K-Means en la tarea: agregar funciones aquí, y solo después
envolverlas con decoradores en el archivo del servidor.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

DATASETS_DIR = Path(__file__).parent / "datasets"


def listar_datasets() -> list[str]:
    """Nombres de los datasets disponibles (sin la extensión .csv)."""
    return sorted(p.stem for p in DATASETS_DIR.glob("*.csv"))


def _ruta_dataset(nombre: str) -> Path:
    ruta = DATASETS_DIR / f"{nombre}.csv"
    if not ruta.exists():
        raise ValueError(
            f"No existe el dataset '{nombre}'. Disponibles: {listar_datasets()}"
        )
    return ruta


def describir_dataset(nombre: str) -> dict:
    """Forma, columnas y tipos de un dataset, sin cargarlo dos veces innecesariamente."""
    df = pd.read_csv(_ruta_dataset(nombre))
    numericas = df.select_dtypes(include="number").columns.tolist()
    categoricas = df.select_dtypes(exclude="number").columns.tolist()
    return {
        "nombre": nombre,
        "filas": len(df),
        "columnas_numericas": numericas,
        "columnas_categoricas": categoricas,
    }


def cargar_dataset(nombre: str) -> pd.DataFrame:
    return pd.read_csv(_ruta_dataset(nombre))


def ejecutar_pca(nombre: str, n_componentes: int) -> dict:
    """
    Ejecuta PCA sobre las columnas numéricas de un dataset.

    Devuelve varianza explicada por componente, varianza acumulada y la carga
    (loading) de cada variable original en cada componente — lo necesario para
    que Claude pueda interpretar qué representa cada componente, no solo cuánta
    varianza captura.
    """
    df = cargar_dataset(nombre)
    numericas = df.select_dtypes(include="number").columns.tolist()
    if not numericas:
        raise ValueError(f"'{nombre}' no tiene columnas numéricas para aplicar PCA.")
    if n_componentes < 1 or n_componentes > len(numericas):
        raise ValueError(
            f"n_componentes debe estar entre 1 y {len(numericas)} "
            f"(número de columnas numéricas en '{nombre}')."
        )

    X = df[numericas].dropna()
    X_escalado = StandardScaler().fit_transform(X)

    pca = PCA(n_components=n_componentes, random_state=42)
    coordenadas = pca.fit_transform(X_escalado)

    varianza = pca.explained_variance_ratio_
    cargas = pd.DataFrame(
        pca.components_.T,
        index=numericas,
        columns=[f"PC{i + 1}" for i in range(n_componentes)],
    )

    return {
        "dataset": nombre,
        "filas_usadas": len(X),
        "variables_originales": numericas,
        "n_componentes": n_componentes,
        "varianza_explicada_por_componente": [round(float(v), 4) for v in varianza],
        "varianza_acumulada": round(float(np.sum(varianza)), 4),
        "cargas": {
            col: {idx: round(float(v), 4) for idx, v in cargas[col].items()}
            for col in cargas.columns
        },
        # Coordenadas de cada fila proyectada sobre los n_componentes -- lo que
        # hace falta para graficar la proyección (no solo describirla).
        "coordenadas": coordenadas.tolist(),
    }
