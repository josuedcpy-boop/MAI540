"""
Visualiza el dataset proyectado en 2 componentes de PCA, coloreado por el
grupo que asignó K-Means, y reporta cuánta varianza conserva esa proyección.

Reutiliza directamente ejecutar_pca() de pca_utils.py (el mismo PCA de la
Tarea 5.1 -- ahora también devuelve las coordenadas de cada fila, no solo
varianza y cargas) y ejecutar_kmeans() de kmeans_utils.py para las etiquetas
de grupo. No vuelve a implementar ninguna de las dos cosas.

Uso:
    python visualizar_pca_kmeans.py [nombre_dataset] [k]
    (por defecto: iris, k=3)
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import kmeans_utils
import pca_utils

GRAFICAS_DIR = Path(__file__).parent / "graficas"


def visualizar_pca_kmeans(nombre: str, k: int) -> float:
    etiquetas = kmeans_utils.ejecutar_kmeans(nombre, k)["etiquetas"]
    proyeccion = pca_utils.ejecutar_pca(nombre, n_componentes=2)
    coordenadas = proyeccion["coordenadas"]
    x = [fila[0] for fila in coordenadas]
    y = [fila[1] for fila in coordenadas]
    var1, var2 = proyeccion["varianza_explicada_por_componente"]

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(x, y, c=etiquetas, cmap="viridis", s=40, edgecolor="k", linewidth=0.3)
    ax.set_xlabel(f"PC1 ({var1:.1%} de la varianza)")
    ax.set_ylabel(f"PC2 ({var2:.1%} de la varianza)")
    ax.set_title(f"{nombre}: proyección PCA coloreada por grupo K-Means (k={k})")
    leyenda = ax.legend(*scatter.legend_elements(), title="Grupo")
    ax.add_artist(leyenda)
    plt.tight_layout()

    GRAFICAS_DIR.mkdir(exist_ok=True)
    ruta = GRAFICAS_DIR / f"pca_kmeans_{nombre}_k{k}.png"
    plt.savefig(ruta, dpi=120)
    plt.close()

    varianza_conservada = proyeccion["varianza_acumulada"]
    print(f"Dataset: {nombre}, k={k}")
    print(f"Varianza conservada por la proyección en 2 componentes: "
          f"{varianza_conservada:.4f} ({varianza_conservada:.1%})")
    print(f"Gráfica guardada en: {ruta}")
    return varianza_conservada


if __name__ == "__main__":
    nombre_arg = sys.argv[1] if len(sys.argv) > 1 else "iris"
    k_arg = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    visualizar_pca_kmeans(nombre_arg, k_arg)
