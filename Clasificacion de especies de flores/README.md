# Comparación de Clasificadores — Especies de Flores (Iris)

## Objetivo

Comparar el desempeño de distintos clasificadores sobre el dataset Iris (3 especies: setosa, versicolor, virginica) usando validación cruzada de 10 particiones, y justificar con evidencia — no solo con impresión visual — tanto la elección del modelo final como la elección de qué variables usar.

## Archivos

- **`App_Comparacion_Clasificadores_Iris.ipynb`** — notebook principal. Contiene todo el flujo: carga de datos, exploración, entrenamiento de los tres clasificadores base (regresión logística, árbol de decisión, Naive Bayes) más KNN y Random Forest, evaluación con validación cruzada, visualización de fronteras de decisión, exploración de profundidad del árbol, comparación pétalos vs. sépalos, tabla comparativa final y recomendación de modelo.
- **`AUDIT_REPORT.md`** — primera auditoría (skill `auditoria-modelos`) sobre la versión de 3 modelos.
- **`AUDIT_REPORT_v2.md`** — segunda auditoría, sobre la versión ampliada con KNN y Random Forest. Los 4 checks aplicables pasan; el check de disparidad entre subgrupos no aplica porque el dataset no tiene ninguna columna demográfica o de subgrupo.

## Cómo correrlo

Abrir `App_Comparacion_Clasificadores_Iris.ipynb` en Jupyter o Google Colab y ejecutar todas las celdas en orden, de arriba hacia abajo. No requiere ningún archivo de datos externo — usa `sklearn.datasets.load_iris()`.

Dependencias: `numpy`, `pandas`, `matplotlib`, `scikit-learn`.

## Hallazgos principales

**Tabla comparativa final (10-fold CV):**

| Clasificador | Accuracy promedio | Desviación estándar |
|---|---|---|
| Regresión logística | 0.9667 | 0.0333 |
| KNN (k=5) | 0.9600 | 0.0533 |
| Naive Bayes | 0.9533 | 0.0521 |
| Random Forest | 0.9533 | 0.0521 |
| Árbol de decisión | 0.9400 | 0.0554 |

Regresión logística tiene el mejor accuracy promedio y, sobre todo, la menor desviación estándar entre particiones — es el modelo más estable, no solo el de mayor promedio.

**Profundidad del árbol de decisión:** al variar `max_depth` (2, 3, 5, sin límite), el accuracy de entrenamiento sube hasta 1.00 en profundidad 5 y sin límite, pero el accuracy de validación cruzada baja de 0.940 (en profundidad 3) a 0.933. Esa combinación —entrenamiento perfecto y validación cruzada que empeora— es la señal de sobreajuste: profundidad 3 es el mejor punto de este barrido, no una profundidad mayor.

**Pétalos vs. sépalos:** se evaluaron los tres clasificadores base sobre ambos pares de variables con la misma función `evaluar_con_cv`:

| Clasificador | Par de variables | Accuracy promedio | Desviación estándar |
|---|---|---|---|
| Regresión logística | Pétalos | 0.9600 | 0.0442 |
| Regresión logística | Sépalos | 0.8000 | 0.0989 |
| Árbol de decisión (max_depth=3) | Pétalos | 0.9467 | 0.0499 |
| Árbol de decisión (max_depth=3) | Sépalos | 0.7467 | 0.0581 |
| Naive Bayes | Pétalos | 0.9600 | 0.0442 |
| Naive Bayes | Sépalos | 0.7867 | 0.1222 |

Los pétalos separan mejor las tres especies que los sépalos con los tres clasificadores — entre 15 y 20 puntos porcentuales de accuracy más, y con menor desviación estándar (más estable entre particiones). La diferencia no depende de qué modelo se use: es una propiedad de qué tan informativas son las variables mismas.
