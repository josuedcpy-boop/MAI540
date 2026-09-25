# Auditoría de modelo — App_Comparacion_Clasificadores_Iris.ipynb

**Fecha:** 2026-09-24
**Notebook:** C:\Users\whate\Downloads\App_Comparacion_Clasificadores_Iris.ipynb
**Columna objetivo:** `target` / `especie` (mapeo de `target` a nombre de especie, dataset Iris de scikit-learn)
**Columna de subgrupo:** no aplica — el dataset no contiene ninguna columna categórica adicional al target (solo 4 medidas numéricas + especie)

## Resumen

3 de 4 checks pasan con evidencia. El check de disparidad entre subgrupos no se puede determinar porque el dataset no tiene ninguna columna de subgrupo (demográfica o de otro tipo) distinta del target.

## Tabla de verificaciones

| # | Check | Veredicto | Evidencia |
|---|-------|-----------|-----------|
| 1 | Orden train_test_split vs. fit() de escaladores | PASA | No se usa ningún `StandardScaler`/`OneHotEncoder`/`SimpleImputer` en todo el notebook (celda 2, imports). El notebook no usa `train_test_split`; la evaluación de desempeño se hace con `cross_val_score(modelo, X, y, cv=StratifiedKFold(...))` (celda 12, `evaluar_con_cv`), que internamente clona y ajusta el estimador solo sobre el fold de entrenamiento en cada partición — no hay fuga posible sin transformadores externos ajustados fuera de ese ciclo. |
| 2 | Fuga del objetivo en columnas predictoras | PASA | `X = iris.data.values` (celda 4) contiene únicamente `sepal length/width` y `petal length/width`; `y = iris.target.values` es la especie. Ninguna columna de `X` se deriva del target ni contiene información posterior al evento a predecir — son medidas físicas tomadas independientemente de la clasificación final. |
| 3 | Métricas reportadas correctamente | PASA | La tabla comparativa final (celda 23) usa `scores_logreg.mean()`, `scores_arbol.mean()`, `scores_nb.mean()`, que vienen de `evaluar_con_cv` con instancias **nuevas** del modelo (celda 13: `evaluar_con_cv(LogisticRegression(...), X, y, ...)`), no de `modelo_logreg`/`modelo_arbol`/`modelo_nb` ya ajustados sobre todo el dataset en esa misma celda — es decir, las métricas reportadas provienen de validación cruzada, no de reentrenar y evaluar sobre el mismo set de entrenamiento. Además, la celda 17 (sección 6) compara explícitamente `accuracy_entrenamiento` (sobre `X, y` completo, etiquetado como medida de "memorización") contra `accuracy_cv`, dejando claro cuál métrica es la válida para generalización. Las 3 clases de Iris están balanceadas (50 muestras cada una), por lo que `accuracy` sola es una métrica adecuada aquí. |
| 4 | Disparidad entre subgrupos | NO SE PUEDE DETERMINAR | El DataFrame (`df`, celda 4) solo contiene las 4 medidas numéricas, `target` y `especie`. No existe ninguna columna de subgrupo (p. ej. origen de la muestra, fecha de recolección u otra variable demográfica/operacional) distinta del propio target, y el usuario no indicó ninguna. `especie` es el target, no un subgrupo independiente, así que no aplica calcular métricas "por subgrupo" sobre ella sin que sea circular. |

## Hallazgos detallados

### Check 4 — Disparidad entre subgrupos: NO SE PUEDE DETERMINAR

El dataset Iris de scikit-learn, tal como se carga en la celda 4 (`load_iris(as_frame=True)`), no trae ninguna columna adicional de agrupación (sexo, ubicación, lote, fecha, etc.). Las únicas columnas categóricas son `target`/`especie`, que son el objetivo de predicción, no un subgrupo sobre el cual evaluar desempeño diferencial. Por lo tanto no hay forma de ejecutar este check con la información disponible en el notebook.

Esto no es un defecto del notebook — es esperable en un dataset académico como Iris — pero se reporta explícitamente en lugar de omitirse en silencio, según pide la auditoría.

## Acciones recomendadas

1. **No se requiere ninguna corrección de código.** Los checks 1-3 pasan con evidencia clara: no hay fuga de datos entre entrenamiento y evaluación, no hay fuga del target en los predictores, y las métricas reportadas en la tabla final (sección 8) provienen correctamente de validación cruzada.
2. **Check 4 queda abierto por falta de datos, no por error.** Si en el futuro se quiere auditar disparidad entre subgrupos, sería necesario usar un dataset con una columna de subgrupo real (p. ej. lote de muestra, ubicación geográfica) — no aplica forzarlo sobre este dataset de Iris.
3. Al momento de escribir la recomendación final (sección 9 del notebook), apóyense en las cifras ya validadas de la tabla comparativa (celda 23): Regresión logística (media 0.9667, DE 0.0333) es la más precisa y estable; Naive Bayes (0.9533, DE 0.0521) queda en medio; Árbol de decisión (0.9400, DE 0.0554) es el menos estable entre particiones.
