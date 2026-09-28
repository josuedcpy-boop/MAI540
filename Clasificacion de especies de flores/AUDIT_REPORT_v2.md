# Auditoría de modelo — App_Comparacion_Clasificadores_Iris.ipynb (versión con KNN y Random Forest)

**Fecha:** 2026-09-27
**Notebook:** `App_Comparacion_Clasificadores_Iris.ipynb`
**Columna objetivo:** `target` / `especie` (Iris, scikit-learn)
**Columna de subgrupo:** no aplica — ver check 4

**Nota de reproducibilidad:** `execution_count` es secuencial (1 a 14) y coincide con el orden de las celdas en el archivo — el notebook corre de arriba hacia abajo tal como está guardado.

Esta auditoría cubre la versión ampliada del notebook (agrega KNN y Random Forest a los tres modelos originales); complementa la auditoría previa (`AUDIT_REPORT.md`), que ya había validado la versión de 3 modelos.

## Resumen

**Los 4 checks pasan** (3 con evidencia directa, 1 no se puede determinar por ausencia de columna de subgrupo). La adición de KNN y Random Forest no introdujo ninguna fuga de datos.

## Tabla de verificaciones

| # | Check | Veredicto | Evidencia |
|---|-------|-----------|-----------|
| 1 | Orden `train_test_split`/CV vs. `fit()` de transformadores | **PASA** | El notebook no usa `train_test_split`; toda evaluación es `cross_val_score` con `StratifiedKFold` dentro de `evaluar_con_cv` (celda `a37aa1ce`), que clona y ajusta el estimador recibido solo sobre el fold de entrenamiento en cada partición. Para KNN (celda `4cb41105`), el `StandardScaler` va **dentro** de un `Pipeline` (`Pipeline([("scaler", StandardScaler()), ("knn", ...)])`) que es lo que se pasa completo a `evaluar_con_cv` — el escalador se ajusta por fold, nunca sobre `X` completo. Se confirmó que `X` (celda `4abcd0e9`, `X = iris.data.values`) permanece sin escalar en todo el notebook; no hay ningún `scaler.fit(X)` global antes del bucle de KNN. Random Forest (celda `36c8577c`) no requiere escalado y se evalúa igual, sin transformador. |
| 2 | Fuga del objetivo en columnas predictoras | **PASA** | `X = iris.data.values` (celda `4abcd0e9`) son las 4 medidas físicas (sépalo/pétalo); KNN y Random Forest usan exactamente esta misma `X`, `y` — no se agregó ninguna columna nueva que pudiera derivarse del target. |
| 3 | Métricas reportadas correctamente | **PASA** | Los 5 modelos (incluyendo KNN y Random Forest) se evalúan con la misma función `evaluar_con_cv` (celdas `baf5d90f`, `4cb41105`, `36c8577c`), que reporta media/desviación estándar/mínimo/máximo de `cross_val_score`. La tabla final (celda `cadcbbdd`) usa `scores_knn = resultados_knn[mejor_k]` — el array de scores ya generado por la búsqueda de `k`, no un recálculo distinto — y `scores_rf` de una evaluación independiente. Las 3 clases de Iris están balanceadas (50/50/50, confirmado en la auditoría previa), por lo que accuracy sola sigue siendo una métrica adecuada. |
| 4 | Disparidad entre subgrupos | **NO SE PUEDE DETERMINAR** | Sin cambios respecto a la auditoría previa: el dataset no tiene ninguna columna de subgrupo distinta del target. |

## Hallazgos detallados

Ningún FALLA. Una observación de rigor metodológico, no cubierta por los 4 checks de esta auditoría pero relevante para la afirmación "se eligió k por validación":

### Nota — selección de k y evaluación final comparten el mismo particionado de CV

La celda `4cb41105` prueba 8 valores de `k`, cada uno evaluado con `evaluar_con_cv`, que internamente crea `StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)` — **el mismo `random_state` en cada llamada**, por lo que los 8 candidatos de `k` se comparan sobre exactamente las mismas 10 particiones. El `k` ganador (`mejor_k`) se elige por su desempeño en esas particiones, y luego ese mismo resultado (`resultados_knn[mejor_k]`) se reutiliza como la métrica "final" de KNN en la tabla comparativa. Esto es una simplificación común (y razonable para un ejercicio de curso), pero técnicamente es una forma leve de sobreajuste al proceso de selección: las mismas particiones que sirvieron para *elegir* `k` son las que se usan para *reportar* su desempeño, lo cual puede optimizar ligeramente el número reportado a favor de KNN frente a los otros modelos (que no pasaron por ningún proceso de selección de hiperparámetros). La forma más rigurosa sería una validación cruzada anidada (un bucle externo para evaluar, uno interno para elegir `k` en cada partición externa). No se marca como FALLA porque no es fuga de datos entre train/test en el sentido de los checks 1-3, ni afecta a los otros 4 modelos — es una nota de rigor sobre la comparación final, no un defecto de la implementación.

## Acciones recomendadas

1. **Ninguna acción crítica pendiente.** Los 4 checks pasan; KNN y Random Forest se integraron sin introducir fuga de datos.
2. **Opcional (rigor metodológico):** si se quiere reportar el desempeño de KNN sin el sesgo optimista de reutilizar las particiones de selección, implementar validación cruzada anidada para `k` (`GridSearchCV` con un `cv` interno, evaluado a su vez dentro de `cross_val_score` con un `cv` externo distinto).
3. **Opcional:** el hallazgo de la auditoría previa sigue vigente — si se agrega una columna de subgrupo real, correr de nuevo el check 4.
