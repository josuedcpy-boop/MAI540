# Auditoría de modelo — Diagnostico_Biopsias_Mama.ipynb (versión corregida)

**Fecha:** 2026-09-27
**Notebook:** `Diagnostico_Biopsias_Mama.ipynb`
**Columna objetivo:** `diagnostico` (0 = maligno, 1 = benigno)
**Columna de subgrupo:** no aplica — ver check 4

**Nota de reproducibilidad:** en esta versión, `execution_count` es secuencial (1 a 16) y coincide exactamente con el orden de las celdas en el archivo — se corrigió el problema de la versión original (`AUDIT_REPORT(Codigo Original).md`), donde una celda de la sección 3 usaba funciones definidas más adelante en el archivo. Aquí `COLUMNAS_PREDICTORAS` y `construir_caracteristicas` se movieron a una sección explícita justo después de los imports, antes de cualquier uso. Las salidas guardadas sí corresponden a una corrida real de arriba hacia abajo.

## Resumen

**Los 4 checks pasan** (3 con evidencia directa, 1 no se puede determinar por ausencia de columna de subgrupo en el dataset). Las dos fallas de la versión original (fuga de datos vía `sesiones_tratamiento_programadas` y el pipeline que nunca llegaba a reportar métricas) están corregidas.

## Tabla de verificaciones

| # | Check | Veredicto | Evidencia |
|---|-------|-----------|-----------|
| 1 | Orden `train_test_split` vs. `fit()` de transformadores | **PASA** | `entrenar_modelo` (celda `881f59c8`): `train_test_split(X, y, ..., stratify=y)` ocurre primero. La corrección del `inf` en `variabilidad_por_biopsia` (`replace` → `median()` → `fillna`) se calcula **después** del split y **solo con `X_train`** (`mediana_train = X_train['variabilidad_por_biopsia'].median()`), y ese mismo valor de train se aplica a `X_test` — no se recalcula con datos de test. `modelo.fit(X_train, y_train)` también es estrictamente posterior al split. No hay ningún `fit(`/`fit_transform(` sobre `X` completo o `X_test` en ninguna celda. |
| 2 | Fuga del objetivo en columnas predictoras | **PASA** | `COLUMNAS_PREDICTORAS` (celda `0435bcfb`) excluye explícitamente `sesiones_tratamiento_programadas`, con un comentario que documenta por qué (`# Esta columna introduce data leakage`). Las columnas restantes son mediciones de imagen o `num_biopsias_previas` (ruido aleatorio independiente de `diagnostico`, `rng.randint(0,4)`) — ninguna se deriva del target. |
| 3 | Métricas reportadas correctamente | **PASA** | El pipeline (celda `93d6fc8f`) corre sin errores y produce accuracy de train/test. La celda `1003f675` calcula, sobre `y_test`/`y_pred` (no train), precisión/exhaustividad/F1 específicamente de la clase `maligno` (`pos_label=0`) además de accuracy — apropiado dado el desbalance 62.7%/37.3% reportado en `explorar_datos` (celda `cba9b56a`). La celda `eb2574d9` interpreta correctamente la matriz de confusión (falso negativo de malignidad = error más grave), y la celda `11b97fec` justifica con evidencia por qué el recall de "maligno" es la métrica prioritaria. |
| 4 | Disparidad entre subgrupos | **NO SE PUEDE DETERMINAR** | El dataset (Wisconsin Breast Cancer + `num_biopsias_previas`) no contiene ninguna variable demográfica o de subgrupo poblacional. No aplica calcular disparidad sin una columna de agrupación real. |

## Hallazgos detallados

Ningún hallazgo de FALLA en esta versión. Dos notas menores, no bloqueantes:

- **`num_biopsias_previas` es ruido no informativo** (`rng.randint(0, 4, size=n)`, sin relación con `diagnostico`): no es fuga ni error, pero probablemente no aporta señal predictiva real — queda como observación para una futura iteración de selección de características, no como acción requerida.
- **`predecir_caso` (celda `103aad66`)** construye `entrada = pd.DataFrame([caso])[COLUMNAS_PREDICTORAS]`. Como `COLUMNAS_PREDICTORAS` ya no incluye la columna con fuga, esto sigue funcionando correctamente sin cambios adicionales.

## Acciones recomendadas

1. **Ninguna acción crítica pendiente.** Los 4 checks están en buen estado; las dos fallas de la versión original ya se corrigieron y se verificaron con esta auditoría independiente.
2. **Opcional:** evaluar si `num_biopsias_previas` aporta valor real al modelo (por ejemplo, comparando métricas con y sin esa columna) — no es un requisito de esta auditoría, es una mejora de calidad de features.
3. **Opcional:** si en el futuro se agrega una columna demográfica real (edad, procedencia del laboratorio, etc.), volver a correr el check 4 para verificar que no haya disparidad de recall entre subgrupos, como sí se encontró en el proyecto de enfermedad cardíaca de este mismo curso (`MAI540/AUDIT_REPORT.md`).
