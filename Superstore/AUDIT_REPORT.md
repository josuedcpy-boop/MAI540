# Auditoría de modelo — Superstore.py

**Fecha:** 2026-09-29
**Archivo auditado:** `Superstore.py` (script `.py`, no notebook — corre siempre de arriba hacia abajo, sin ambigüedad de orden de ejecución)
**Columna objetivo:** `Sales`
**Columna de subgrupo:** `Region` (`Central`, `East`, `South`, `West`)

## Resumen

**3 de 5 checks pasan** (1, 2 y 3). El check 4 (disparidad entre subgrupos) **FALLA**: el error del modelo en la región South es más del triple (RMSE) que en Central/West. El check 5 (selección de hiperparámetros) **no aplica** — el script no hace ninguna búsqueda de hiperparámetros.

## Tabla de verificaciones

| # | Check | Veredicto | Evidencia |
|---|-------|-----------|-----------|
| 1 | Orden `train_test_split` vs. `fit()` de transformadores | **PASA** | `train_test_split` en la línea 48. La función `evaluar()` (líneas 81-98) construye un `Pipeline` con el `preprocessor` y llama `pipeline.fit(X_train, y_train)` (línea 86) — siempre después del split, siempre sobre `X_train`. Ningún `fit(`/`fit_transform(` aparece sobre `X` completo o `X_test` en todo el archivo. `Order Month` (líneas 32-33) se calcula sobre `df` completo *antes* del split, pero es una transformación fila por fila (mes de la fecha de esa misma fila) sin ninguna estadística agregada entre filas — no filtra información de test hacia train. |
| 2 | Fuga del objetivo en columnas predictoras | **PASA** | `X` (línea 42) se construye a partir de `numeric_features + categorical_features` (líneas 35-36) — no incluye `Sales` ni `comision_vendedor`. El propio script verifica esto programáticamente con un `assert` (líneas 38-40) que revienta si alguna columna prohibida se cuela; corrió sin error (confirmado: la salida real del script imprime "Verificación de fuga de datos: OK"). |
| 3 | Métricas reportadas correctamente | **PASA** | `evaluar()` calcula `mae`, `rmse`, `r2` (líneas 89-91) usando `y_test` y `pred = pipeline.predict(X_test)` (línea 87) — nunca sobre datos de entrenamiento. Para un target continuo y sesgado como `Sales`, reportar los tres (no solo R²) es apropiado — es el equivalente, en regresión, a no reportar solo accuracy en un problema desbalanceado. |
| 4 | Disparidad entre subgrupos (`Region`) | **FALLA** | Ver "Hallazgos detallados". El script calcula el residuo promedio por región (líneas 163-164) pero no MAE/RMSE por región — se calculó aquí, reutilizando el mismo modelo y las mismas predicciones de test ya generadas (regresión lineal, la de mejor desempeño global), sin reentrenar. |
| 5 | Selección de hiperparámetros y validación anidada | **NO SE PUEDE DETERMINAR** | No hay búsqueda de hiperparámetros en el script: `RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE)` (línea 77) usa un valor fijo elegido de antemano, no un bucle ni `GridSearchCV`. El check no aplica tal como está el código hoy. |

## Hallazgos detallados

### Check 4 — Disparidad de error entre regiones (regresión lineal)

El script reporta MAE/RMSE/R² solo de forma global (línea 108-111) y, por separado, el **residuo promedio** por región (sesgo direccional, líneas 163-164) — pero nunca el **tamaño del error** (MAE/RMSE) por región, que es la métrica que el propio script usa para comparar modelos. Calculándola sobre las mismas predicciones de test ya generadas:

| Región | n (test) | MAE | RMSE | R² | MAE vs. global ($237.44) |
|---|---|---|---|---|---|
| Central | 580 | $192.07 | $401.19 | 0.327 | −$45.36 |
| West | 814 | $205.75 | $404.18 | 0.243 | −$31.68 |
| East | 651 | $268.38 | $758.57 | 0.257 | +$30.95 |
| **South** | 405 | **$316.33** | **$1,294.37** | **0.113** | **+$78.89** |

**Por qué importa:** el RMSE en South (\$1,294) es **más de 3 veces** el de Central/West (\$401/\$404), y su R² (0.113) es menos de un tercio del de Central (0.327) — el modelo explica mucho menos de la variación de ventas ahí. Si este modelo se usara para la comparación entre regiones que motiva el proyecto (¿en qué región conviene enfocar un producto?), las recomendaciones para productos vendidos en South serían sistemáticamente menos confiables que las de Central o West, sin que el reporte global (que solo muestra un RMSE combinado de \$722.66) deje ver esa diferencia.

Esto es consistente con, pero distinto de, el hallazgo de sesgo direccional que ya tenía el script: el residuo promedio de South (+\$136.20) ya insinuaba que esa región se subestima en promedio, pero no decía nada sobre qué tan grande es el error típico ahí — dos regiones pueden tener el mismo sesgo promedio y un tamaño de error muy distinto.

## Acciones recomendadas

1. **Prioridad alta:** agregar al script el cálculo de MAE/RMSE/R² por región (no solo el residuo promedio) junto al resto de las métricas, para que esta disparidad quede visible cada vez que se corra — no solo cuando alguien audite manualmente.
2. **Prioridad media:** investigar por qué South tiene tanto peor ajuste — podría deberse a una composición de categorías/sub-categorías distinta en South (ej. más ventas de `Machines`/`Copiers`, las sub-categorías con mayor error según el diagnóstico de residuos ya hecho), no necesariamente a un problema del modelo en sí. Cruzar `Region` con `Sub-Category` en el diagnóstico existente confirmaría o descartaría esto.
3. **Prioridad baja:** si en el futuro se agrega búsqueda de hiperparámetros (ej. para `RandomForestRegressor`), volver a correr el check 5 — hoy no aplica porque no existe ese código.
4. Los checks 1-3 están bien fundamentados — no requieren acción.
