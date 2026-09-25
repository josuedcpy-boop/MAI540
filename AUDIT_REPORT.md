# Auditoría de modelo — pipeline_colab.ipynb

**Fecha:** 2026-09-24
**Notebook:** `pipeline_colab.ipynb` (equivalente exacto de `main.py`; mismo pipeline, mismos resultados verificados)
**Columna objetivo:** `target`
**Columna de subgrupo:** `sex` (elegida por ser la única variable demográfica sensible ya documentada en `Contexto.md`; no se auditó ninguna otra por decisión del usuario)

**Nota de reproducibilidad:** el orden de ejecución de las celdas (`execution_count` 1 a 14) coincide exactamente con el orden en que aparecen en el archivo, sin saltos ni reordenamientos — el notebook fue corrido de arriba hacia abajo tal como está guardado, así que las salidas mostradas corresponden al código actual.

## Resumen

3 de 4 checks **PASAN** con evidencia explícita. El check 4 (disparidad entre subgrupos) **FALLA**: el modelo DESPUÉS tiene un recall notablemente más bajo en pacientes con `sex=0` (0.74) que con `sex=1` (0.88) — una brecha de 14 puntos porcentuales que el notebook no reporta ni menciona.

## Tabla de verificaciones

| # | Check | Veredicto | Evidencia |
|---|-------|-----------|-----------|
| 1 | Orden `train_test_split` vs. `fit()` de escaladores/encoders/imputadores | **PASA** | Celda 7 (índice de código 6): `X_train, X_test, y_train, y_test = train_test_split(...)`. Todos los `.fit(` posteriores — celda 11 `baseline_model.fit(X_train[baseline_features], y_train)`, celda 12 `final_model.fit(X_train, y_train)`, celda 14 `selected_model.fit(X_train[selected_columnas], y_train)` — se llaman después del split y usan exclusivamente `X_train`/subconjuntos de `X_train`. Ningún `fit(` o `fit_transform(` aparece sobre `X` completo o `X_test` en ninguna celda. |
| 2 | Fuga del objetivo en columnas predictoras | **PASA** | Celda 6 (índice 5): `X = df[numeric_features + categorical_features + indicator_features]` — `target` no aparece en ninguna de esas tres listas. La misma celda calcula `FORBIDDEN_FEATURES = {"target"}` y verifica que ninguna columna prohibida se use, con salida impresa confirmando `"Verificacion de fuga de datos: OK (columnas prohibidas no usadas: ['target'])"`. El indicador `thal_missing` (celda 5) se deriva de `thal.isna()`, no de `target` — no es fuga. |
| 3 | Métricas reportadas correctamente | **PASA** | Las tres llamadas a `print_report` (celdas 11, 12, 14) reciben `y_test` y predicciones de `.predict(X_test...)`, nunca de train. `print_report` (celda 10) reporta accuracy, precision, recall, F1 y matriz de confusión juntos — no solo accuracy — lo cual importa porque las clases no están perfectamente balanceadas (train: 3,998 filas con clases ~54/46). |
| 4 | Disparidad entre subgrupos (`sex`) | **FALLA** | Ver "Hallazgos detallados". El notebook no calcula ninguna métrica por subgrupo; se calculó externamente usando el mismo `final_model` ya entrenado en el notebook (mismo `random_state=42`, mismos datos), sin reentrenar con configuración distinta. |

## Hallazgos detallados

### Check 4 — Disparidad de recall por sexo en el modelo DESPUÉS

El notebook nunca desagrega sus métricas por ninguna variable demográfica, a pesar de que `sex` es un predictor usado en los tres modelos y está marcada como variable sensible en `Contexto.md`. Usando el `final_model` (DESPUÉS) ya entrenado y sus predicciones sobre `X_test`, agrupadas por `sex`:

| Subgrupo | n (test) | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Global | 1,333 | 0.8740 | 0.8804 | **0.8558** | 0.8679 |
| `sex=0.0` | 446 | 0.9170 | 0.9368 | **0.7417** | 0.8279 |
| `sex=1.0` | 887 | 0.8523 | 0.8703 | **0.8819** | 0.8761 |

**Por qué importa:** el recall es la métrica que este proyecto declara explícitamente como prioritaria (`Contexto.md`, sección 3: "un falso negativo es mucho más peligroso que un falso positivo"). El modelo detecta correctamente el 88.2% de los casos positivos reales en `sex=1.0`, pero solo el 74.2% en `sex=0.0` — una brecha de **14.0 puntos porcentuales**. En términos concretos: de los pacientes con `sex=0.0` que sí tienen la condición, el modelo deja pasar (falso negativo) a 31 de 120 (25.8%), mientras que en `sex=1.0` deja pasar a 62 de 525 (11.8%). Esto significa que, tal como está, el modelo es sistemáticamente peor detectando la enfermedad en un subgrupo — justo el tipo de sesgo silencioso que una métrica global (accuracy 0.8740, que de hecho es *más alta* en el subgrupo con peor recall) no muestra y puede incluso ocultar.

No se pudo determinar con los datos disponibles si esta disparidad viene del propio dataset (ej. distribución distinta de severidad de la enfermedad por sexo en los datos de Cleveland) o de un efecto del `class_weight="balanced"` u otro paso del pipeline — eso requeriría un análisis adicional fuera del alcance de esta auditoría.

## Acciones recomendadas

1. **Prioridad alta:** agregar una celda al notebook (y una sección equivalente en `main.py`) que calcule y reporte accuracy/precision/recall/F1 por valor de `sex` para el modelo DESPUÉS (y SELECCIONADO), para que esta disparidad quede visible cada vez que se corra el pipeline, no solo cuando alguien la audite manualmente.
2. **Prioridad media:** documentar esta disparidad en `Contexto.md` (sección de Criterios de Evaluación) y en `BITACORA.md`, ya que es información relevante para decidir si el modelo es aceptable para el uso previsto — sobre todo dado que el proyecto ya declara el recall como la métrica que más importa.
3. **Prioridad media:** investigar la causa raíz (¿la brecha viene de la distribución del dataset, o el modelo la amplifica?) antes de decidir si se necesita una corrección específica (ej. umbrales de decisión distintos por subgrupo, o revisar si `sex` está sobre-representado de cierta forma en los casos difíciles).
4. **Prioridad baja:** los checks 1-3 pasaron con evidencia sólida — no se requiere acción sobre el orden del pipeline ni el manejo de `target`; esa parte del trabajo está bien fundamentada.
