# Heart Disease — Tarea 1.2 (versión equilibrada)

## Objetivo y nivel de dificultad
Este proyecto es un problema de **clasificación binaria** cuyo objetivo es predecir **target (1 = clase positiva)**.

Esta versión fue ajustada para que tenga una carga de trabajo comparable con las demás variantes de la Tarea 1.2. El alcance del modelo final está limitado a **10 predictores**, un único algoritmo y un mismo patrón de preprocesamiento y evaluación.

## Punto de partida
`main.py` contiene un modelo funcional pero deliberadamente limitado que usa solo unas pocas variables numéricas. Debes ejecutarlo sin modificar nada y conservar su salida como evidencia del **ANTES**.

## Predictores obligatorios del modelo final
Usa **exactamente estas 10 variables**; no es necesario buscar ni añadir otras:

- `age`
- `trestbps`
- `chol`
- `thalach`
- `oldpeak`
- `sex`
- `cp`
- `restecg`
- `exang`
- `thal`

Para el preprocesamiento, considera:
- **Numéricas:** `age`, `trestbps`, `chol`, `thalach`, `oldpeak`.
- **Categóricas:** `sex`, `cp`, `restecg`, `exang`, `thal`.

## Mejora solicitada
Completa el proyecto para que:
1. mantenga un `train_test_split` con `test_size=0.25` y `random_state=42` (usa `stratify=y` solo en clasificación);
2. construya un `ColumnTransformer` con dos ramas:
   - numérica: `SimpleImputer(strategy="median")` + `StandardScaler()`;
   - categórica: `SimpleImputer(strategy="most_frequent")` + `OneHotEncoder(handle_unknown="ignore")`;
3. mantenga el conjunto de prueba separado y ajuste todo el preprocesamiento únicamente con los datos de entrenamiento;
4. use como algoritmo final `LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")`; **no es necesario probar otros algoritmos ni hacer tuning de hiperparámetros**;
   - `class_weight="balanced"` permite trabajar de forma uniforme con la clase positiva/minoritaria sin añadir búsqueda de hiperparámetros;
5. reporte accuracy, precision, recall y F1 para `target=1`;
6. presente una comparación clara entre **ANTES** y **DESPUÉS** usando el mismo `random_state` y el mismo tamaño de prueba;
7. explique con sus propias palabras qué cambió y por qué.

## Consideraciones específicas del dataset
- Aunque `sex`, `cp`, `restecg`, `exang` y `thal` aparecen como números, trátalas como variables categóricas/códigos, no como cantidades continuas.
- `thal` contiene algunos faltantes: no elimines filas completas; usa imputación dentro del pipeline.

## Flujo obligatorio
1. Pide a Claude Code que lea el proyecto y lo explique **sin modificar archivos**.
2. Ejecuta `python main.py` y conserva la salida como evidencia del **ANTES**.
3. Pide un plan que respete exactamente el alcance de este README. Revísalo antes de autorizar cambios.
4. Autoriza la implementación.
5. Ejecuta nuevamente el proyecto y conserva evidencia del **DESPUÉS**.
6. Completa `BITACORA.md` con tus propias palabras.

## Entrega
- Proyecto completado.
- Evidencia de ejecución antes y después.
- Bitácora de desarrollo asistido.
- Explicación propia del pipeline, los tipos de variables y las métricas.

## Límite de alcance
Para mantener una dificultad equivalente entre estudiantes, **no se requiere** selección automática de variables, validación cruzada, búsqueda de hiperparámetros, ingeniería avanzada de características ni comparación de múltiples algoritmos.

## Extensión más allá del alcance original (14 de septiembre de 2026)
Después de entregar la mejora pedida arriba, se hizo un diagnóstico de calidad de datos y varias extensiones que **se salen a propósito** del límite de alcance de esta tarea. Se documentan aquí para dejar clara la diferencia entre la entrega base y el trabajo adicional; el detalle completo está en `Contexto.md` y `BITACORA.md`.

- **Diagnóstico de datos:** 1,669 filas duplicadas exactas (23.8%, probablemente por cómo se generó este archivo académico), faltantes en `ca`/`thal`, y atípicos (IQR) en `trestbps`/`chol`/`oldpeak`/`thalach` — todos clínicamente plausibles. Los duplicados se dejaron documentados como limitación conocida, sin modificar.
- **Manejo de faltantes:** se agregó `thal_missing`, indicador binario de que `thal` fue imputado.
- **Manejo de atípicos:** `RobustScaler` en vez de `StandardScaler` para las variables numéricas.
- **Más variables:** se agregaron `fbs`, `slope` y `ca` como predictores categóricos (14 variables en total, en vez de las 10 originales).
- **Selección de características:** un modelo alterno ("SELECCIONADO") usa solo las 10 variables con mayor correlación con `target` (calculada en train, |r| >= 0.15), con desempeño casi idéntico al de 14 variables mediante un modelo más simple.
- **Salvaguardas adicionales:** verificación de fuga de datos, demostración de fuga real usando `target` como predictor a propósito, y un control que detiene el script si se intenta mostrar `age`/`sex` públicamente.

`main.py` ahora imprime cuatro modelos en un solo reporte: ANTES, DESPUÉS, CON FUGA y SELECCIONADO.
