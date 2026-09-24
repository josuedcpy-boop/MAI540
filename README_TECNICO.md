# README técnico — pipeline de `main.py`

> Escrito por mí (no es el enunciado del profesor — ese sigue en `README.md`). Explica el orden real de las transformaciones que hace `main.py` y por qué ese orden específico evita fuga de datos en cada paso. La evidencia numérica de cada decisión está en `BITACORA.md`; las reglas de contexto/restricciones están en `Contexto.md`.

## Resumen del problema
Clasificación binaria: predecir `target` (1 = enfermedad cardíaca) a partir de variables clínicas de `data/datos.csv`. El pipeline entrena y compara tres modelos (ANTES, DESPUÉS, SELECCIONADO) sobre exactamente el mismo split de datos.

## Orden exacto de las transformaciones, y por qué ese orden evita fuga

### 1. Cargar el CSV
`df = pd.read_csv(DATA)`. Sin transformaciones todavía — es el dataset crudo, 7,000 filas.

### 2. Eliminar duplicados — ANTES de dividir train/test
`df = df.drop_duplicates()` (7,000 → 5,331 filas).

**Por qué en este punto y no después:** `datos.csv` tiene 1,669 filas duplicadas exactas (copias del mismo paciente). Si se dividiera primero en train/test y se eliminaran duplicados después, algunas de esas copias ya habrían quedado repartidas entre los dos conjuntos — el modelo podría "memorizar" en entrenamiento una fila casi idéntica a una que luego se evalúa en test, e inflar las métricas sin que se note. Deduplicar **antes** del split garantiza que cada paciente (fila única) caiga en un solo lado de la división.

### 3. Diagnóstico de valores faltantes (solo lectura)
Se cuentan los `NaN` por columna (`ca`, `thal`) únicamente para reportarlos en la salida. No modifica los datos, así que no hay riesgo de fuga aquí — es solo inspección.

### 4. Ingeniería de `thal_missing`
`df["thal_missing"] = df["thal"].isna().astype(int)`.

**Por qué es seguro hacerlo aquí, antes del split:** este indicador se calcula fila por fila, a partir del propio valor de `thal` de esa fila — no usa información de otras filas, ni del test, ni del target. Es determinístico y no depende de ninguna estadística agregada (a diferencia de una media o una mediana), así que no importa si se calcula antes o después de dividir train/test; el resultado sería idéntico.

### 5. Definir las listas de columnas y el chequeo de columnas prohibidas
Solo listas de nombres (`numeric_features`, `categorical_features`, `FORBIDDEN_FEATURES = {"target"}`) y una verificación de que `target` no se coló como predictor. No toca los datos.

### 6. Construir `X` / `y`
`X` se arma explícitamente a partir de `numeric_features + categorical_features + indicator_features` — nunca incluye `target`. Esta es la defensa estructural contra la fuga más directa posible: `target` no puede filtrarse por accidente porque físicamente no está en `X`.

### 7. `train_test_split` — el punto de corte que separa "lo que el modelo puede ver" de "lo que no"
`train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)`.

Este es el paso más importante del orden completo: **todo lo que viene después y que aprende algún parámetro de los datos (medianas para imputar, límites de atípicos, medias/escalas, categorías de one-hot, qué variables tienen buena correlación con el target) debe calcularse usando *solo* `X_train`/`y_train`.** `random_state=42` fijo hace el split reproducible; `stratify=y` mantiene la misma proporción de clases en ambos conjuntos.

### 8. Selección de características — correlación calculada solo en train
Se arma una tabla `train_con_target = X_train.copy(); train_con_target["target"] = y_train` y se calcula la correlación de Pearson de cada predictor con `target` **usando exclusivamente esa tabla** (nunca `X_test`).

**Por qué el orden importa:** si esta correlación se calculara sobre el dataset completo (antes del split, o usando también test), estaríamos decidiendo qué variables usar basándonos parcialmente en el conjunto que se supone que el modelo nunca ha visto — eso sesga la evaluación final a favor del modelo, porque el conjunto de prueba ya influyó en el diseño del modelo antes de evaluarlo.

### 9. Los tres pipelines (ANTES, DESPUÉS, SELECCIONADO) — ajustados solo con train
Cada modelo es un `Pipeline` de scikit-learn que encadena, en este orden, dentro de cada rama de un `ColumnTransformer`:

1. `SimpleImputer` (mediana para numéricas, moda para categóricas) — rellena faltantes.
2. `RecorteIQR` (solo numéricas) — recorta atípicos a los límites del IQR.
3. `RobustScaler` (solo numéricas) — escala usando mediana/IQR.
4. `OneHotEncoder` (solo categóricas) — codifica categorías.
5. `LogisticRegression` — el modelo final.

**Por qué este orden de pasos dentro del pipeline, y por qué evita fuga:** al llamar `pipeline.fit(X_train, y_train)`, cada paso (imputador, recorte, escalador, codificador) aprende sus parámetros (la mediana, los límites del IQR, la media/escala, las categorías vistas) **únicamente de `X_train`**. Cuando después se llama `pipeline.predict(X_test)`, esos mismos parámetros ya aprendidos se aplican para transformar el test — el test nunca influye en cómo se transforma a sí mismo. Si se hubiera hecho `imputer.fit(X)` sobre el dataset completo antes del split, la mediana usada para imputar ya habría "visto" el test, y eso es fuga de datos aunque sea sutil.

El imputador va antes del recorte y del escalador porque `RecorteIQR` y `RobustScaler` no saben qué hacer con un `NaN` — necesitan que los faltantes ya estén resueltos. El recorte va antes del escalador porque, si se escalara primero, los límites del IQR calculados después ya estarían en una escala distinta a la original y perderían su interpretación clínica directa.

### 10. Evaluación — solo con `X_test`
`pipeline.predict(X_test)` y las métricas (`accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`) se calculan comparando contra `y_test`. Es la única vez que el test se usa, y solo para medir — nunca para entrenar ni para tomar ninguna decisión de diseño.

## Los cuatro tipos de fuga que este orden previene

| Tipo de fuga | Dónde podría colarse | Cómo se previene |
|---|---|---|
| **Fuga por fila** (duplicados repartidos en train y test) | Si se deduplicara después del split | `drop_duplicates()` antes del split (paso 2) |
| **Fuga de estadísticas de preprocesamiento** (mediana/escala/límites calculados con test) | Si el imputador/escalador se ajustara sobre `X` completo | Todo el preprocesamiento vive dentro de un `Pipeline` que solo se ajusta (`.fit`) con `X_train` (paso 9) |
| **Fuga en la selección de características** (elegir variables mirando el test) | Si la correlación se calculara sobre el dataset completo | Correlación calculada solo con `X_train`/`y_train` (paso 8) |
| **Fuga directa del objetivo** (`target` usado como predictor) | Si `target` se incluyera en `X` | `X` se construye explícitamente sin `target` (paso 6) + chequeo `FORBIDDEN_FEATURES` + regla absoluta en `Contexto.md` |

## Referencias
- Evidencia numérica de cada decisión (comparaciones de imputación, atípicos, selección de características): `BITACORA.md`.
- Reglas de negocio, restricciones de seguridad/privacidad y decisiones vigentes: `Contexto.md`.
- Enunciado original de la tarea: `README.md`.
