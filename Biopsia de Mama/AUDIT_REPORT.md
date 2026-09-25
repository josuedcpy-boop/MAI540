# Auditoría de modelo — App_Diagnostico_Biopsias_Mama.ipynb

**Fecha:** 2026-09-24
**Notebook:** `App_Diagnostico_Biopsias_Mama.ipynb`
**Columna objetivo:** `diagnostico` (0 = maligno, 1 = benigno)
**Columna de subgrupo:** no aplica — ver check 4

**Nota de reproducibilidad (hallazgo, no solo metadato):** ninguna celda del notebook tiene `execution_count` (todas son `null`), así que no hay metadato para comparar orden de ejecución contra orden del archivo. Sin embargo, hay evidencia directa de que **las salidas guardadas no corresponden a una corrida de arriba hacia abajo tal como está el archivo hoy**: la celda de la sección "3. Validación de calidad de datos" (`fbef5102`) llama a `construir_caracteristicas(df_crudo)` y usa `COLUMNAS_PREDICTORAS` — ambos definidos en celdas que aparecen **después** en el archivo (`b397da6d`, sección 4, y `881f59c8`, sección 5). En una ejecución real de arriba hacia abajo con kernel limpio, esa celda lanzaría `NameError` de inmediato. En cambio, muestra una salida coherente (`{'variabilidad_por_biopsia': {'nulos': 0, 'infinitos': 140}}`). Esto solo es posible si esas funciones ya existían en el kernel por haberse ejecutado antes, fuera del orden actual del archivo — es decir, alguien depuró el bug interactivamente y dejó esa celda de diagnóstico pegada en una sección anterior a donde se definen sus dependencias. Las salidas mostradas en el notebook no son confiables como evidencia de "qué pasa si lo corres tal cual está".

## Resumen

**1 de 4 checks pasa** (orden de `fit` de transformadores). **2 fallan**: hay fuga de datos directa del objetivo en una columna predictora, y las métricas ni siquiera llegan a calcularse en una corrida real porque el pipeline se cae antes. **1 no se puede determinar** (no existe columna de subgrupo en este dataset).

## Tabla de verificaciones

| # | Check | Veredicto | Evidencia |
|---|-------|-----------|-----------|
| 1 | Orden `train_test_split` vs. `fit()` de transformadores | **PASA** | `entrenar_modelo` (celda `881f59c8`) hace `train_test_split(X, y, ...)` y luego `modelo.fit(X_train, y_train)` directamente — no hay `StandardScaler`/`SimpleImputer`/`OneHotEncoder` en el pipeline en absoluto (no hace falta: no hay nulos según `explorar_datos`, y todas las columnas son numéricas). La característica `variabilidad_por_biopsia` se construye antes del split (celda `b397da6d`), pero es una operación fila por fila (`mean texture / num_biopsias_previas`) sin ninguna estadística agregada entre filas, así que no filtra información de test hacia train. |
| 2 | Fuga del objetivo en columnas predictoras | **FALLA** | Ver "Hallazgos detallados" — `sesiones_tratamiento_programadas` se construye a partir de `diagnostico` y se usa como predictor. |
| 3 | Métricas reportadas correctamente | **FALLA** | Ver "Hallazgos detallados" — solo se calcula `accuracy_score` (sin precision/recall/F1), y además el pipeline nunca llega a ejecutar esas líneas porque se cae antes con un `ValueError`. |
| 4 | Disparidad entre subgrupos | **NO SE PUEDE DETERMINAR** | El dataset (Wisconsin Breast Cancer + las 2 columnas sintéticas agregadas) no contiene ninguna variable demográfica o de subgrupo poblacional (sin sexo, edad, etc.). El usuario tampoco indicó una columna de subgrupo esperada para este notebook. |

## Hallazgos detallados

### Check 2 — `sesiones_tratamiento_programadas` es una fuga directa del objetivo

En `cargar_datos()` (celda `bbb980c2`):

```python
df['sesiones_tratamiento_programadas'] = np.where(
    df['diagnostico'] == 0,
    rng.randint(3, 9, size=n),
    0,
)
```

Esta columna vale **0 si y solo si `diagnostico == 1` (benigno)**, y un número entre 3 y 8 si `diagnostico == 0` (maligno). Es prácticamente una copia disfrazada del target — el modelo no necesita aprender nada sobre textura ni suavidad del tumor; le basta con la regla "si `sesiones_tratamiento_programadas == 0`, predecir benigno" para acertar casi siempre. Esta columna está en `COLUMNAS_PREDICTORAS` (celda `881f59c8`) y se le pasa directamente al modelo.

**Por qué importa clínicamente, no solo estadísticamente:** el número de sesiones de tratamiento programadas es información que **solo existe después de que ya se hizo el diagnóstico** — en el momento real de predecir (cuando llega una paciente nueva, como en `predecir_caso`, celda `103aad66`), todavía no se sabe si es maligno o benigno, así que tampoco se sabría cuántas sesiones programar. El notebook de hecho lo reconoce implícitamente: en `caso_nuevo` (celda `103aad66`), le pasan `'sesiones_tratamiento_programadas': 0` a mano — es decir, para usar el modelo en un caso real hay que inventar un valor de una columna que en teoría no debería existir todavía. Eso es la señal más clara de fuga de datos: si para predecir necesitas adivinar el valor de una columna que depende del resultado que quieres predecir, esa columna no debería estar en el modelo.

`num_biopsias_previas` (también sintética: `rng.randint(0, 4, size=n)`, independiente de `diagnostico`) no tiene este problema — es ruido aleatorio no informativo, pero no es fuga.

### Check 3 — el pipeline nunca reporta métricas en una corrida real, y las que definiría son incompletas

Dos problemas distintos, ambos bajo este check:

1. **El pipeline se cae antes de reportar nada.** La celda "oficial" de producción (`93d6fc8f`, sección 6: *"Esto es lo que corre en producción"*) ejecuta `entrenar_modelo(df)`, que falla con `ValueError: Input X contains infinity or a value too large for dtype('float64')`. La causa: `variabilidad_por_biopsia = mean texture / num_biopsias_previas` (celda `b397da6d`) produce `inf` en las 140 filas donde `num_biopsias_previas == 0` (confirmado en la salida de la celda `fbef5102`: `{'variabilidad_por_biopsia': {'nulos': 0, 'infinitos': 140}}`, sobre 569 filas totales — un 24.6% del dataset). `LogisticRegression.fit` rechaza esos valores infinitos. Como resultado, `acc_train`/`acc_test` nunca se calculan ni se imprimen en una corrida de arriba hacia abajo — y las celdas posteriores que dependen de `modelo` (coeficientes, `guardar_modelo`, `predecir_caso`) tampoco pueden ejecutarse (todas muestran cero salidas guardadas, consistente con que nunca corrieron con éxito).
2. **Incluso si no se cayera, solo se reportaría accuracy.** `entrenar_modelo` (celda `881f59c8`) devuelve únicamente `accuracy_score` de train y test — no precision, recall, F1 ni matriz de confusión. `diagnostico` tiene una distribución de 62.7% benigno / 37.3% maligno (celda `cba9b56a`) — no es un desbalance extremo, pero en un contexto de diagnóstico de cáncer, un falso negativo (predecir benigno cuando en realidad es maligno) es el error más costoso posible, y el accuracy solo no lo expone. La propia nota metodológica del notebook (celda `2d08dac0`, punto 4) ya advierte de esto: *"`diagnostico` está desbalanceado. No aceptes una corrección solo porque el número subió; compara antes/después sobre la misma partición."* — pero el código como está no calcula ninguna métrica que permita hacer esa comparación con rigor.

## Acciones recomendadas

1. **Prioridad crítica:** quitar `sesiones_tratamiento_programadas` de `COLUMNAS_PREDICTORAS` (celda `881f59c8`) — es fuga directa del objetivo, no un predictor válido. Esto probablemente cambiará el accuracy reportado de forma drástica (hacia abajo), porque gran parte del "desempeño" actual (una vez arreglado el bug de `inf`) vendría de esta columna.
2. **Prioridad alta:** arreglar la fuga de infinitos en `variabilidad_por_biopsia` (celda `b397da6d`) antes de poder evaluar nada — por ejemplo, decidir qué hacer con las 140 filas donde `num_biopsias_previas == 0` (¿excluir la división cuando el denominador es 0 y usar otro valor, o descartar la columna?). Esto no es parte del alcance de esta auditoría (que solo audita fuga/métricas/disparidad, no corrige código), pero bloquea cualquier evaluación real hasta que se resuelva.
3. **Prioridad alta:** una vez el pipeline corra sin errores, agregar precision/recall/F1/matriz de confusión a `entrenar_modelo` — no solo accuracy — dado el desbalance de clases y el costo clínico de un falso negativo.
4. **Prioridad media:** revisar si `num_biopsias_previas` aporta señal real o es puro ruido (viene de `rng.randint` sin relación con `diagnostico`) — no es fuga, pero probablemente tampoco ayuda al modelo.
5. El check 1 (orden de split/fit) está bien — no requiere acción.
