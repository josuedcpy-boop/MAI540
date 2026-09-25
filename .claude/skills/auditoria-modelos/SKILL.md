---
name: auditoria-modelos
description: Usar para auditar un notebook (.ipynb) de clasificación o regresión antes de aceptar sus resultados o métricas. Aplica siempre que el usuario pida "revisar", "auditar" o "validar" un notebook de machine learning, pregunte si un modelo tiene fuga de datos (data leakage), pida verificar el orden de train_test_split o de un scaler/encoder, o quiera saber si el desempeño del modelo es distinto entre subgrupos (fairness/disparidad). Úsalo también si el usuario simplemente pega un notebook y pregunta "¿estas métricas son confiables?" o "¿este pipeline está bien hecho?", incluso si no menciona la palabra "auditoría".
---

# Auditoría de modelos

## Propósito

Antes de confiar en las métricas de un notebook de ML, hay que verificar que:
- no hay fuga de datos (data leakage) entre train y test,
- ninguna columna usada para predecir contiene información que solo existiría después del evento que se quiere predecir,
- el desempeño del modelo no oculta una disparidad importante entre subgrupos.

Estos tres problemas son silenciosos: el notebook corre sin errores y produce un número de accuracy/AUC que parece razonable, pero el número puede estar inflado o esconder un sesgo. La auditoría existe para separar "el notebook corrió" de "el resultado es confiable".

No corrijas el notebook ni re-entrenes el modelo. Tu trabajo es leer, verificar con evidencia y reportar — no arreglar el código a menos que el usuario lo pida explícitamente.

## Entradas esperadas

Pide estos datos si no vienen en la solicitud del usuario:

1. **Ruta del notebook** (`.ipynb`) a auditar.
2. **Columna objetivo** (target), p. ej. `"diagnostico"`.
3. **Columna de subgrupo**, si existe (p. ej. `"mean_radius_bin"`, `"sexo"`, `"grupo_etario"`). Si el usuario no sabe si existe, revisa el DataFrame del notebook y pregúntale solo si no es evidente.

Si falta la ruta del notebook, pídela antes de continuar — no hay nada que auditar sin ella.

## Cómo leer el notebook

Usa la herramienta de notebooks para leer las celdas de código **en el orden en que aparecen en el archivo**, no en el orden de ejecución (`execution_count`) si difieren — lo que importa para reproducibilidad es lo que corre alguien que ejecuta el notebook de arriba hacia abajo. Si el orden de ejecución y el orden de las celdas no coinciden, anótalo: es evidencia de que el notebook se editó después de correr y los resultados mostrados pueden no corresponder al código actual.

Lee todas las celdas de código antes de emitir cualquier veredicto. Un check que parece fallar en la celda 5 puede resolverse en la celda 8 (por ejemplo, un split que ocurre antes de lo esperado pero que en realidad es un split exploratorio descartado después).

## Pasos de verificación

Ejecuta estos cuatro checks en orden. Para cada uno, busca evidencia concreta (número de celda y línea o fragmento de código) antes de decidir un veredicto — nunca infieras un PASA por ausencia de evidencia en contra.

### 1. Orden de `train_test_split` vs. `fit()` de escaladores/encoders

**Por qué importa:** si un `StandardScaler`, `MinMaxScaler`, `OneHotEncoder`, `SimpleImputer`, etc. se ajusta (`fit` o `fit_transform`) sobre el dataset completo *antes* de separar train/test, el conjunto de test contamina el conjunto de entrenamiento (el scaler "vio" estadísticas de datos que se supone son desconocidos). Esto infla las métricas reportadas.

**Cómo verificar:**
- Localiza la(s) llamada(s) a `train_test_split` (o equivalente: `KFold`, `StratifiedKFold`, split manual por índice/fecha).
- Localiza cada `fit(` o `fit_transform(` de un transformador (scaler, encoder, imputer, selector de features, PCA, etc.).
- Verifica que todo `fit`/`fit_transform` de un transformador ocurra **después** del split y **solo sobre los datos de entrenamiento** (`X_train`, no `X` o `X_test`). En test/validación solo debe usarse `transform(`, nunca `fit(` ni `fit_transform(`.
- Si hay validación cruzada, verifica que el transformador se ajuste dentro de cada fold (p. ej. dentro de un `Pipeline` pasado a `cross_val_score`), no una sola vez sobre todo el dataset antes del CV.

### 2. Fuga del objetivo en columnas predictoras (target leakage)

**Por qué importa:** si una columna usada como predictor solo se conoce *después* de que ocurre el evento que se predice (o es una función directa/casi directa del target), el modelo "hace trampa" aprendiendo un atajo que no existirá en producción.

**Cómo verificar:**
- Lista las columnas que entran a `X` (features) justo antes del `fit` del modelo.
- Para cada una, pregúntate: ¿esta información estaría disponible en el momento real de la predicción, antes de saber el resultado? Señales de alerta típicas: columnas derivadas del target (`resultado_final`, `dias_hasta_evento`, `fecha_diagnostico` cuando el target es `diagnostico`), IDs que codifican el resultado, o columnas con correlación casi perfecta con el target que no tienen explicación clínica/de negocio.
- Revisa también el paso de ingeniería de features (si existe): una columna agregada calculada usando todo el dataset (incluyendo filas de test) antes del split es una forma sutil de fuga, aunque no sea fuga "del target" en sentido estricto — repórtala como hallazgo relacionado al check 1.

### 3. Reporte de métricas correctas

**Por qué importa:** una métrica calculada sobre el set de entrenamiento, o con un promedio inadecuado para clases desbalanceadas, puede verse bien sin serlo.

**Cómo verificar:**
- Confirma que las métricas finales (`accuracy_score`, `roc_auc_score`, `f1_score`, `mean_squared_error`, etc.) se calculan sobre `y_test`/`y_pred` de test, no sobre datos de entrenamiento.
- Si hay clases desbalanceadas (revisa `value_counts()` del target si aparece en el notebook), verifica que no se reporte solo `accuracy` sin acompañarla de precision/recall/F1 o AUC — un accuracy alto puede ser trivial si una clase domina.

### 4. Disparidad entre subgrupos

**Por qué importa:** un modelo puede tener buen desempeño agregado y mal desempeño (o sesgo sistemático) en un subgrupo específico. Esto no aparece si solo se mira la métrica global.

**Cómo verificar:**
- Si el usuario indicó una columna de subgrupo, confirma que existe en el DataFrame usado para test.
- Si el notebook **no** calcula métricas por subgrupo, calcúlalas tú: para cada valor único de la columna de subgrupo, filtra `y_test`/`y_pred` por ese subgrupo y calcula la(s) misma(s) métrica(s) que el notebook reporta globalmente (usa el mismo modelo y las mismas predicciones ya generadas en el notebook, no reentrenes).
- Señala como hallazgo cualquier subgrupo cuya métrica se desvíe notablemente del promedio global (usa criterio: reporta la diferencia numérica y deja que el lector juzgue si es aceptable, no impongas un umbral arbitrario salvo que el usuario dé uno).
- Si la columna de subgrupo no existe y el usuario no la mencionó como esperada, marca este check como NO SE PUEDE DETERMINAR con esa razón explícita — no lo omitas en silencio.

## Criterios de veredicto

Para cada uno de los 4 checks, emite exactamente uno de estos veredictos:

- **PASA** — verificaste explícitamente que el check se cumple, con evidencia (celda + línea o fragmento).
- **FALLA** — encontraste evidencia concreta de que el problema existe.
- **NO SE PUEDE DETERMINAR** — el notebook no contiene suficiente información para verificar el check (p. ej. una celda que carga datos externos sin mostrar su contenido, o un paso que ocurre fuera del notebook).

Nunca asumas PASA por defecto ni por ausencia de evidencia en contra: la ausencia de evidencia es "NO SE PUEDE DETERMINAR", no "PASA". Esto es lo más importante de esta auditoría — un veredicto sin evidencia citada no es válido.

## Salida: `AUDIT_REPORT.md`

Genera un archivo `AUDIT_REPORT.md` en el mismo directorio que el notebook auditado (o donde indique el usuario), con esta estructura:

```markdown
# Auditoría de modelo — [nombre del notebook]

**Fecha:** [fecha]
**Notebook:** [ruta]
**Columna objetivo:** [nombre]
**Columna de subgrupo:** [nombre o "no aplica"]

## Resumen

[1-2 líneas: cuántos checks pasaron, fallaron o no se pudieron determinar]

## Tabla de verificaciones

| # | Check | Veredicto | Evidencia |
|---|-------|-----------|-----------|
| 1 | Orden train_test_split vs. fit() de escaladores | PASA/FALLA/NO SE PUEDE DETERMINAR | Celda N: `código citado` |
| 2 | Fuga del objetivo en columnas predictoras | ... | ... |
| 3 | Métricas reportadas correctamente | ... | ... |
| 4 | Disparidad entre subgrupos | ... | Tabla de métricas por subgrupo si aplica |

## Hallazgos detallados

[Para cada FALLA o NO SE PUEDE DETERMINAR: explica el problema, cita la celda exacta, y por qué importa]

## Acciones recomendadas

[Lista concreta y accionable, ordenada por prioridad — p. ej. "Mover el fit_transform del StandardScaler después del train_test_split (celda 4)"]
```

Si un check da FALLA, incluye en "Acciones recomendadas" el cambio concreto de código necesario para corregirlo (no solo la descripción del problema).

Al terminar, dile al usuario en el chat cuántos checks pasaron/fallaron/no se pudieron determinar y dónde está el reporte — no repitas toda la tabla en el chat, el detalle vive en el archivo.
