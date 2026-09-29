# Informe de depuración — Diagnostico_Biopsias_Mama.ipynb

Documenta el proceso de depuración de los dos defectos del notebook: el error de funcionamiento (`ValueError` de valores infinitos) y la fuga de información en el pipeline. Para cada uno se registran las hipótesis formuladas, cuál se descartó (con evidencia) y cuál se confirmó como causa real.

## Defecto 1: `ValueError: Input X contains infinity or a value too large for dtype('float64')`

El notebook fallaba en la sección "6. Ejecutar el pipeline completo", dentro de `modelo.fit(...)`.

**Hipótesis A — el dato crudo ya viene mal formado.** Antes de sospechar de ninguna transformación propia, la primera hipótesis razonable es que el error venga de los datos tal como se cargan: quizás `load_breast_cancer()` o las columnas sintéticas (`num_biopsias_previas`, `sesiones_tratamiento_programadas`) generadas en `cargar_datos()` ya traen algún valor extremo o corrupto.

*Cómo se descartó:* se corrió `validar_datos()` directamente sobre `df_crudo = cargar_datos()`, revisando `mean texture`, `num_biopsias_previas` y `sesiones_tratamiento_programadas`. Resultado: `{}` — ningún nulo ni infinito. **Hipótesis A descartada** (ver notebook, sección 3, celda `fbef5102`).

**Hipótesis B — la división en `construir_caracteristicas()` produce el infinito.** `variabilidad_por_biopsia = mean texture / num_biopsias_previas` puede dar `inf` si `num_biopsias_previas == 0` para alguna paciente.

*Cómo se confirmó:* se llamó `construir_caracteristicas(df_crudo, arreglar_inf=False)` (el parámetro `arreglar_inf` se agregó específicamente para poder reproducir el comportamiento original) y se corrió `validar_datos()` de nuevo, esta vez sobre `COLUMNAS_PREDICTORAS`. Resultado: `{'variabilidad_por_biopsia': {'nulos': 0, 'infinitos': 141}}`. Al filtrar las filas no finitas, las 141 tenían exactamente `num_biopsias_previas == 0` — ninguna otra combinación de valores produce el problema. **Hipótesis B confirmada.**

**Corrección aplicada:** en vez de arreglarlo dentro de `construir_caracteristicas()` de forma incondicional (lo cual habría obligado a calcular la mediana de imputación sobre todo el dataset, filtrando información de test hacia el entrenamiento), la función ahora acepta `mediana_variabilidad` como parámetro opcional. `entrenar_modelo()` divide train/test **antes** de construir la característica, calcula la mediana únicamente con los datos de entrenamiento, y la reutiliza para test. Se agregó además `validar_datos()` como una aserción real dentro de `entrenar_modelo()` (antes solo se usaba en la celda de diagnóstico de la sección 3) — así cualquier regresión futura de este tipo se detectaría de inmediato, no solo si alguien vuelve a correr el diagnóstico manual.

## Defecto 2: fuga de información en `COLUMNAS_PREDICTORAS`

**Hipótesis A — `num_biopsias_previas` es la columna con fuga.** Es una de las dos columnas "clínicas" agregadas por el equipo; podría estar codificando información posterior al diagnóstico.

*Cómo se descartó:* se calculó la correlación de Pearson entre cada columna sintética y `diagnostico`. `num_biopsias_previas` tiene una correlación de **0.034** — esencialmente ruido, consistente con su construcción real en el código (`rng.randint(0, 4, size=n)`, generada de forma independiente del diagnóstico). **Hipótesis A descartada.**

**Hipótesis B — `sesiones_tratamiento_programadas` es la columna con fuga.**

*Cómo se confirmó:* la misma correlación para `sesiones_tratamiento_programadas` es **-0.932** — casi perfecta. Revisando `cargar_datos()`, se confirma por qué: la columna se construye literalmente como `np.where(df['diagnostico'] == 0, rng.randint(3, 9, size=n), 0)` — vale 0 si y solo si el diagnóstico es benigno. En el momento real de predecir un caso nuevo (ver `predecir_caso()`), el número de sesiones de tratamiento programadas todavía no existiría, porque depende de un diagnóstico que aún no se ha hecho. **Hipótesis B confirmada.**

**Corrección aplicada:** se quitó `sesiones_tratamiento_programadas` de `COLUMNAS_PREDICTORAS` (con un comentario explicando por qué). Para cuantificar el efecto, se entrenó el mismo modelo con y sin la columna, sobre el mismo split (notebook, sección 6b):

| Configuración | Accuracy de prueba |
|---|---|
| Con la columna con fuga | 1.0000 |
| Sin la columna con fuga | 0.7413 |
| **Optimismo que introducía la fuga** | **+0.2587** |

La fuga por sí sola explicaba casi 26 puntos porcentuales de accuracy — el modelo no estaba aprendiendo de las mediciones de la biopsia, estaba leyendo la respuesta directamente en una de las columnas.

## Nota complementaria: efecto del balanceo de clases

Con la fuga ya corregida, se comparó `entrenar_modelo(df, usar_class_weight=False)` contra `usar_class_weight=True)` (notebook, sección 6c):

| Configuración | Accuracy de prueba | Recall de "maligno" |
|---|---|---|
| `usar_class_weight=False` | 0.7343 | 0.5283 |
| `usar_class_weight=True` | 0.7203 | 0.6792 |

El accuracy global en realidad **baja levemente** con `class_weight="balanced"`, mientras que el recall de "maligno" **sube 15 puntos**. Esto tiene sentido: el accuracy pondera todos los errores por igual y la clase mayoritaria (benigno, 62.7%) domina el conteo total, así que mejorar la detección de la clase minoritaria no se refleja — e incluso puede costar algo de accuracy si se sacrifican algunos aciertos en la clase mayoritaria. Aun así, vale la pena usar `balanced`: la métrica que importa clínicamente en este proyecto es el recall de "maligno" (un falso negativo — decirle "benigno" a una paciente con cáncer real — es el error más grave posible aquí), y en esa métrica la mejora es clara e inequívoca, aunque el accuracy no lo muestre.
