# Bitácora de desarrollo asistido

## 1. Comprensión inicial
- **¿Qué problema resuelve el proyecto?** El proyecto busca predecir si un paciente tiene una enfermedad cardíaca dadas ciertas variables clínicas.
- **¿Qué datos utiliza?** Un archivo `datos.csv` con variables como `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`.
- **¿Cuál es la variable objetivo?** `target` (0 = no tiene la condición, 1 = sí la tiene).
- **¿Qué modelo utiliza actualmente?** Al inicio, una regresión logística deliberadamente limitada, usando solo 4 de las 10 variables disponibles (`age`, `trestbps`, `chol`, `thalach`).
- **¿Qué falta por completar?** Ampliar el modelo a las 10 variables aprobadas por el README, con preprocesamiento diferenciado numérico/categórico y balanceo de clases.

## 2. Ejecución inicial (ANTES)
- **Comando utilizado:** `python main.py`
- **Resultado obtenido:** Al principio me dio un error porque faltaba pandas instalado en el sistema de VS Code. Una vez instalado, corrió bien sin problema.
- **Métricas o salida:**
  ```
  === ANTES (punto de partida) ===
  Variables usadas: 4
  Accuracy: 0.7063
  Precision: 0.7027
  Recall: 0.6239
  F1: 0.6609
  Matriz de confusión:
  [[735 212]
   [302 501]]
  ```
  *(Nota 2026-09-20: esta es la evidencia histórica de la entrega original de Tarea 1.2, sobre las 7,000 filas sin deduplicar. Ver sección 9 para las métricas corregidas después de eliminar duplicados.)*
- **Errores, advertencias o limitaciones observadas:** Un accuracy de aprox. 70% suena bien, pero el README dice que hay problemas: no se estaban usando todas las variables de un paciente, así que no me debía confiar de ese resultado.

## 3. Interacción con Claude Code
Para hacer las mejoras del README, le pedí a Claude Code que las implementara primero de forma normal y que después hiciéramos modificaciones manuales si veía algo raro. El prompt que le di fue: **"Implementa las mejoras sugeridas"**.

- **Mejora del modelo (ANTES → DESPUÉS):** Claude propuso un plan que cumplía exactamente lo pedido: mantener `train_test_split` con `test_size=0.25` y `random_state=42`, construir un `ColumnTransformer` (numérica: `SimpleImputer(median)` + `StandardScaler`; categórica: `SimpleImputer(most_frequent)` + `OneHotEncoder`), usar `LogisticRegression(class_weight="balanced")`, reportar accuracy/precision/recall/F1 para `target=1`, y comparar ANTES vs DESPUÉS con el mismo split. Revisé el plan y lo autoricé; Claude hizo todos los cambios como se pidió y me dejó la plantilla de esta bitácora para que la completara yo.

- **Prueba de fuga de datos:** Ya con las restricciones de prohibir el uso de columnas no requeridas y un chequeo de si se usaron datos prohibidos, corrí el `.py` y los resultados confirmaron que no se usaba ningún dato prohibido del `.csv`, así que no hubo fuga. También le pedí correr un ejemplo de fuga a propósito con la columna `target` — que es una fuga muy importante porque le da la respuesta al programa. Si esa columna se filtra, todas las métricas salen en 100%, lo cual, aunque suene bien, crea un sesgo enorme: si se usara el modelo con clientes nuevos que no vinieron con esa respuesta ya dada, aumentarían mucho los falsos positivos y negativos.

- **Reglas de Contexto 1 y 2 (aviso de que los resultados son sugerencia/predicción, no diagnóstico, y reporte de datos faltantes):** Corrió tal como esperaba — no arregla los valores faltantes sin autorización, solo los menciona, y muestra claramente el aviso sobre los resultados.

- **Reglas de Contexto 3 y 4 (priorizar eliminar falsos negativos, no modificar el `.csv` sin autorización, y esconder sexo/edad si aparecen públicamente):** El programa ahora, en las comparaciones, resalta la mejora en lo más importante: que hayan menos falsos negativos. Como el programa ya de por sí no publicaba datos individuales, le pedí crear una prueba que violara el contexto a propósito para ver cómo respondía. En esa prueba, el programa falló en esconder los datos de sexo y edad — los publicó sin problema. Claude no me dio automáticamente una salvaguarda para evitarlo; eso era algo que yo tenía que especificar en el contexto.

- **Revisión y arreglos:** Le añadí al Contexto.md que si se ve la edad y el sexo públicamente, el programa no debe mostrar los resultados, debe parar el script, y enviar un mensaje de error de "información sensible en riesgo". Claude modificó el `.py` de prueba para probarlo y funcionó bien. Algo interesante fue que, sin que yo se lo pidiera, Claude también añadió las columnas sensibles y el mismo protocolo de protección al `main.py` real, aunque en el código actual es imposible que ocurra esa fuga — dijo que lo dejaba puesto por si acaso, para no correr el riesgo en el futuro.

## 4. Verificación
- Ejecuté `python main.py` después de cada cambio para confirmar que las métricas del ANTES se mantuvieran exactamente iguales (0.7063 de accuracy), lo cual confirmaba que el split de datos seguía bien alineado entre ambos modelos.
- Inspeccioné la salida de consola en cada paso: el reporte de valores faltantes, el aviso de que los resultados son solo predicciones, la verificación de fuga de datos, y el veredicto basado en recall.
- Corrí pruebas deliberadas de violación de las reglas (`test_violacion.py`): una usando una columna prohibida (`ca`), otra forzando que el recall empeorara, y otra intentando mostrar edad/sexo. Las tres dieron la respuesta esperada — advertencia de fuga, alerta de que el recall no mejoró, y finalmente el script deteniéndose con el error de "información sensible en riesgo".
- La evidencia de que funciona: los resultados del ANTES coinciden exactamente con la ejecución original de la Tarea 1.2, y las advertencias/errores solo aparecen cuando efectivamente se viola una regla, nunca antes.

## 5. Resultado (DESPUÉS)
- **Métricas o salida final:**
  ```
  === DESPUÉS (mejora equilibrada) ===
  Variables usadas: 10
  Accuracy: 0.8309
  Precision: 0.8229
  Recall: 0.8045
  F1: 0.8136
  Matriz de confusión:
  [[808 139]
   [157 646]]
  ```
  *(Nota 2026-09-20: también evidencia histórica, previa a la corrección de duplicados. Ver sección 9.)*
- **Comparación con el punto de partida:** Los falsos negativos bajaron de 302 a 157, y las cuatro métricas subieron respecto al ANTES (accuracy 0.7063 → 0.8309, precision 0.7027 → 0.8229, recall 0.6239 → 0.8045, F1 0.6609 → 0.8136).
- **¿Qué mejoró y por qué?** Mejoró porque ahora se usan las 10 variables aprobadas en vez de solo 4, las variables categóricas se manejan correctamente (one-hot encoding en vez de tratarlas como números), las numéricas se escalan, y se usa `class_weight="balanced"` para que el modelo no ignore la clase minoritaria. Eso es justo lo que bajó los falsos negativos, que es el criterio que más importa en este proyecto: no detectar una condición real es más grave que una falsa alarma.

## 6. Explicación propia
El código del programa primero importa las herramientas necesarias para hacer su trabajo. Luego busca en el repositorio la data (`datos.csv`). Con esa data empieza a crear las variables predictoras y arma los conjuntos de prueba y entrenamiento con un split de 0.25 y un `random_state` de 42; esto asegura que el split siempre sea el mismo y que la comparación entre modelos sea justa. El programa hace dos entrenamientos y pruebas: uno como estaba originalmente, sin las mejoras, y otro con las mejoras, para poder comparar los resultados al final. En los datos faltantes le añade el valor más común (o la mediana, según el tipo de variable), y en categorías nuevas que nunca vio, las ignora, poniéndole cero a esos valores como parte de las mejoras. Al final crea un reporte comparando ambas versiones.

Además, después de aplicarle el contexto del proyecto, el programa también: avisa que sus resultados son una predicción/sugerencia estadística y no un diagnóstico definitivo; reporta cuántos valores faltantes hay antes de imputarlos; verifica que no se usen columnas prohibidas como predictor (fuga de datos) y corre un ejemplo a propósito usando `target` como predictor, que muestra cómo las métricas se inflan artificialmente a 100% — evidencia clara de por qué la fuga de datos es peligrosa; prioriza el Recall sobre el accuracy al comparar ANTES y DESPUÉS, porque un falso negativo es más peligroso que un falso positivo en un contexto de salud; y protege la privacidad de la edad y el sexo, deteniendo el script con un mensaje de error si en algún momento se intentaran mostrar públicamente.

## 7. Extensión (14 de septiembre de 2026): Diagnóstico de datos y selección de características

Esta sección documenta trabajo adicional que hicimos después de la entrega de la Tarea 1.2, y que **se sale a propósito** del "Límite de alcance" del README original (que decía que no se requería selección de variables ni ingeniería avanzada de características). Lo dejo documentado por transparencia.

- **Diagnóstico de calidad de datos:** le pedí a Claude que revisara `datos.csv` completo. Encontró tres cosas importantes: (1) 1,669 filas duplicadas exactas de 7,000 (23.8%) — probablemente porque el archivo académico se generó repitiendo registros del dataset original para llegar a 7,000 filas, según ya explicaba `DESCRIPCION.md`; (2) faltantes solo en `ca` (87) y `thal` (54), ya conocidos; (3) valores atípicos (método IQR) en `trestbps`, `chol`, `oldpeak` y `thalach`, pero ninguno clínicamente imposible (no hay presión o colesterol en 0). Decidí no tocar los duplicados por ahora — quedó documentado en `Contexto.md` como una limitación conocida, no como algo que se arregló.

- **Manejo de faltantes y atípicos:** le pedí implementar las sugerencias de Claude para lidiar con los faltantes y atípicos detectados. Se agregó `thal_missing`, un indicador binario de que el valor de `thal` fue imputado (en vez de imputar en silencio), y se cambió `StandardScaler` por `RobustScaler` en las variables numéricas, porque este último usa mediana/IQR y no se deja dominar por los valores extremos. Confirmé que esto se salía de lo que pedía el README (exactamente 10 variables + StandardScaler) y decidí autorizarlo de todas formas.

- **Variables categóricas adicionales:** después le pedí usar las variables categóricas que habíamos dejado fuera (`fbs`, `slope`, `ca` — estas últimas antes estaban en la lista de "prohibidas" justo por no ser parte de las 10 originales). Las agregué como predictores oficiales y quité esa prohibición, dejando solo `target` como columna realmente prohibida (por ser fuga directa). El resultado mejoró bastante: accuracy subió de 0.828 a 0.876 y recall de 0.803 a 0.843 con 14 variables en total.

- **Selección de características:** le pregunté a Claude si aplicar un criterio de selección de variables (justificando con correlación y conocimiento de dominio) ayudaría. Me explicó que iría en sentido contrario a lo que habíamos hecho (que fue agregar variables), pero que podía servir para tener un modelo más simple e interpretable. Se calculó la correlación de Pearson de cada variable con `target`, usando **solo los datos de entrenamiento** para no hacer trampa mirando el test. Con un umbral de |r| >= 0.15 se descartaron `trestbps` (r=0.12), `chol` (r=0.06), `fbs` (r=0.02) y `thal_missing` (r=0.02), y quedaron 10 variables (`thal`, `exang`, `ca`, `cp`, `thalach`, `oldpeak`, `slope`, `sex`, `age`, `restecg`). Me pareció interesante que esto coincide con cosas que se conocen de este dataset específico: `chol` y `fbs` son conocidos por ser predictores clínicamente intuitivos pero estadísticamente débiles en el dataset de Cleveland.

- **Resultado de la selección:** el modelo con 10 variables seleccionadas dio accuracy 0.858 y recall 0.838 — casi idéntico al modelo de 14 variables (0.876/0.843), pero más simple. Esto me deja con una decisión pendiente para justificar: si prefiero el modelo completo (mejor desempeño) o el reducido (más interpretable, casi el mismo desempeño). Por ahora `main.py` reporta los dos para poder comparar. *(Nota 2026-09-20: estos números son de antes de la corrección de duplicados — ver sección 9 para los actualizados.)*

En ese momento, `main.py` terminó con **cuatro modelos** en un solo reporte: ANTES (4 variables, punto de partida fijo), DESPUÉS (14 variables, la mejora equilibrada del README más las extensiones), CON FUGA (demostración de por qué no se debe filtrar el target), y SELECCIONADO (10 variables por correlación). Todas las decisiones que se salen del alcance original del README están documentadas en `Contexto.md` con su justificación y fecha. *(El modelo CON FUGA se eliminó después — ver sección 9.)*

## 8. Pendientes / ideas futuras 
- **Umbral clínico de "alto riesgo":** no definí un umbral como "si probabilidad > 0.7, es alto riesgo". Costo de incluirlo: requeriría criterio médico que ni yo ni Claude tenemos, y le daría al modelo una falsa autoridad diagnóstica — justo lo que la sección 1 de `Contexto.md` ya prohíbe explícitamente ("no puede decir definitivo que la tiene").

## 9. Corrección de fuga por duplicados (retroalimentación del profesor, 20 de septiembre de 2026)

Los duplicados se estaban considerando, pero nunca se llegaron a eliminar del dataset antes de dividir train/test. Si se hubieran quitado *después* del split, copias exactas de un mismo paciente ya habrían quedado repartidas entre entrenamiento y prueba — el modelo podría "memorizar" en train una fila casi idéntica a una que luego se evalúa en test, inflando las métricas sin que se note.

**La corrección:** en `main.py`, ahora se hace `df.drop_duplicates()` inmediatamente después de cargar `datos.csv`, antes de construir `X`/`y` y antes de `train_test_split`.

- **Filas antes:** 7,000. **Filas después:** 5,331 (1,669 duplicados eliminados, 23.8%).
- Esto también cambió los conteos de valores faltantes: `ca` pasó de 87 a 65, `thal` de 54 a 37 — algunas de las filas con faltantes eran duplicados.

**Métricas actualizadas** (reemplazan a las de la sección 7, que quedan como referencia histórica de antes de esta corrección):

| Modelo | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| ANTES (4 variables) | 0.7097 | 0.7164 | 0.6620 | 0.6882 |
| DESPUÉS (14 variables) | 0.8702 | 0.8746 | 0.8543 | 0.8643 |
| SELECCIONADO (10 variables) | 0.8612 | 0.8628 | 0.8481 | 0.8554 |

Algo que me pareció interesante: la selección de características (umbral |r| >= 0.15) dio **exactamente las mismas 10 variables** que antes de la corrección (`thal`, `exang`, `ca`, `cp`, `thalach`, `oldpeak`, `slope`, `sex`, `age`, `restecg`). La conclusión de qué variables importan no cambió — solo cambiaron los números finos de desempeño. Eso me da más confianza en que la selección de características era razonablemente robusta y no un artefacto de los duplicados.

## 10. Comparación de estrategias de imputación para `ca`/`thal`.



- **Estrategia A — `most_frequent` (moda):** accuracy 0.8800, precision 0.8880, recall 0.8605, F1 0.8740.
- **Estrategia B — `constant=-1`** (trata el faltante como su propia categoría, en vez de rellenar con el valor más común): accuracy 0.8702, precision 0.8746, recall 0.8543, F1 0.8643.

**Elegí la Estrategia A (moda)** porque gana en las cuatro métricas, y porque el valor `-1` no corresponde a ningún código clínico real de `ca` o `thal` — es una categoría inventada que el modelo trata como una más, sin que le aporte información real. Con solo 55 faltantes de `ca` y 31 de `thal` en los datos de entrenamiento (de casi 4,000 filas), la moda es una estimación razonable que no distorsiona la variable. `main.py` no necesitó cambios porque ya usaba la moda, lo que faltaba era la evidencia comparativa que respaldara esa elección con números, no solo con "porque Claude lo sugirió".

## 11. Tratamiento real de los atípicos (`trestbps`, `chol`, `oldpeak`, `thalach`)

El profesor señaló que decía haber "tratado" los atípicos, pero en el código solo estaba el cambio a `RobustScaler` — eso atenúa su efecto en la escala, pero no corrige ni elimina nada; los valores extremos seguían intactos. Tenía que decidir de verdad qué hacer con ellos, con evidencia.

Comparé, sobre los mismos datos deduplicados y el mismo split:

- **Sin tratar (la que ya tenía):** solo `RobustScaler`, atípicos intactos.
- **Recorte / winsorizing:** cada valor fuera de `[Q1 - 1.5×IQR, Q3 + 1.5×IQR]` se recorta a ese límite (los límites se calculan solo con datos de entrenamiento), y después se escala con `RobustScaler`.

| Variante | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Sin tratar | 0.8800 | 0.8880 | 0.8605 | 0.8740 |
| Con recorte | 0.8800 | 0.8855 | 0.8636 | 0.8744 |

Cuántos valores se recortaron en train (de 3,998 filas): `trestbps` 120 (3.0%), `chol` 54 (1.4%), `thalach` 19 (0.5%), `oldpeak` 31 (0.8%), `age` 0.

**Decisión: recortar (winsorizing).** No baja el accuracy, y sube el recall (+0.0031) — la métrica que más me importa en este proyecto. También tiene sentido de dominio: no elimino pacientes con valores extremos (que probablemente son justo los casos más graves que quiero detectar), solo evito que un valor extremo aislado tenga un peso desproporcionado en el modelo. No consideré eliminar filas como tercera opción porque el diagnóstico anterior ya había confirmado que estos valores son clínicamente plausibles (no hay errores de captura como presión o colesterol en 0) — eliminarlos habría sido descartar pacientes reales sin ninguna razón técnica válida.

Implementé esto como una clase `RecorteIQR` (transformador de scikit-learn) dentro de `main.py`, insertada entre el imputador y el `RobustScaler` en el pipeline numérico de DESPUÉS y SELECCIONADO. Los límites se calculan únicamente con `fit()` (datos de train), nunca con el test, para no filtrar información.

**Resultado en `main.py` con el recorte aplicado:** DESPUÉS pasó de accuracy 0.8702/recall 0.8543 a **accuracy 0.8740/recall 0.8558**. SELECCIONADO no cambió (0.8612/0.8481) porque las 3 variables numéricas que sobrevivieron la selección de características (`age`, `thalach`, `oldpeak`) casi no tenían atípicos que recortar.
