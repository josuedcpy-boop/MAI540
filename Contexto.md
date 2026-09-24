# Contexto del proyecto

> Última actualización: 2026-09-20.

## 1. Qué es este proyecto
- Objetivo del proyecto: Determinar si un cliente tiene una condicion cardiaca o no.
- Alcance permitido (qué SÍ se puede hacer, qué NO):El programa puede sugerir que un cliente tiene una condicion caridaca, pero no puede decir definitivo que la tiene, ni sugerir medicamentos o diagnistar una condicion especificas, sola sugerir y predecir.

## 2. Datos
- Fuente de los datos: datos.csv
- Variable objetivo: `target` (0 = no tiene la condición, 1 = sí la tiene).
- Predictores disponibles: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal.
- Particularidades a tener en cuenta (valores faltantes, variables categóricas disfrazadas de números, etc.): De tener alguna irregularidad en la data como los mencionados , sugerir limpieza de datos, arreglar los datos, de no ser possible ignorar los datos irregular.
- Manejo correcto de variables categóricas disfrazadas de números
- Diagnóstico de calidad de datos (2026-09-14): `datos.csv` tiene 1,669 filas duplicadas exactas de 7,000 (23.8%, probablemente por cómo se generó el archivo académico a partir del dataset original), faltantes solo en `ca` y `thal`, y atípicos (IQR) en `trestbps`, `chol`, `oldpeak` y `thalach` que son clínicamente plausibles (no hay valores imposibles como presión/colesterol en 0).
- **Corrección (2026-09-20):** los 1,669 duplicados ahora se eliminan en `main.py` inmediatamente después de cargar el CSV, **antes** de `train_test_split` — quitarlos después habría dejado copias exactas de un mismo paciente repartidas entre train y test (fuga de datos por fila). Filas antes: 7,000 → después: 5,331. Esto también cambió el conteo de faltantes (`ca`: 87→65, `thal`: 54→37, porque algunas filas con faltantes eran duplicados) y todas las métricas de ANTES/DESPUÉS/SELECCIONADO. Ver `BITACORA.md` para los números actualizados. Esto revierte la omisión deliberada que se había documentado en la sección 4 (ya no aplica).
- **Regla — imputación de `ca`/`thal`:** moda (`most_frequent`), no una categoría constante. Justificación y comparación con evidencia en `BITACORA.md` sección 10.
- **Regla — atípicos numéricos (`trestbps`, `chol`, `oldpeak`, `thalach`):** se recortan (winsorizing) a los límites del IQR calculados solo en train, antes de escalar con `RobustScaler`. Justificación y comparación con evidencia en `BITACORA.md` sección 11.

## 3. Criterios de Evaluacion
- Tener lo menos possibles de falsos negativos(Persona que sale que no tiene condicion cuando si es possible) es mucho mas peligroso que un falso positivo
-No halla mucha fuga de datos en el preprocesamiento para garantizar el pipeline
- Target = 1 tiene prioridad en las metricas ya que es lo que dice que alguien tiene una condicion, pero no debe ser criterio principal en accuracy.
- Columna prohibida como predictor (fuga de datos): `target` (fuga directa del objetivo). `main.py` rastrea esto y avisa si se cuela en ANTES/DESPUÉS (no detiene la ejecución). (`fbs`, `slope`, `ca` ya no están prohibidas — ver decisión en sección 2: ahora son predictores aprobados.) Ver también sección 5 (Restricciones de Seguridad) — la prohibición de `target` es absoluta, sin excepción.
- Criterio principal de comparación ANTES/DESPUÉS: Recall de target=1 (no accuracy). `main.py` imprime un veredicto explícito basado en esto.
- Selección de características (2026-09-14, fuera del alcance literal del README): se agregó un modelo "SELECCIONADO" que aplica correlación de Pearson con `target` (calculada solo en train) más conocimiento de dominio, con umbral |r| >= 0.15. Se descartan `trestbps` (r=0.12), `chol` (r=0.06), `fbs` (r=0.02) y `thal_missing` (r=0.02) — coincide con la literatura conocida sobre este dataset, donde `chol` y `fbs` son predictores célebremente débiles pese a su relevancia clínica intuitiva. Quedan 10 variables: `thal`, `exang`, `ca`, `cp`, `thalach`, `oldpeak`, `slope`, `sex`, `age`, `restecg`. Resultado: accuracy 0.858 y recall 0.838 (vs. 0.876/0.843 de DESPUÉS con 14 variables) — casi el mismo desempeño con un modelo más simple e interpretable.

## 4. Cosas a tener en cuenta para futuras sesiones
- **Omisión deliberada — umbral clínico de "alto riesgo" (2026-09-20):** no se definió un umbral clínico de "alto riesgo" (ej. "si probabilidad > 0.7, es alto riesgo"). Costo de incluirlo: requeriría criterio médico que ni el estudiante ni Claude tienen, y le daría al modelo una falsa autoridad diagnóstica — justo lo que la sección 1 ya prohíbe explícitamente ("no puede decir definitivo que la tiene").
- ~~**Omisión deliberada — filas duplicadas (2026-09-20):** no se especificó qué hacer con las 1,669 filas duplicadas...~~ **Revertido (2026-09-20):** el profesor pidió corregir esto explícitamente. Ver sección 2 — ahora se eliminan antes del split.

## 5. Restricciones
- **Edad y sexo no públicos:** las variables de edad y sexo no deben aparecer ni verse públicamente, pero sí usarlas para el entrenamiento y predicciones. Si en algún momento se fueran a mostrar públicamente, el programa debe parar el script y enviar un mensaje de error diciendo que información sensitiva está en riesgo. Implementado en `main.py` como `safe_print`: cualquier intento de imprimir un DataFrame/Series con columnas `age` o `sex` detiene la ejecución con `sys.exit(...)`.
- **`target` jamás como predictor, sin excepción (2026-09-20):** `target` no debe usarse como variable predictora bajo ninguna circunstancia — ni en el modelo oficial ni en experimentos o demostraciones educativas, aunque estén claramente etiquetados como tales. Esto revierte la decisión anterior de permitir un experimento "CON FUGA" con fines demostrativos: ese bloque se eliminó de `main.py`, y el script `demo_fuga_target.py` se borró. Cualquier solicitud futura de construir un modelo así debe rechazarse citando esta regla, no solo advertirse.
- Qué NO modificar sin preguntar: la base de datos "datos.csv"
- No permitir crear copias de "datos.csv"
