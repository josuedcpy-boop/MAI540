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
- Diagnóstico de calidad de datos (2026-09-14): `datos.csv` tiene 1,669 filas duplicadas exactas de 7,000 (23.8%, probablemente por cómo se generó el archivo académico a partir del dataset original), faltantes solo en `ca` (87, no se usa como predictor) y `thal` (54), y atípicos (IQR) en `trestbps`, `chol`, `oldpeak` y `thalach` que son clínicamente plausibles (no hay valores imposibles como presión/colesterol en 0).
- Decisión explícita (fuera del alcance literal del README): el modelo DESPUÉS de `main.py` usa 14 variables en vez de 10 — se agregó `thal_missing` (indicador de que `thal` era faltante), `RobustScaler` en vez de `StandardScaler` para lidiar con atípicos, y `fbs`, `slope`, `ca` como predictores categóricos adicionales (las 3 variables clínicas del dataset que el README no exigía usar). Con esto el recall subió de 0.80 a 0.84 y el accuracy de 0.83 a 0.88 — la ganancia real vino de agregar esas 3 variables. Se autorizó a propósito, sabiendo que se desvía de "exactamente estas 10 variables; no es necesario buscar ni añadir otras" del README.

## 3. Criterios de Evaluacion
- Tener lo menos possibles de falsos negativos(Persona que sale que no tiene condicion cuando si es possible) es mucho mas peligroso que un falso positivo
-No halla mucha fuga de datos en el preprocesamiento para garantizar el pipeline
- Target = 1 tiene prioridad en las metricas ya que es lo que dice que alguien tiene una condicion, pero no debe ser criterio principal en accuracy.
- Columna prohibida como predictor (fuga de datos): `target` (fuga directa del objetivo). `main.py` rastrea esto y avisa si se cuela en ANTES/DESPUÉS (no detiene la ejecución), y además incluye un experimento "CON FUGA" que agrega `target` a propósito para mostrar el efecto real de la fuga. (`fbs`, `slope`, `ca` ya no están prohibidas — ver decisión en sección 2: ahora son predictores aprobados.) Ver también sección 5 (Restricciones de Seguridad).
- Criterio principal de comparación ANTES/DESPUÉS: Recall de target=1 (no accuracy). `main.py` imprime un veredicto explícito basado en esto.
- Selección de características (2026-09-14, fuera del alcance literal del README): se agregó un modelo "SELECCIONADO" que aplica correlación de Pearson con `target` (calculada solo en train) más conocimiento de dominio, con umbral |r| >= 0.15. Se descartan `trestbps` (r=0.12), `chol` (r=0.06), `fbs` (r=0.02) y `thal_missing` (r=0.02) — coincide con la literatura conocida sobre este dataset, donde `chol` y `fbs` son predictores célebremente débiles pese a su relevancia clínica intuitiva. Quedan 10 variables: `thal`, `exang`, `ca`, `cp`, `thalach`, `oldpeak`, `slope`, `sex`, `age`, `restecg`. Resultado: accuracy 0.858 y recall 0.838 (vs. 0.876/0.843 de DESPUÉS con 14 variables) — casi el mismo desempeño con un modelo más simple e interpretable.

## 4. Cosas a tener en cuenta para futuras sesiones
- Qué NO modificar sin preguntar: la base de datos "datos.csv"
- Pendientes o ideas futuras:

## 5. Restricciones de Seguridad y Privacidad
- **Edad y sexo no públicos:** las variables de edad y sexo no deben aparecer ni verse públicamente, pero sí usarlas para el entrenamiento y predicciones. Si en algún momento se fueran a mostrar públicamente, el programa debe parar el script y enviar un mensaje de error diciendo que información sensitiva está en riesgo. Implementado en `main.py` como `safe_print`: cualquier intento de imprimir un DataFrame/Series con columnas `age` o `sex` detiene la ejecución con `sys.exit(...)`.
- **`target` nunca como predictor:** ver sección 3 (Criterios de Evaluación) para el detalle de cómo se rastrea y demuestra esta regla.
