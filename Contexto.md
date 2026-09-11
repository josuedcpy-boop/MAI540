# Contexto del proyecto

> Plantilla para completar con tus propias palabras. Claude Code lee este archivo automáticamente al iniciar una sesión en este proyecto, así que lo que escribas aquí guiará futuras interacciones.

## 1. Qué es este proyecto
- Objetivo del proyecto: Determinar si un cliente tiene una condicion cardiaca o no.
- Alcance permitido (qué SÍ se puede hacer, qué NO):El programa puede sugerir que un cliente tiene una condicion caridaca, pero no puede decir definitivo que la tiene, ni sugerir medicamentos o diagnistar una condicion especificas, sola sugerir y predecir.

## 2. Datos
- Fuente de los datos: datos.csv
- Variable objetivo: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal,
- Particularidades a tener en cuenta (valores faltantes, variables categóricas disfrazadas de números, etc.): De tener alguna irregularidad en la data como los mencionados , sugerir limpieza de datos, arreglar los datos, de no ser possible ignorar los datos irregular.
- Manejo correcto de variables categóricas disfrazadas de números

## 3. Criterios de Evaluacion
- Tener lo menos possibles de falsos negativos(Persona que sale que no tiene condicion cuando si es possible) es mucho mas peligroso que un falso positivo
-No halla mucha fuga de datos en el preprocesamiento para garantizar el pipeline
- Target = 1 tiene prioridad en las metricas ya que es lo que dice que alguien tiene una condicion, pero no debe ser criterio principal en accuracy.
- Columnas prohibidas como predictor (fuga de datos): `target` (fuga directa del objetivo) y `fbs`, `slope`, `ca` (fuera de las 10 variables aprobadas por el README). `main.py` rastrea esto y avisa si se cuelan en ANTES/DESPUÉS (no detiene la ejecución), y además incluye un experimento "CON FUGA" que agrega `target` a propósito para mostrar el efecto real de la fuga.
- Criterio principal de comparación ANTES/DESPUÉS: Recall de target=1 (no accuracy). `main.py` imprime un veredicto explícito basado en esto.

## 4. Cosas a tener en cuenta para futuras sesiones
- Qué NO modificar sin preguntar: la base de datos "datos.csv"
- Contexto que Claude debería recordar entre sesiones: :Las variables de edad y sexo no deben aparacer ni verse publicamente, pero si usarlas para el entrenamiento y predicciones, de enseñarse publicamente, parar el script y enviar mensaje de error diciendo que Informacion sensitiva esta en riesgo.
- Pendientes o ideas futuras:
