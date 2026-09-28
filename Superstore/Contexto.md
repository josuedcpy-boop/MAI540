# Contexto del proyecto — Superstore

> Plantilla generalizada a partir del `Contexto.md` de MAI540 (proyecto de enfermedad cardíaca). Misma estructura de 5 secciones, contenido específico de ese proyecto removido — complétala con los datos reales de Superstore. Este archivo no se autocarga (solo `CLAUDE.md` lo hace); Claude Code lo lee cuando se le pide o cuando explora el proyecto.

## 1. Qué es este proyecto
- Objetivo del proyecto: _(qué predice/explica el modelo — ej. clasificación, regresión, forecasting de ventas)_
- Alcance permitido (qué SÍ se puede hacer, qué NO): _(ej. el modelo puede sugerir/estimar, pero no debe presentarse como una decisión de negocio definitiva sin revisión humana — ajustar según el caso real)_

## 2. Datos
- Fuente de los datos: _(nombre del archivo, ej. Superstore.csv)_
- Variable objetivo: _(nombre exacto de la columna objetivo y qué significa cada valor)_
- Predictores disponibles: _(lista de columnas del dataset)_
- Particularidades a tener en cuenta (valores faltantes, variables categóricas disfrazadas de números, fechas, texto libre, etc.): _(completar tras el diagnóstico de datos)_
- Diagnóstico de calidad de datos: _(fecha) — duplicados, faltantes, atípicos encontrados, con conteos. Ver patrón usado en MAI540/Contexto.md sección 2 como referencia de formato._
- Reglas de imputación/atípicos que se adopten, con su justificación y evidencia (referenciar la sección correspondiente en la bitácora de este proyecto).

## 3. Criterios de Evaluación
- _(Qué métrica es prioritaria y por qué — ej. en un problema de clasificación desbalanceada, definir si importa más precision, recall, F1, o si aplica RMSE/MAE en un problema de regresión.)_
- _(Columnas prohibidas como predictor, si alguna representa fuga de datos — ej. una columna calculada a partir del objetivo, o disponible solo después del evento a predecir.)_
- _(Criterio de comparación entre versiones del modelo — qué se considera "mejora".)_
- _(Selección de características, si aplica, con su criterio y justificación.)_

## 4. Cosas a tener en cuenta para futuras sesiones
- _(Omisiones deliberadas: qué se decidió NO incluir en el alcance, y qué costaría incluirlo — seguir el mismo patrón de MAI540/Contexto.md sección 4.)_

## 5. Restricciones
- _(Variables sensibles que no deben exponerse públicamente, si el dataset tiene alguna — ej. información de clientes identificable.)_
- _(Columnas que nunca deben usarse como predictor por fuga de datos, y si la regla es absoluta o solo de advertencia.)_
- Qué NO modificar sin preguntar: _(archivo(s) de datos fuente de este proyecto)_
- _(Otras restricciones específicas de este proyecto/dataset.)_
