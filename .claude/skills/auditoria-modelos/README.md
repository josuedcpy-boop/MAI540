# Skill: auditoria-modelos

Skill de Claude Code que audita un notebook (`.ipynb`) de clasificación o regresión antes de aceptar sus resultados o métricas como confiables.

## Qué hace

Antes de confiar en el accuracy/AUC/F1 que reporta un notebook de ML, hay que verificar que:

- no hay fuga de datos (*data leakage*) entre entrenamiento y prueba,
- ninguna columna usada como predictor contiene información que solo existiría después del evento que se quiere predecir,
- el desempeño del modelo no oculta una disparidad importante entre subgrupos (fairness),
- si se eligió algún hiperparámetro probando varios valores, esa elección no infla el desempeño final reportado (falta de validación anidada).

Estos cuatro problemas son silenciosos: el notebook corre sin errores y produce un número que parece razonable, pero puede estar inflado o esconder un sesgo. Este skill separa "el notebook corrió" de "el resultado es confiable" — **lee, verifica con evidencia y reporta; no corrige el código ni reentrena modelos**, a menos que se le pida explícitamente después de la auditoría.

## Cuándo se activa

Cuando el usuario pide "revisar", "auditar" o "validar" un notebook de ML; pregunta si un modelo tiene fuga de datos; pide verificar el orden de `train_test_split` frente a un `scaler`/`encoder`; pregunta si el desempeño difiere entre subgrupos; o simplemente pega un notebook y pregunta si sus métricas son confiables — incluso sin usar la palabra "auditoría".

## Los 5 checks que ejecuta

1. **Orden de `train_test_split` vs. `fit()` de transformadores** — que ningún scaler/encoder/imputer se ajuste sobre el dataset completo antes de separar train/test (o, en validación cruzada, que se ajuste dentro de cada fold).
2. **Fuga del objetivo en columnas predictoras** — que ninguna columna de `X` sea una función directa o casi directa del target, ni contenga información solo disponible después del evento a predecir.
3. **Reporte de métricas correctas** — que las métricas finales se calculen sobre el conjunto de prueba (no de entrenamiento), y que no se reporte solo accuracy cuando las clases están desbalanceadas.
4. **Disparidad entre subgrupos** — si existe una columna de subgrupo (demográfica u operacional), calcula el desempeño por subgrupo (con `cross_val_predict` si el notebook solo usa validación cruzada y no tiene un `y_test`/`y_pred` fijo) y falla explícitamente si hay una disparidad notable; si no existe ninguna columna de subgrupo, lo marca como "no se puede determinar" en vez de omitirlo.
5. **Selección de hiperparámetros y validación anidada** — si el notebook elige un hiperparámetro (p. ej. `k` de KNN) probando varios valores, verifica que esa elección no use las mismas particiones que luego se reportan como desempeño final; si no hay ninguna búsqueda de hiperparámetros, se marca como "no se puede determinar".

Cada check recibe un veredicto — **PASA**, **FALLA**, o **NO SE PUEDE DETERMINAR** — siempre con evidencia citada (celda + fragmento de código). Nunca se asume PASA por ausencia de evidencia en contra.

## Entradas que pide

1. Ruta del notebook a auditar (obligatoria).
2. Columna objetivo (target).
3. Columna de subgrupo, si existe — si no es evidente, revisa el DataFrame del notebook y pregunta.

## Salida

Un archivo `AUDIT_REPORT.md`, generado en el mismo directorio que el notebook auditado, con: resumen, tabla de los 5 veredictos con evidencia, hallazgos detallados de cada FALLA/NO SE PUEDE DETERMINAR, y acciones recomendadas (incluyendo el cambio de código concreto cuando un check falla). Si ya existe un `AUDIT_REPORT.md` de una auditoría anterior en esa carpeta, no se sobrescribe — el nuevo se nombra con la fecha entre paréntesis, p. ej. `AUDIT_REPORT (2026-09-27).md`.

## Historial de versiones

### 1.1.0 (actual)

Revisión del skill a partir de usarlo en la práctica sobre notebooks reales de este proyecto (enfermedad cardíaca, biopsias de mama, clasificación de especies de flores). Cambios sobre 1.0.0, sin quitar nada de lo que ya tenía:

- **Lectura del notebook aclarada:** se especifica que el notebook es un archivo `.ipynb` (antes decía "usa la herramienta de notebooks", ambiguo); y se aclara qué hacer si `execution_count` es `null` en todas las celdas (no importa, se sigue el orden del archivo igual) — antes el skill solo cubría el caso de que los órdenes *difirieran*, no el caso de que el metadato faltara por completo.
- **Nuevo check 5 — Selección de hiperparámetros y validación anidada:** verifica que, cuando un notebook elige un hiperparámetro probando varios valores (p. ej. `k` de KNN), esa elección no use las mismas particiones de validación cruzada que luego se reportan como desempeño final — un sesgo optimista distinto a los 4 problemas originales, pero igual de silencioso. Se marca "no se puede determinar" si el notebook no hace ninguna búsqueda de hiperparámetros.
- **Check 4 (disparidad entre subgrupos) ampliado:**
  - Ahora cubre notebooks que solo evalúan con validación cruzada (sin un `y_test`/`y_pred` fijo guardado, como sucede con `cross_val_score`): en ese caso se usa `cross_val_predict` con el mismo modelo y el mismo `cv` para obtener predicciones por muestra, en vez de asumir que siempre existe un split fijo.
  - El veredicto ya no es ambiguo: una disparidad notable ahora se marca explícitamente como **FALLA** (antes solo decía "señálalo como hallazgo", sin aclarar si eso contaba como falla o no).
- **Manejo de `AUDIT_REPORT.md` existente:** si ya hay un reporte de una auditoría anterior en la misma carpeta, ya no se sobrescribe en silencio — el nuevo archivo se nombra con la fecha de la auditoría entre paréntesis (p. ej. `AUDIT_REPORT (2026-09-27).md`), para conservar el historial.

Usado y validado (además de los notebooks de 1.0.0) sobre las versiones corregidas/ampliadas de `Diagnostico_Biopsias_Mama.ipynb` y `App_Comparacion_Clasificadores_Iris.ipynb` (con KNN y Random Forest agregados) — este último expuso en la práctica la necesidad del check 5 y de la extensión del check 4 a flujos de solo validación cruzada.

### 1.0.0

Versión inicial del skill. Incluye:

- Lectura del notebook en orden de archivo (no de `execution_count`), con detección explícita de discrepancias entre ambos órdenes como evidencia de que las salidas guardadas podrían no corresponder al código actual.
- Los 4 checks descritos arriba, cada uno con su propio procedimiento de verificación y ejemplos de señales de alerta.
- Formato fijo de veredicto de tres estados (PASA / FALLA / NO SE PUEDE DETERMINAR).
- Generación automática de `AUDIT_REPORT.md` con plantilla fija (resumen, tabla, hallazgos, acciones recomendadas).
- Alcance explícitamente limitado a auditar y reportar — no corrige código ni reentrena modelos salvo pedido explícito posterior del usuario.

Usado y validado en este proyecto sobre dos notebooks reales: `pipeline_colab.ipynb` (enfermedad cardíaca, MAI540) y `App_Diagnostico_Biopsias_Mama.ipynb`/`Diagnostico_Biopsias_Mama.ipynb` (biopsias de mama) — este último con hallazgos reales de fuga de datos (columna `sesiones_tratamiento_programadas` derivada del target) y de discrepancia entre orden de archivo y orden de ejecución.
