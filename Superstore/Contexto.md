# Contexto del proyecto — Superstore

> Última actualización: 2026-09-28.

## 1. Qué es este proyecto
- Objetivo del proyecto: Predecir `Sales` (ventas en dólares) que generaría un producto dado en cada región del dataset Superstore, a partir de variables conocidas del producto y la orden (categoría/sub-categoría, segmento de cliente, región, modo de envío, fechas). No se usa para estimar una orden puntual, sino de forma **comparativa**: para un producto/categoría dado, se predice el `Sales` esperado en `Central`, `East`, `South` y `West`, y la región con mayor venta predicha es la que se sugiere priorizar para ese producto. Esto apoya la decisión de **en qué región enfocar la venta o distribución de un producto**. El error del modelo se expresa en **dólares** (la misma unidad que `Sales`) — típicamente MAE o RMSE.
- Alcance permitido (qué SÍ se puede hacer, qué NO): El modelo puede sugerir en qué región un producto tiende a vender más, como insumo para priorizar esfuerzo comercial — no debe tratarse como garantía de resultado en una región específica, ni usarse como única base para retirar un producto de una región sin más análisis. Es una comparación estadística entre regiones, sujeta al error del modelo, para apoyar (no reemplazar) el juicio del equipo comercial.

## 2. Datos
- Fuente de los datos: `data/train.csv` (dataset Superstore, 9,800 filas, 18 columnas). Versión aumentada con la columna de fuga: `data/train_con_comision.csv` (mismo contenido + `comision_vendedor`).
- Variable objetivo: `Sales` (ventas en dólares de esa línea de orden). Rango real: $0.44 – $22,638.48, media $230.77, mediana $54.49 (distribución con cola larga a la derecha).
- Predictores disponibles: `Ship Mode`, `Segment`, `Country`, `City`, `State`, `Postal Code`, `Region`, `Category`, `Sub-Category`, `Order Date`, `Ship Date` (más `Order ID`/`Customer ID`/`Product ID`/nombres, que son identificadores, no predictores útiles).
- Particularidades a tener en cuenta: `Postal Code` tiene 11 valores faltantes. No hay filas duplicadas exactas. `Order Date`/`Ship Date` son fechas en texto — no vienen ya convertidas a `datetime`. Varias columnas categóricas tienen alta cardinalidad (`City`, `State`, `Product Name`, `Customer Name`) — usarlas directamente en one-hot sin agrupar puede ser poco práctico.
- Diagnóstico de calidad de datos (2026-09-28): sin duplicados exactos; único faltante relevante es `Postal Code` (11 filas, 0.1%); `Sales` sin valores negativos ni cero (mínimo $0.44).

## 3. Criterios de Evaluación
- Métrica prioritaria: por definir (candidatas: MAE en dólares, más interpretable para negocio; o RMSE, que penaliza más los errores grandes — relevante porque `Sales` tiene cola larga con órdenes muy grandes).
- **Columna prohibida como predictor (fuga de datos): `comision_vendedor`.** Se calcula como un porcentaje de `Sales` (más ruido) — es decir, se conoce *después* de que la venta ya ocurrió. Correlación con `Sales`: 0.985. No debe usarse para predecir `Sales` de una orden nueva, porque en ese momento la venta (y por tanto la comisión) todavía no existe.
- Criterio de comparación entre versiones del modelo: por definir. Además de la métrica de error global (MAE/RMSE), conviene revisar si el modelo predice de forma sensata entre regiones para un mismo producto — si dos versiones del modelo dan el mismo error global pero una invierte el orden de regiones recomendado para varios productos, eso importa más que el número agregado, dado el uso comparativo del modelo (ver sección 1).
- Selección de características: por definir.

## 4. Cosas a tener en cuenta para futuras sesiones
- _(Omisiones deliberadas: se irán documentando a medida que avance el proyecto.)_

## 5. Restricciones
- Variables sensibles: `Customer Name` y `Customer ID` identifican clientes individuales — no deben exponerse públicamente en reportes ni salidas agregadas.
- **`comision_vendedor` nunca debe usarse como predictor** (fuga de datos, ver sección 3).
- Qué NO modificar sin preguntar: `data/train.csv` (el archivo fuente original). Los archivos derivados (como `data/train_con_comision.csv`) sí pueden regenerarse.
- Variable de subgrupo para auditorías de disparidad (check 4 del skill `auditoria-modelos`): `Region` (`Central`, `East`, `South`, `West`) — existe nativamente en el dataset, no es sintética.

## 6. Limitaciones

- **Rango de valores de entrenamiento (no extrapolar fuera de él):**
  - `Sales`: el modelo se entrenó con órdenes entre \$0.44 y \$17,499.95 (media \$217.70, mediana \$53.14; percentil 99 = \$2,072.50). Una orden muy por encima de ese rango (como la venta de \$22,638.48 que sí existe en el dataset completo pero cayó del lado de test) no tiene un equivalente comparable en los datos de entrenamiento — el modelo no debe usarse para estimar el valor de ventas atípicamente grandes.
  - `Order Date`: los datos cubren del 3 de enero de 2015 al 30 de diciembre de 2018 (4 años). El modelo solo usa el mes del pedido (`Order Month`), no el año — no capta tendencias de crecimiento/caída entre 2015 y 2018 ni nada posterior a 2018.
  - Categóricas (`Ship Mode`, `Segment`, `Region`, `Category`, `Sub-Category`): el modelo solo reconoce las categorías vistas en entrenamiento (4 modos de envío, 3 segmentos, 4 regiones, 3 categorías, 17 sub-categorías). Un modo de envío, categoría o sub-categoría nuevo que Superstore introduzca en el futuro no tiene representación en el modelo — se codificaría como "desconocido" y la predicción resultante no es confiable.

- **Subgrupos donde el error es mayor:** la región `South` tiene un error considerablemente mayor que el resto (RMSE \$1,294 vs. \$401-\$759 en las demás regiones; R² de 0.11 vs. 0.24-0.33; ver `AUDIT_REPORT.md`, check 4). Dentro de eso, las combinaciones Región × Sub-Categoría con menos de 30 observaciones en entrenamiento son las menos confiables — en particular `South`/`Copiers` (n=4) y `South`/`Machines` (n=13), que además contienen las ventas más grandes y más variables del dataset. El 7.9% de las predicciones de test en South caen en alguna de esas combinaciones de bajo sustento, más del doble que en cualquier otra región (2.2%-3.6%). Se probaron seis correcciones distintas (interacción Región×Sub-Categoría, `HuberRegressor`, `log(Sales)`, log+interacción, Ridge regularizado con interacción, y winsorización del target) y ninguna cerró la brecha — es una limitación de datos (pocas observaciones + valores extremos), no un problema de qué modelo o transformación se use.

- **Variables que podrían cambiar con el tiempo:** la mezcla de categorías/sub-categorías que vende Superstore, los modos de envío ofrecidos, la política de precios (y por tanto el rango típico de `Sales`), y la demanda relativa entre regiones pueden cambiar de un año a otro. El modelo se entrenó sobre una fotografía histórica (2015-2018) y no se reentrena automáticamente — si el catálogo de productos, las regiones operativas o los rangos de precio cambian significativamente, las predicciones dejan de ser representativas sin que el modelo lo señale por sí solo.

- **Condiciones en las que esta herramienta NO debe usarse para tomar decisiones:**
  - Como única base para decidir si un producto específico se vende o se retira de una región (ver alcance permitido, sección 1) — es un insumo comparativo, no una garantía.
  - Para decisiones sobre `Copiers` o `Machines` en la región South, dado el número de observaciones extremadamente bajo (n=4 y n=13 en entrenamiento) detrás de esa estimación.
  - Para estimar el valor de una orden individual de monto muy alto (fuera del rango de entrenamiento, ver arriba) o de una categoría/sub-categoría/modo de envío que no existía en los datos de entrenamiento.
  - Para proyectar ventas en años posteriores a 2018 sin volver a evaluar si la relación entre las variables predictoras y `Sales` sigue siendo válida.
