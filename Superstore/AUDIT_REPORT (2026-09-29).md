# Auditoría de modelo — Superstore.py (2ª auditoría, tras la señal de confianza)

**Fecha:** 2026-09-29
**Archivo auditado:** `Superstore.py` (script `.py`, no notebook)
**Columna objetivo:** `Sales`
**Columna de subgrupo:** `Region` (`Central`, `East`, `South`, `West`)

Esta auditoría cubre la versión actualizada del script, después de agregar la sección "ANTES/DESPUÉS" con la señal de "baja confianza" (líneas 170-214), que respondía al hallazgo FALLA de la auditoría anterior (`AUDIT_REPORT.md`, 2026-09-29, primera versión).

## Resumen

**3 de 5 checks pasan** (1, 2, 3) — sin cambios respecto a la auditoría anterior. El check 4 (disparidad entre subgrupos) **sigue en FALLA**: la nueva sección no corrige el error de South, solo lo señala. El check 5 sigue **NO SE PUEDE DETERMINAR** (no hay búsqueda de hiperparámetros).

## Tabla de verificaciones

| # | Check | Veredicto | Evidencia |
|---|-------|-----------|-----------|
| 1 | Orden `train_test_split` vs. `fit()` de transformadores | **PASA** | Sin cambios respecto a la auditoría anterior. La nueva sección de "baja confianza" (líneas 196-198) cuenta filas de `X_train` agrupadas por `Region`/`Sub-Category` (`X_train.groupby(...).size()`) — es una cuenta, no un `fit()` de ningún transformador, y usa exclusivamente `X_train`, nunca `X_test`. No introduce fuga nueva. |
| 2 | Fuga del objetivo en columnas predictoras | **PASA** | Sin cambios. La nueva sección no agrega ninguna columna a `X`/predictoras — solo usa `Region` y `Sub-Category`, que ya eran predictores aprobados. |
| 3 | Métricas reportadas correctamente | **PASA** | Sin cambios — mismas métricas, mismo `y_test`/`pred`. |
| 4 | Disparidad entre subgrupos (`Region`) | **FALLA (sin cambio real, solo mejor documentado)** | Ver "Hallazgos detallados". La tabla "DESPUÉS" (líneas 200-209) es **numéricamente idéntica** a la tabla "ANTES" en MAE/RMSE/R² — South sigue en MAE \$316.33 / RMSE \$1,294.37 / R² 0.1127, igual que en la primera auditoría. Lo único nuevo es la columna `% baja confianza` (South: 7.9%, más del doble que cualquier otra región) y la lista de combinaciones Región×Sub-Categoría con menos de 30 observaciones en train (líneas 211-214). |
| 5 | Selección de hiperparámetros y validación anidada | **NO SE PUEDE DETERMINAR** | Sin cambios — sigue sin haber búsqueda de hiperparámetros en el script. |

## Hallazgos detallados

### Check 4 — La disparidad de South no se corrigió; se documentó

La auditoría anterior recomendó (prioridad alta) reportar MAE/RMSE/R² por región — eso ya se hizo antes de esta segunda auditoría. Esta vez el usuario pidió ir más allá e intentar **corregir** la disparidad. Según la conversación de desarrollo, se probaron cuatro variantes fuera del script (interacción Región×Categoría, `HuberRegressor`, `log(Sales)`, y la combinación de las dos últimas) comparando MAE/RMSE/R² globales y específicos de South contra el modelo base. Ninguna cerró la brecha relativa: la interacción no cambió casi nada (South seguía en ~\$318 de MAE), y tanto `HuberRegressor` como `log(Sales)` bajaron el MAE pero **empeoraron el R² y el RMSE**, tanto global como en South — un trade-off, no una mejora neta.

Ante eso, la decisión documentada en el código (comentario, líneas 189-195) fue no forzar un cambio de modelo, sino agregar transparencia: identificar qué combinaciones Región×Sub-Categoría tienen pocos datos de entrenamiento (`< 30` observaciones, umbral elegido en el código, línea 196) y reportar qué porcentaje de las predicciones de cada región caen en alguna de esas combinaciones. South resultó con 7.9% de sus predicciones de test en combinaciones de bajo sustento (`Bookcases` n=22, `Copiers` n=4, `Fasteners` n=19, `Machines` n=13, `Supplies` n=21, todas en train) — más del doble que Central (3.6%), East (2.2%) o West (2.7%).

**Por qué esto sigue siendo FALLA y no PASA:** el propósito del proyecto (comparar `Sales` esperado entre regiones para decidir dónde enfocar un producto, según `Contexto.md`) sigue viéndose afectado por un error sustancialmente mayor en South. Señalar la baja confianza es una mejora real de transparencia — evita que alguien tome una predicción de South con la misma confianza que una de Central — pero no es una corrección del desempeño en sí. El check audita si hay disparidad de error, no si esa disparidad está bien documentada; la disparidad de error sigue intacta.

## Acciones recomendadas

1. **Prioridad alta (ya no pendiente, verificada):** la señalización de baja confianza por combinación Región×Sub-Categoría ya está implementada y funcionando (líneas 189-214) — esto resuelve la falta de transparencia, aunque no la disparidad de error en sí.
2. **Prioridad media, sigue pendiente:** si se necesita reducir (no solo señalar) el error de South, la vía más prometedora no explorada aún es conseguir más datos para las combinaciones de bajo sustento (en particular `Copiers` en South, con solo 4 observaciones) — ningún ajuste de modelo puede compensar la falta de datos reales.
3. **Prioridad baja:** documentar en `Contexto.md` que las comparaciones entre regiones para `Copiers`/`Machines` en South deben tratarse como orientativas, no como base única de decisión, dado el hallazgo de este check.
4. Los checks 1, 2, 3 y 5 siguen sin requerir acción.
