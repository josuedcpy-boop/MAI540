from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
DATA = Path(__file__).parent / "data" / "train_con_comision.csv"

df = pd.read_csv(DATA)

# Columnas que nunca deben usarse como predictor (Contexto.md, sección 3):
# - "Sales" es el objetivo, no un predictor.
# - "comision_vendedor" es fuga de datos: se calcula a partir de Sales, así que
#   solo se conoce después de que la venta ya ocurrió.
FORBIDDEN_FEATURES = {"Sales", "comision_vendedor"}

# "Order Month" es estacionalidad conocida de antemano (mes en que se hace el
# pedido). Se calcula fila por fila a partir de Order Date -- no depende de
# ninguna otra fila ni de Sales, así que es seguro calcularla antes del split.
df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d/%m/%Y")
df["Order Month"] = df["Order Date"].dt.month

numeric_features = ["Order Month"]
categorical_features = ["Ship Mode", "Segment", "Region", "Category", "Sub-Category"]

used_features = set(numeric_features) | set(categorical_features)
fuga = used_features & FORBIDDEN_FEATURES
assert not fuga, f"Columnas prohibidas coladas como predictor: {fuga}"

X = df[numeric_features + categorical_features]
y = df["Sales"]

# Split ANTES de ajustar cualquier imputador/escalador/codificador -- todo lo
# que aprenda una estadística de los datos (mediana, media/desviación,
# categorías vistas) debe calcularse solo con X_train.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE
)

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), numeric_features),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical_features),
])

# Tres modelos sobre la MISMA partición train/test:
# (a) DummyRegressor: referencia trivial que siempre predice la media de
#     entrenamiento -- cualquier modelo real debe superarla para justificar
#     su complejidad.
# (b) LinearRegression: modelo lineal simple, rápido de interpretar (cada
#     categoría suma/resta un monto fijo al precio base).
# (c) RandomForestRegressor: puede capturar interacciones no lineales entre
#     categoría/región/estación (ej. que una categoría venda desproporcionadamente
#     más en una región específica) sin tener que definirlas a mano como
#     términos de interacción, y es menos sensible a la cola larga de Sales
#     que un modelo lineal ajustado por mínimos cuadrados.
modelos = {
    "Referencia (media)": DummyRegressor(strategy="mean"),
    "Regresión lineal": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE),
}


def evaluar(nombre, estimador):
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("modelo", estimador),
    ])
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = mean_squared_error(y_test, pred) ** 0.5
    r2 = r2_score(y_test, pred)

    print(f"=== {nombre} ===")
    print(f"MAE:  ${mae:,.2f}")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"R2:   {r2:.4f}")
    print()
    return mae, rmse, r2, pred


print(f"Filas: {len(df):,}")
print(f"Split: test_size=0.25, random_state={RANDOM_STATE}")
print(f"Verificación de fuga de datos: OK (columnas prohibidas no usadas: {sorted(FORBIDDEN_FEATURES)})")
print()

resultados = {nombre: evaluar(nombre, estimador) for nombre, estimador in modelos.items()}

print("=== Comparación ===")
print(f"{'Modelo':<20s} {'MAE':>12s} {'RMSE':>12s} {'R2':>8s}")
for nombre, (mae, rmse, r2, _) in resultados.items():
    print(f"{nombre:<20s} ${mae:>10,.2f} ${rmse:>10,.2f} {r2:>8.4f}")

# --- Analisis de residuos ---
# Se hace sobre la regresion lineal: fue el modelo con mejor MAE/R2 de los dos
# "reales" (le gano a Random Forest en esta corrida), asi que es el candidato
# a usarse de verdad, y por tanto el que mas vale la pena diagnosticar.
mae_lineal, rmse_lineal, r2_lineal, pred_lineal = resultados["Regresión lineal"]
residuos = y_test.to_numpy() - pred_lineal

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(pred_lineal, residuos, alpha=0.25, s=12, edgecolor="none")
ax.axhline(0, color="red", linestyle="--", linewidth=1)
ax.set_xlabel("Valor predicho ($)")
ax.set_ylabel("Residuo = real - predicho ($)")
ax.set_title("Residuos vs. valores predichos — Regresión lineal")
plt.tight_layout()
ruta_dispersión = Path(__file__).parent / "residuos_vs_predichos.png"
plt.savefig(ruta_dispersión, dpi=120)
plt.close()

fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(residuos, bins=60, color="#2980b9", edgecolor="white")
ax.axvline(0, color="red", linestyle="--", linewidth=1)
ax.set_xlabel("Residuo ($)")
ax.set_ylabel("Frecuencia")
ax.set_title("Histograma de residuos — Regresión lineal")
plt.tight_layout()
ruta_histograma = Path(__file__).parent / "histograma_residuos.png"
plt.savefig(ruta_histograma, dpi=120)
plt.close()

print("=== Diagnóstico de residuos (regresión lineal) ===")
print(f"Gráficas guardadas en: {ruta_dispersión.name}, {ruta_histograma.name}")
print()
print(f"Residuo promedio: {residuos.mean():+.2f}")
print(f"Residuo mediano:  {np.median(residuos):+.2f}")
print(f"Percentiles del residuo (5/25/50/75/95): "
      f"{np.percentile(residuos, [5, 25, 50, 75, 95]).round(2)}")
print()

# Heterocedasticidad: ¿el tamaño del error crece con el valor predicho?
corr_dispersión = np.corrcoef(np.abs(residuos), pred_lineal)[0, 1]
print(f"Correlación |residuo| vs. valor predicho: {corr_dispersión:.4f}")

# Sesgo sistemático por grupo: ¿alguna categoría/región queda sesgada?
diagnostico = X_test.copy()
diagnostico["y_real"] = y_test.to_numpy()
diagnostico["residuo"] = residuos
diagnostico["predicho"] = pred_lineal
print()
print("Residuo promedio por Category:")
print(diagnostico.groupby("Category")["residuo"].mean().round(2))
print()
print("Residuo promedio por Region:")
print(diagnostico.groupby("Region")["residuo"].mean().round(2))
print()
print("Residuo promedio por Sub-Category (ordenado):")
print(diagnostico.groupby("Sub-Category")["residuo"].mean().sort_values().round(2))

# El residuo promedio (arriba) solo muestra el SESGO direccional -- dos
# regiones pueden tener el mismo sesgo promedio y un tamaño de error muy
# distinto. Por eso también se reporta MAE/RMSE/R² por región (AUDIT_REPORT.md,
# check 4): South resultó con un error notablemente mayor que las demás.
estadisticas_por_region = {}
for region, grupo in diagnostico.groupby("Region"):
    mae_region = mean_absolute_error(grupo["y_real"], grupo["predicho"])
    rmse_region = mean_squared_error(grupo["y_real"], grupo["predicho"]) ** 0.5
    r2_region = r2_score(grupo["y_real"], grupo["predicho"])
    estadisticas_por_region[region] = (len(grupo), mae_region, rmse_region, r2_region)

print()
print("=== ANTES: error (MAE/RMSE/R2) por Region -- no solo el sesgo promedio ===")
print(f"{'Region':<10s} {'n':>5s} {'MAE':>10s} {'RMSE':>10s} {'R2':>8s} {'MAE vs. global':>16s}")
for region, (n, mae_region, rmse_region, r2_region) in estadisticas_por_region.items():
    print(f"{region:<10s} {n:>5d} ${mae_region:>9,.2f} ${rmse_region:>9,.2f} "
          f"{r2_region:>8.4f} {mae_region - mae_lineal:>+15,.2f}")
print("(Este reporte no dice si alguna región es menos confiable por tener pocos datos de entrenamiento.)")

# --- Corrección: señal de "baja confianza" por combinación Region x Sub-Category
# con pocos datos de entrenamiento (AUDIT_REPORT.md, investigación del hallazgo
# South+Machines: n=18 en train, con la venta más grande de todo el dataset
# adentro). No cambia ninguna predicción -- probamos agregar interacciones,
# HuberRegressor y log(Sales) y ninguna cerró la brecha realmente (ver
# conversación) -- así que en vez de forzar una "corrección" al modelo, se
# etiqueta qué predicciones vienen de una combinación con poco sustento.
UMBRAL_POCOS_DATOS = 30
conteo_train = X_train.groupby(["Region", "Sub-Category"]).size()
combinaciones_pocos_datos = conteo_train[conteo_train < UMBRAL_POCOS_DATOS]

print()
print("=== DESPUÉS: mismo reporte + señal de baja confianza por pocos datos ===")
print(f"{'Region':<10s} {'n':>5s} {'MAE':>10s} {'RMSE':>10s} {'R2':>8s} "
      f"{'MAE vs. global':>16s} {'% baja confianza':>18s}")
for region, (n, mae_region, rmse_region, r2_region) in estadisticas_por_region.items():
    combos_region_bajos = [sub for (reg, sub) in combinaciones_pocos_datos.index if reg == region]
    grupo = diagnostico[diagnostico["Region"] == region]
    pct_baja_confianza = grupo["Sub-Category"].isin(combos_region_bajos).mean() * 100
    print(f"{region:<10s} {n:>5d} ${mae_region:>9,.2f} ${rmse_region:>9,.2f} "
          f"{r2_region:>8.4f} {mae_region - mae_lineal:>+15,.2f} {pct_baja_confianza:>17.1f}%")

print()
print(f"Combinaciones Region x Sub-Category con menos de {UMBRAL_POCOS_DATOS} observaciones en train "
      "(las predicciones ahí deben tratarse como poco confiables, no al mismo nivel que el resto):")
print(combinaciones_pocos_datos.to_string())
