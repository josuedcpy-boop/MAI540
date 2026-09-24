import sys
from pathlib import Path
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

DATA = Path(__file__).parent / "data" / "datos.csv"

df = pd.read_csv(DATA)

# Corrección (2026-09-20, retroalimentación del profesor): quitar duplicados
# ANTES de dividir train/test. Si se hiciera después, copias exactas de un
# mismo paciente ya habrían quedado repartidas entre ambos conjuntos, y el
# modelo podría "memorizar" en train una fila casi idéntica a una del test.
filas_antes_dedup = len(df)
df = df.drop_duplicates().reset_index(drop=True)
filas_despues_dedup = len(df)
filas_duplicadas_eliminadas = filas_antes_dedup - filas_despues_dedup

# Contexto.md, sección 2 (Datos): reportar valores faltantes antes de que el
# pipeline los impute, para que quede visible qué se está "arreglando".
missing_counts = df.isna().sum()
missing_counts = missing_counts[missing_counts > 0]

# Diagnóstico de datos (faltantes/atípicos) aplicado al modelo DESPUÉS -- esto
# se DESVÍA a propósito de lo que pide literalmente el README (exactamente 10
# variables + StandardScaler), decisión explícita documentada en Contexto.md:
# 1) indicador binario de que "thal" era faltante (variable 11), en vez de
#    solo imputar en silencio con la moda;
# 2) RobustScaler en vez de StandardScaler para las numéricas;
# 3) recorte (winsorizing) de atípicos a los límites del IQR antes de escalar
#    (retroalimentación del profesor, 2026-09-20): RobustScaler por sí solo
#    no "trata" los atípicos, solo evita que dominen la escala -- los valores
#    seguían intactos. Evidencia (BITACORA.md sección 11): recortar da
#    accuracy igual (0.8800) y recall +0.0031 frente a no recortar, así que
#    se adopta el recorte.
df["thal_missing"] = df["thal"].isna().astype(int)


class RecorteIQR(BaseEstimator, TransformerMixin):
    """Recorta (winsoriza) cada columna a [Q1 - k*IQR, Q3 + k*IQR].

    Los límites se calculan únicamente en fit() (datos de train), nunca con
    datos de test, para no filtrar información del conjunto de prueba.
    """

    def __init__(self, k=1.5):
        self.k = k

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        q1, q3 = X.quantile(0.25), X.quantile(0.75)
        iqr = q3 - q1
        self.lower_ = (q1 - self.k * iqr).to_numpy()
        self.upper_ = (q3 + self.k * iqr).to_numpy()
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for i, columna in enumerate(X.columns):
            X[columna] = X[columna].clip(self.lower_[i], self.upper_[i])
        return X.to_numpy()

numeric_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
# Decisión explícita (fuera del alcance literal del README, que pedía exactamente
# 10 variables): se agregan fbs, slope y ca como predictores categóricos también,
# usando las 13 variables clínicas disponibles en vez de solo 10.
categorical_features = ["sex", "cp", "restecg", "exang", "thal", "fbs", "slope", "ca"]
indicator_features = ["thal_missing"]
baseline_features = ["age", "trestbps", "chol", "thalach"]

# Única columna que nunca debe usarse como predictor: el target (fuga directa).
# Prohibición absoluta, sin excepción (Contexto.md, sección 5, 2026-09-20):
# ni el modelo oficial ni experimentos/demostraciones pueden usar "target"
# como predictor. Este chequeo solo confirma que ninguno de los modelos de
# abajo la incluye.
FORBIDDEN_FEATURES = {"target"}
used_features = set(numeric_features) | set(categorical_features) | set(baseline_features)
leaked = used_features & FORBIDDEN_FEATURES

X = df[numeric_features + categorical_features + indicator_features]
y = df["target"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# SELECCIÓN DE CARACTERÍSTICAS (fuera del alcance literal del README, que decía
# que no se requería selección de variables): correlación de Pearson de cada
# predictor con target, calculada SOLO con datos de train para no filtrar
# información del test. Criterio: conservar |r| >= 0.15; el umbral coincide con
# conocimiento de dominio ya documentado sobre este dataset -- chol y fbs son
# célebres por ser predictores débiles de enfermedad cardíaca en Cleveland, y
# nuestro propio indicador thal_missing no aporta señal real.
train_con_target = X_train.copy()
train_con_target["target"] = y_train
correlacion_target = (
    train_con_target.corr(numeric_only=True)["target"].drop("target").sort_values(key=abs, ascending=False)
)
UMBRAL_CORRELACION = 0.15
variables_conservadas = correlacion_target[correlacion_target.abs() >= UMBRAL_CORRELACION].index.tolist()
variables_descartadas = correlacion_target[correlacion_target.abs() < UMBRAL_CORRELACION].index.tolist()
selected_numeric_features = [f for f in numeric_features if f in variables_conservadas]
selected_categorical_features = [f for f in categorical_features if f in variables_conservadas]

# ANTES: punto de partida deliberadamente limitado, solo variables numéricas simples.
baseline_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", LogisticRegression(max_iter=1000, random_state=42)),
])
baseline_model.fit(X_train[baseline_features], y_train)
baseline_pred = baseline_model.predict(X_test[baseline_features])

# DESPUÉS: 10 predictores del README + fbs/slope/ca + 1 indicador de faltante
# ("thal_missing"), con preprocesamiento diferenciado por tipo de variable,
# RobustScaler para atenuar atípicos en las numéricas, y balanceo de clases;
# ajustado únicamente con datos de train.
preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("recorte", RecorteIQR(k=1.5)),
        ("scaler", RobustScaler()),
    ]), numeric_features),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical_features),
    ("ind", "passthrough", indicator_features),
])
final_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")),
])
final_model.fit(X_train, y_train)
final_pred = final_model.predict(X_test)

# SELECCIONADO: mismo preprocesamiento que DESPUÉS, pero solo con las variables
# que pasaron el criterio de correlación (|r| >= 0.15 con target, calculado en train).
selected_preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("recorte", RecorteIQR(k=1.5)),
        ("scaler", RobustScaler()),
    ]), selected_numeric_features),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), selected_categorical_features),
])
selected_model = Pipeline([
    ("preprocessor", selected_preprocessor),
    ("model", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")),
])
selected_model.fit(X_train[selected_numeric_features + selected_categorical_features], y_train)
selected_pred = selected_model.predict(X_test[selected_numeric_features + selected_categorical_features])


# Contexto.md, sección 4: no exponer age/sex de pacientes individuales en la
# salida (sí se usan para entrenar). print_report solo recibe agregados
# (accuracy, precision, recall, F1, matriz de confusión), nunca filas ni
# predicciones por paciente. safe_print es el control técnico: cualquier
# DataFrame/Series que se muestre públicamente debe pasar por aquí; si trae
# columnas sensibles, detiene el script con un error en vez de imprimirlas.
SENSITIVE_COLUMNS = {"age", "sex"}


def safe_print(obj):
    columnas = None
    if isinstance(obj, pd.DataFrame):
        columnas = set(obj.columns)
    elif isinstance(obj, pd.Series):
        columnas = {obj.name}
    expuestas = columnas & SENSITIVE_COLUMNS if columnas else set()
    if expuestas:
        sys.exit(
            "ERROR: información sensible está en riesgo — se intentó mostrar "
            f"públicamente columnas protegidas: {sorted(expuestas)}."
        )
    print(obj)


def print_report(titulo, n_variables, y_true, y_pred):
    print(f"=== {titulo} ===")
    print(f"Variables usadas: {n_variables}")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"F1:        {f1_score(y_true, y_pred):.4f}")
    print("Matriz de confusión:")
    print(confusion_matrix(y_true, y_pred))
    print()


print(f"Filas antes de quitar duplicados: {filas_antes_dedup:,}")
print(f"Filas duplicadas eliminadas (antes del split): {filas_duplicadas_eliminadas:,}")
print(f"Filas después de quitar duplicados: {filas_despues_dedup:,}")
if len(missing_counts) > 0:
    print("Valores faltantes detectados (se imputan dentro del pipeline, solo con datos de train):")
    for columna, cantidad in missing_counts.items():
        print(f"  - {columna}: {cantidad}")
else:
    print("Valores faltantes detectados: ninguno")
print(f"Split: test_size=0.25, random_state=42 (igual para ambos modelos)")
if leaked:
    print(f"ADVERTENCIA - fuga de datos detectada en ANTES/DESPUÉS: columnas prohibidas usadas como predictor: {sorted(leaked)}")
else:
    print(f"Verificación de fuga de datos en ANTES/DESPUÉS: OK (columnas prohibidas no usadas: {sorted(FORBIDDEN_FEATURES)})")
print()
print("AVISO: estos resultados son una predicción/sugerencia estadística, no un diagnóstico")
print("definitivo. No sustituyen evaluación médica ni sugieren tratamiento o medicamentos.")
print()
print_report("ANTES (punto de partida)", len(baseline_features), y_test, baseline_pred)
print_report("DESPUÉS (mejora equilibrada)", len(numeric_features) + len(categorical_features) + len(indicator_features), y_test, final_pred)

# Contexto.md, sección 3: el criterio principal es Recall de target=1 (menos
# falsos negativos), no accuracy, porque no detectar una condición real es
# más grave que una falsa alarma.
recall_antes = recall_score(y_test, baseline_pred)
recall_despues = recall_score(y_test, final_pred)
print("=== VEREDICTO (criterio principal: Recall de target=1) ===")
print(f"Recall ANTES:   {recall_antes:.4f}")
print(f"Recall DESPUÉS: {recall_despues:.4f}")
if recall_despues > recall_antes:
    print(f"Mejora: menos falsos negativos en DESPUÉS (recall subió {recall_despues - recall_antes:+.4f}).")
else:
    print(f"Alerta: el recall no mejoró (cambio de {recall_despues - recall_antes:+.4f}); revisar el modelo.")
print()

print(f"=== SELECCIÓN DE CARACTERÍSTICAS (correlación con target, umbral |r| >= {UMBRAL_CORRELACION}) ===")
print(correlacion_target.round(4).to_string())
print(f"Conservadas ({len(variables_conservadas)}): {variables_conservadas}")
print(f"Descartadas ({len(variables_descartadas)}): {variables_descartadas}")
print()
print_report(
    "SELECCIONADO (solo variables con |r| >= 0.15)",
    len(selected_numeric_features) + len(selected_categorical_features),
    y_test,
    selected_pred,
)
