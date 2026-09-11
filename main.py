import sys
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = Path(__file__).parent / "data" / "datos.csv"

df = pd.read_csv(DATA)

# Contexto.md, sección 2 (Datos): reportar valores faltantes antes de que el
# pipeline los impute, para que quede visible qué se está "arreglando".
missing_counts = df.isna().sum()
missing_counts = missing_counts[missing_counts > 0]

numeric_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
categorical_features = ["sex", "cp", "restecg", "exang", "thal"]
baseline_features = ["age", "trestbps", "chol", "thalach"]

# Columnas que nunca deben usarse como predictor: el target (fuga directa) y las
# variables fuera del alcance de 10 predictores del README (fbs, slope, ca).
# Esto solo registra (tracking) si alguna se cuela en ANTES/DESPUÉS; no detiene
# la ejecución, para poder observar el efecto de la fuga en el experimento de abajo.
FORBIDDEN_FEATURES = {"target", "fbs", "slope", "ca"}
used_features = set(numeric_features) | set(categorical_features) | set(baseline_features)
leaked = used_features & FORBIDDEN_FEATURES

X = df[numeric_features + categorical_features]
y = df["target"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ANTES: punto de partida deliberadamente limitado, solo variables numéricas simples.
baseline_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", LogisticRegression(max_iter=1000, random_state=42)),
])
baseline_model.fit(X_train[baseline_features], y_train)
baseline_pred = baseline_model.predict(X_test[baseline_features])

# DESPUÉS: 10 predictores con preprocesamiento diferenciado por tipo de variable
# y balanceo de clases, ajustado únicamente con los datos de entrenamiento.
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
final_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")),
])
final_model.fit(X_train, y_train)
final_pred = final_model.predict(X_test)

# CON FUGA (demostración a propósito): se agrega "target" como si fuera un
# predictor más, para medir cuánto se infla artificialmente el desempeño
# cuando el objetivo se filtra al modelo. Nunca debe hacerse esto en un
# modelo real; existe solo para comprobar el efecto de la fuga de datos.
leak_numeric_features = numeric_features + ["target"]
X_leak = df[leak_numeric_features + categorical_features]
X_leak_train = X_leak.loc[X_train.index]
X_leak_test = X_leak.loc[X_test.index]

leak_preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), leak_numeric_features),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical_features),
])
leak_model = Pipeline([
    ("preprocessor", leak_preprocessor),
    ("model", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")),
])
leak_model.fit(X_leak_train, y_train)
leak_pred = leak_model.predict(X_leak_test)


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


print(f"Filas: {len(df):,}")
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
print_report("DESPUÉS (mejora equilibrada)", len(numeric_features) + len(categorical_features), y_test, final_pred)

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

print("ADVERTENCIA: el siguiente resultado incluye 'target' como predictor A PROPÓSITO,")
print("solo para demostrar el efecto de la fuga de datos. NUNCA usar así un modelo real.")
print_report("CON FUGA (demostración, NO usar en producción)", len(leak_numeric_features) + len(categorical_features), y_test, leak_pred)
