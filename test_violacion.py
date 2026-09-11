"""Script temporal de prueba: viola a propósito las reglas de Contexto.md
secciones 3 y 4, para confirmar si los controles de main.py las detectan.
No es parte del proyecto - bórralo cuando ya no lo necesites.
"""
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

# VIOLACIÓN 1 (sección 3): se agrega "ca" -- columna prohibida -- a las
# variables categóricas, para ver si el tracking de fuga la detecta.
numeric_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
categorical_features = ["sex", "cp", "restecg", "exang", "thal", "ca"]
baseline_features = ["age", "trestbps", "chol", "thalach"]

FORBIDDEN_FEATURES = {"target", "fbs", "slope", "ca"}
used_features = set(numeric_features) | set(categorical_features) | set(baseline_features)
leaked = used_features & FORBIDDEN_FEATURES

X = df[numeric_features + categorical_features]
y = df["target"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

baseline_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", LogisticRegression(max_iter=1000, random_state=42)),
])
baseline_model.fit(X_train[baseline_features], y_train)
baseline_pred = baseline_model.predict(X_test[baseline_features])

print("=== PRUEBA 1: violar la lista de columnas prohibidas (sección 3) ===")
if leaked:
    print(f"ADVERTENCIA - fuga de datos detectada: columnas prohibidas usadas como predictor: {sorted(leaked)}")
else:
    print(f"Verificación de fuga de datos: OK (columnas prohibidas no usadas: {sorted(FORBIDDEN_FEATURES)})")
print()

# VIOLACIÓN 2 (sección 3): forzar que el recall "empeore" en DESPUÉS,
# para ver si el veredicto emite la alerta en vez del mensaje de mejora.
print("=== PRUEBA 2: forzar que el recall empeore (sección 3) ===")
recall_antes = 0.90   # valores inventados a propósito
recall_despues = 0.50  # DESPUÉS "peor" que ANTES
print(f"Recall ANTES:   {recall_antes:.4f}")
print(f"Recall DESPUÉS: {recall_despues:.4f}")
if recall_despues > recall_antes:
    print(f"Mejora: menos falsos negativos en DESPUÉS (recall subió {recall_despues - recall_antes:+.4f}).")
else:
    print(f"Alerta: el recall no mejoró (cambio de {recall_despues - recall_antes:+.4f}); revisar el modelo.")
print()

# VIOLACIÓN 3 (sección 4): intentar mostrar age/sex individuales a través del
# nuevo control safe_print (copiado de main.py), que debe detener el script.
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


print("=== PRUEBA 3: mostrar age/sex individuales vía safe_print (sección 4) ===")
print("Si el control funciona, el script debe detenerse aquí con un error.")
safe_print(df[["age", "sex", "target"]].head(3))
print("(No deberías ver esta línea si safe_print funcionó correctamente.)")
