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

numeric_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
categorical_features = ["sex", "cp", "restecg", "exang", "thal"]
baseline_features = ["age", "trestbps", "chol", "thalach"]

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
print(f"Split: test_size=0.25, random_state=42 (igual para ambos modelos)\n")
print_report("ANTES (punto de partida)", len(baseline_features), y_test, baseline_pred)
print_report("DESPUÉS (mejora equilibrada)", len(numeric_features) + len(categorical_features), y_test, final_pred)
