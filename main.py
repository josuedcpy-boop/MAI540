from pathlib import Path
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

DATA = Path(__file__).parent / "data" / "datos.csv"

df = pd.read_csv(DATA)

# Punto de partida deliberadamente limitado y comparable con las demás tareas:
# solo usa unas pocas variables sencillas. La mejora completa está en README.md.
features = ['age', 'trestbps', 'chol', 'thalach']
X = df[features]
y = df["target"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", LogisticRegression(max_iter=1000, random_state=42)),
])
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("=== HEART DISEASE: PUNTO DE PARTIDA ===")
print(f"Filas: {len(df):,}")
print(f"Variables usadas por el modelo inicial: {len(features)}")
print(f"Accuracy: {accuracy_score(y_test, pred):.4f}")
print("Matriz de confusión:")
print(confusion_matrix(y_test, pred))
print("\nEl proyecto funciona, pero la mejora equilibrada descrita en README.md todavía no está implementada.")
