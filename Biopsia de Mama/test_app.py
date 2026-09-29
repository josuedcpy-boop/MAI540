"""Pruebas automatizadas para Diagnostico_Biopsias_Mama.ipynb.

Las funciones viven en el notebook, no en un módulo .py, así que este archivo
carga el notebook con nbformat y ejecuta únicamente las celdas de código que
definen las funciones/constantes necesarias (imports, COLUMNAS_PREDICTORAS,
construir_caracteristicas, cargar_datos, predecir_caso) -- sin ejecutar las
celdas de entrenamiento/gráficas/guardado (lentas y con efectos secundarios
como escribir modelo_biopsias.joblib o mostrar figuras).
"""
from pathlib import Path

import nbformat
import numpy as np
import pytest

NOTEBOOK_PATH = Path(__file__).parent / "Diagnostico_Biopsias_Mama.ipynb"

# IDs de las celdas de código necesarias para tener disponibles las funciones y
# constantes que se prueban aquí (ver el propio notebook para los IDs).
CELDAS_NECESARIAS = {
    "b7d1693a",  # imports (numpy, pandas, sklearn, RANDOM_STATE, rng)
    "0435bcfb",  # COLUMNAS_PREDICTORAS
    "4ad2497e",  # construir_caracteristicas
    "bbb980c2",  # cargar_datos
    "103aad66",  # predecir_caso
}


class _ModeloFalso:
    """Estimador mínimo con .predict() fijo, para probar el contrato de
    predecir_caso() sin depender de un modelo real entrenado."""

    def __init__(self, valor=1):
        self._valor = valor

    def predict(self, X):
        return np.array([self._valor] * len(X))


def _cargar_namespace():
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    # Se pre-siembra "modelo" porque la celda de predecir_caso incluye, al final,
    # una llamada de demostración (predecir_caso(modelo, caso_nuevo)) que de otro
    # modo fallaría con NameError al no haberse ejecutado la celda de entrenamiento.
    namespace = {"modelo": _ModeloFalso()}
    for cell in nb.cells:
        if cell.cell_type == "code" and cell.get("id") in CELDAS_NECESARIAS:
            exec(compile(cell.source, str(NOTEBOOK_PATH), "exec"), namespace)
    return namespace


@pytest.fixture(scope="module")
def ns():
    return _cargar_namespace()


def test_construir_caracteristicas_sin_infinitos_ni_nulos(ns):
    """construir_caracteristicas(), usada con sus parámetros por defecto (los
    que usa el pipeline real), nunca debe dejar infinitos ni nulos."""
    df = ns["cargar_datos"]()
    df_features = ns["construir_caracteristicas"](df)
    columna = df_features["variabilidad_por_biopsia"]
    assert not np.isinf(columna).any(), "construir_caracteristicas() produjo valores infinitos"
    assert not columna.isna().any(), "construir_caracteristicas() produjo valores nulos"


def test_columna_con_fuga_no_esta_en_predictoras(ns):
    """sesiones_tratamiento_programadas es fuga de datos (se deriva del
    diagnóstico) y nunca debe usarse como predictor."""
    assert "sesiones_tratamiento_programadas" not in ns["COLUMNAS_PREDICTORAS"], (
        "La columna con fuga de datos (sesiones_tratamiento_programadas) no debe "
        "usarse como predictor."
    )


def test_predecir_caso_devuelve_benigno_o_maligno(ns):
    """predecir_caso() debe traducir la predicción numérica del modelo (0/1) a
    exactamente 'maligno' o 'benigno', nunca otro valor, para cualquier salida
    posible del modelo."""
    caso = {col: 1.0 for col in ns["COLUMNAS_PREDICTORAS"]}
    casos_esperados = {0: "maligno", 1: "benigno"}
    for valor_crudo, esperado in casos_esperados.items():
        resultado = ns["predecir_caso"](_ModeloFalso(valor_crudo), caso)
        assert resultado in ("benigno", "maligno"), f"Valor inesperado: {resultado!r}"
        assert resultado == esperado
