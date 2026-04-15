"""
================================================================================
PRACTICA FINAL — EJERCICIO 2 Inferencia con Scikit-Learn
================================================================================
DESCRIPCION: Usando el mismo dataset del Ejercicio 1, entrena y evalúa un modelo
de aprendizaje automático supervisado para predecir la variable objetivo.
Variable objetivo (target): le_2023 (esperanza de vida en años, 2023)
LIBRERIAS PERMITIDAS: numpy, pandas, matplotlib, seaborn, scikit-learn, scipy
SALIDAS ESPERADAS (carpeta output):
    output/ej2_metricas_regresion.txt → MAE, RMSE y R² de la regresión lineal
    output/ej2_residuos.png           → Gráfico de residuos del modelo
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

np.random.seed(42)
os.makedirs("output", exist_ok=True)

DATASET_PATH = "data/HDR25_Composite_indices_complete_time_series.csv"

# Excluimos hdi_2023 e ihdi_2023 porque tienen multicolinealidad muy alta
# con el target (r > 0.9) — los dejamos fuera para un modelo más honesto
VARS_NUMERICAS = [
    "eys_2023",       # Años de escolaridad esperados
    "mys_2023",       # Años de escolaridad medios
    "gnipc_2023",     # Ingreso Nacional Bruto per cápita
    "gii_2023",       # Índice de desigualdad de género
    "co2_prod_2023",  # Emisiones CO2 per cápita
]

VARS_CATEGORICAS = ["hdicode", "region"]
TARGET = "le_2023"


def cargar_y_preprocesar(path):
    """
    Carga el dataset y aplica preprocesamiento completo:
    - Elimina filas con nulos en columnas relevantes.
    - Codifica variables categóricas con get_dummies.
    - Escala las variables numéricas con StandardScaler.
    - Divide en Train (80%) y Test (20%) con random_state=42.

    Parametros
    ----------
    path : str
        Ruta al fichero CSV.

    Retorna
    -------
    tuple
        X_train, X_test, y_train, y_test, scaler, feature_names
    """
    df = pd.read_csv(path, encoding="latin-1")
    cols = VARS_CATEGORICAS + VARS_NUMERICAS + [TARGET]
    df = df[cols].dropna()

    print(f"\n  Registros tras eliminar nulos: {len(df)}")

    # Codificar variables categóricas con get_dummies (drop_first evita multicolinealidad)
    df_encoded = pd.get_dummies(df, columns=VARS_CATEGORICAS, drop_first=True)

    # Separar features y target
    X = df_encoded.drop(columns=[TARGET])
    y = df_encoded[TARGET].values
    feature_names = X.columns.tolist()

    # Escalar variables numéricas (solo las originales, no las dummies)
    cols_escalar = VARS_NUMERICAS
    scaler = StandardScaler()
    X[cols_escalar] = scaler.fit_transform(X[cols_escalar])

    # Split 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=0.2, random_state=42
    )

    print(f"  Features totales: {X_train.shape[1]}")
    print(f"  Train: {X_train.shape[0]} muestras  |  Test: {X_test.shape[0]} muestras")

    return X_train, X_test, y_train, y_test, scaler, feature_names


def entrenar_regresion_lineal(X_train, y_train):
    """
    Entrena un modelo de Regresión Lineal con scikit-learn.

    Parametros
    ----------
    X_train : np.ndarray
        Matriz de features de entrenamiento.
    y_train : np.ndarray
        Vector de valores objetivo de entrenamiento.

    Retorna
    -------
    LinearRegression
        Modelo entrenado.
    """
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    return modelo


def evaluar_modelo(modelo, X_test, y_test, feature_names):
    """
    Evalúa el modelo sobre el test set calculando MAE, RMSE y R².
    Muestra los coeficientes más influyentes.
    Guarda las métricas en output/ej2_metricas_regresion.txt.

    Parametros
    ----------
    modelo : LinearRegression
    X_test : np.ndarray
    y_test : np.ndarray
    feature_names : list
        Nombres de las features para identificar los coeficientes.

    Retorna
    -------
    dict
        mae, rmse, r2, y_pred.
    """
    y_pred = modelo.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    # Coeficientes ordenados por importancia (valor absoluto)
    coefs = pd.Series(modelo.coef_, index=feature_names).abs().sort_values(ascending=False)

    print(f"\n  Top-5 variables más influyentes:")
    for nombre, val in coefs.head(5).items():
        print(f"    {nombre:40s}: {val:.4f}")

    # Guardar métricas
    with open("output/ej2_metricas_regresion.txt", "w", encoding="utf-8") as f:
        f.write("Regresión Lineal (Scikit-Learn) — Métricas sobre test set\n")
        f.write("=" * 55 + "\n")
        f.write(f"  MAE  = {mae:.4f}  (error medio absoluto en años)\n")
        f.write(f"  RMSE = {rmse:.4f}  (error cuadrático medio en años)\n")
        f.write(f"  R²   = {r2:.4f}  (varianza explicada por el modelo)\n")
        f.write("\nInterpretación:\n")
        f.write(f"  El modelo explica el {r2*100:.1f}% de la varianza de la esperanza de vida.\n")
        f.write(f"  El error medio de predicción es de {mae:.2f} años.\n")
        f.write("\nCoeficientes del modelo (ordenados por importancia):\n")
        for nombre, coef in pd.Series(modelo.coef_, index=feature_names).sort_values(key=abs, ascending=False).items():
            f.write(f"  {nombre:40s}: {coef:.4f}\n")
        f.write(f"  {'Intercepto':40s}: {modelo.intercept_:.4f}\n")

    print(f"\n  Guardado: output/ej2_metricas_regresion.txt")
    return {"mae": mae, "rmse": rmse, "r2": r2, "y_pred": y_pred}


def plot_residuos(y_test, y_pred, r2):
    """
    Genera el gráfico de residuos (valores predichos en X, residuos en Y).
    Un buen modelo muestra residuos distribuidos aleatoriamente alrededor de 0.
    Guarda en output/ej2_residuos.png.

    Parametros
    ----------
    y_test : np.ndarray
        Valores reales del test set.
    y_pred : np.ndarray
        Predicciones del modelo.
    r2 : float
        Coeficiente de determinación para anotarlo en el gráfico.
    """
    GRIS   = "#4A4A4A"
    AZUL   = "#4C72B0"
    ROJO   = "#E05A2B"

    sns.set_theme(style="white", font_scale=1.0)

    residuos = y_test - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Panel izquierdo: Residuos vs Predichos ---
    axes[0].scatter(y_pred, residuos, color=AZUL, alpha=0.6, edgecolors="white",
                    linewidth=0.4, s=50)
    axes[0].axhline(0, color=ROJO, linestyle="--", linewidth=1.4)
    axes[0].set_xlabel("Valores predichos (años)", color=GRIS)
    axes[0].set_ylabel("Residuos (real − predicho)", color=GRIS)
    axes[0].set_title("Residuos vs Valores predichos", fontweight="bold",
                      color=GRIS, pad=10)
    axes[0].tick_params(labelsize=9, colors=GRIS)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].annotate(f"R² = {r2:.3f}", xy=(0.05, 0.93), xycoords="axes fraction",
                     fontsize=10, color=GRIS, fontweight="bold")

    # --- Panel derecho: Real vs Predicho ---
    min_val = min(y_test.min(), y_pred.min()) - 1
    max_val = max(y_test.max(), y_pred.max()) + 1

    axes[1].scatter(y_test, y_pred, color=AZUL, alpha=0.6, edgecolors="white",
                    linewidth=0.4, s=50)
    axes[1].plot([min_val, max_val], [min_val, max_val], color=ROJO,
                 linestyle="--", linewidth=1.4, label="Predicción perfecta")
    axes[1].set_xlabel("Valores reales (años)", color=GRIS)
    axes[1].set_ylabel("Valores predichos (años)", color=GRIS)
    axes[1].set_title("Valores reales vs Predichos", fontweight="bold",
                      color=GRIS, pad=10)
    axes[1].tick_params(labelsize=9, colors=GRIS)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].legend(fontsize=9, frameon=False)

    fig.suptitle("Diagnóstico del modelo — Regresión Lineal (Scikit-Learn)",
                 fontsize=13, fontweight="bold", color=GRIS, y=1.01)
    plt.tight_layout()
    plt.savefig("output/ej2_residuos.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print("  Guardado: output/ej2_residuos.png")


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("=" * 55)
    print("EJERCICIO 2 — Inferencia con Scikit-Learn")
    print("=" * 55)

    resultado = cargar_y_preprocesar(DATASET_PATH)
    if resultado is None:
        print("ERROR: cargar_y_preprocesar() devolvió None.")
        exit(1)

    X_train, X_test, y_train, y_test, scaler, feature_names = resultado

    modelo = entrenar_regresion_lineal(X_train, y_train)

    metricas = evaluar_modelo(modelo, X_test, y_test, feature_names)

    print(f"\n  Métricas sobre test set:")
    print(f"    MAE  = {metricas['mae']:.4f} años")
    print(f"    RMSE = {metricas['rmse']:.4f} años")
    print(f"    R²   = {metricas['r2']:.4f}")

    plot_residuos(y_test, metricas["y_pred"], metricas["r2"])

    print("\n" + "=" * 55)
    print("Salidas en output/")
    salidas = ["ej2_metricas_regresion.txt", "ej2_residuos.png"]
    for s in salidas:
        existe = os.path.exists(f"output/{s}")
        estado = "✓" if existe else "✗ (pendiente)"
        print(f"  {estado} output/{s}")
