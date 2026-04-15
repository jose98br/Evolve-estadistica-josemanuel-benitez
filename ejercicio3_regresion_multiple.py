"""
=============================================================================
PRÁCTICA FINAL — EJERCICIO 3
Regresión Lineal Múltiple implementada desde cero con NumPy
=============================================================================

DESCRIPCIÓN
-----------
En este ejercicio debes implementar la función `regresion_lineal_multiple`
que ajusta un modelo de regresión lineal múltiple utilizando la solución
analítica de Mínimos Cuadrados Ordinarios (OLS):

    β = (XᵀX)⁻¹ Xᵀy

La función debe ser capaz de:
  1. Añadir el término independiente (intercepto) automáticamente.
  2. Calcular los coeficientes β₀, β₁, ..., βₙ.
  3. Devolver las predicciones ŷ para un conjunto de datos nuevo.
  4. Calcular las métricas de evaluación: MAE, RMSE y R².

LIBRERÍAS PERMITIDAS
--------------------
  - numpy   (cálculos matriciales)
  - matplotlib (visualización, opcional)

NO está permitido usar sklearn para el ajuste del modelo en este ejercicio.

SALIDAS ESPERADAS (carpeta output/)
------------------------------------
  - output/ej3_coeficientes.txt   → Coeficientes del modelo ajustado
  - output/ej3_metricas.txt       → MAE, RMSE y R² sobre datos de test
  - output/ej3_predicciones.png   → Gráfico Real vs. Predicho

=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("output", exist_ok=True)


# =============================================================================
# FUNCIÓN PRINCIPAL — COMPLETA ESTA SECCIÓN
# =============================================================================

def regresion_lineal_multiple(X_train, y_train, X_test):
    """
    Ajusta un modelo de Regresión Lineal Múltiple usando OLS y devuelve
    las predicciones sobre el conjunto de test.

    La solución analítica es:
        β = (XᵀX)⁻¹ Xᵀy

    Parámetros
    ----------
    X_train : np.ndarray de forma (n_train, p)
        Matriz de features de entrenamiento. Cada fila es una observación
        y cada columna es una variable predictora.
    y_train : np.ndarray de forma (n_train,)
        Vector de valores objetivo de entrenamiento.
    X_test : np.ndarray de forma (n_test, p)
        Matriz de features sobre la que se quiere predecir.

    Retorna
    -------
    coefs : np.ndarray de forma (p+1,)
        Vector de coeficientes ajustados [β₀, β₁, ..., βₚ].
        β₀ es el intercepto (término independiente).
    y_pred : np.ndarray de forma (n_test,)
        Predicciones del modelo sobre X_test.
    """
    # columna de unos para el intercepto
    n_train = X_train.shape[0]
    unos_train = np.ones((n_train, 1))
    X_train_b = np.hstack([unos_train, X_train])

    # β = (XᵀX)⁻¹ Xᵀy — lstsq es más estable que invertir directamente
    coefs, _, _, _ = np.linalg.lstsq(X_train_b, y_train, rcond=None)

    n_test = X_test.shape[0]
    unos_test = np.ones((n_test, 1))
    X_test_b = np.hstack([unos_test, X_test])

    y_pred = X_test_b @ coefs

    return coefs, y_pred


# =============================================================================
# FUNCIONES DE MÉTRICAS — COMPLETA ESTA SECCIÓN
# =============================================================================

def calcular_mae(y_real, y_pred):
    """
    Calcula el Mean Absolute Error (MAE).

        MAE = (1/n) * Σ |y_real - y_pred|

    Parámetros
    ----------
    y_real : np.ndarray — Valores reales
    y_pred : np.ndarray — Valores predichos

    Retorna
    -------
    float — Valor del MAE
    """
    return np.mean(np.abs(y_real - y_pred))


def calcular_rmse(y_real, y_pred):
    """
    Calcula el Root Mean Squared Error (RMSE).

        RMSE = sqrt((1/n) * Σ (y_real - y_pred)²)

    Parámetros
    ----------
    y_real : np.ndarray — Valores reales
    y_pred : np.ndarray — Valores predichos

    Retorna
    -------
    float — Valor del RMSE
    """
    return np.sqrt(np.mean((y_real - y_pred) ** 2))


def calcular_r2(y_real, y_pred):
    """
    Calcula el coeficiente de determinación R².

        R² = 1 - SS_res / SS_tot
        SS_res = Σ (y_real - y_pred)²
        SS_tot = Σ (y_real - ȳ)²

    Parámetros
    ----------
    y_real : np.ndarray — Valores reales
    y_pred : np.ndarray — Valores predichos

    Retorna
    -------
    float — Valor del R² (entre -∞ y 1; cuanto más cercano a 1, mejor)
    """
    ss_res = np.sum((y_real - y_pred) ** 2)
    ss_tot = np.sum((y_real - np.mean(y_real)) ** 2)
    return 1 - ss_res / ss_tot


# =============================================================================
# FUNCIÓN DE VISUALIZACIÓN
# =============================================================================

def graficar_real_vs_predicho(y_real, y_pred, ruta_salida="output/ej3_predicciones.png"):
    """
    Genera un scatter plot de Valores Reales vs. Valores Predichos.

    Un modelo perfecto produciría todos los puntos sobre la diagonal y=x.
    La dispersión alrededor de esa línea representa el error del modelo.

    Parámetros
    ----------
    y_real      : np.ndarray — Valores reales del test set
    y_pred      : np.ndarray — Predicciones del modelo
    ruta_salida : str        — Ruta donde guardar la imagen
    """
    GRIS   = "#4A4A4A"
    AZUL   = "#4C72B0"
    NARANJA = "#E05A2B"

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(y_real, y_pred, color=AZUL, alpha=0.65, edgecolors="white",
               linewidth=0.4, s=55, label="Predicciones")

    # línea y = x: predicción perfecta
    lim_min = min(y_real.min(), y_pred.min()) - 0.5
    lim_max = max(y_real.max(), y_pred.max()) + 0.5
    ax.plot([lim_min, lim_max], [lim_min, lim_max], color=NARANJA,
            linestyle="--", linewidth=1.5, label="Predicción perfecta")

    r2 = calcular_r2(y_real, y_pred)
    ax.annotate(f"R² = {r2:.3f}", xy=(0.05, 0.93), xycoords="axes fraction",
                fontsize=10, color=GRIS, fontweight="bold")

    ax.set_xlabel("Valores reales", color=GRIS)
    ax.set_ylabel("Valores predichos", color=GRIS)
    ax.set_title("Regresión Lineal Múltiple (NumPy)\nValores Reales vs. Predichos",
                 fontweight="bold", color=GRIS, pad=10)
    ax.tick_params(labelsize=9, colors=GRIS)
    ax.legend(fontsize=9, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Guardado: {ruta_salida}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Datos sintéticos con semilla fija para reproducibilidad
    # -------------------------------------------------------------------------
    SEMILLA = 42
    rng = np.random.default_rng(SEMILLA)

    n_muestras = 200
    n_features = 3

    # Generamos features aleatorias
    X = rng.standard_normal((n_muestras, n_features))

    # Coeficientes "reales" conocidos: β₀=5, β₁=2, β₂=-1, β₃=0.5
    coefs_reales = np.array([5.0, 2.0, -1.0, 0.5])

    # Variable objetivo con ruido gaussiano (σ=1.5)
    ruido = rng.normal(0, 1.5, n_muestras)
    y = coefs_reales[0] + X @ coefs_reales[1:] + ruido

    # -------------------------------------------------------------------------
    # Split Train / Test (80% / 20%) — sin mezclar aleatoriamente
    # -------------------------------------------------------------------------
    corte = int(0.8 * n_muestras)
    X_train, X_test = X[:corte], X[corte:]
    y_train, y_test = y[:corte], y[corte:]

    # -------------------------------------------------------------------------
    # Ajuste del modelo
    # -------------------------------------------------------------------------
    coefs, y_pred = regresion_lineal_multiple(X_train, y_train, X_test)

    # -------------------------------------------------------------------------
    # Métricas
    # -------------------------------------------------------------------------
    mae  = calcular_mae(y_test, y_pred)
    rmse = calcular_rmse(y_test, y_pred)
    r2   = calcular_r2(y_test, y_pred)

    # -------------------------------------------------------------------------
    # Mostrar resultados en consola
    # -------------------------------------------------------------------------
    print("=" * 50)
    print("RESULTADOS — Regresión Lineal Múltiple (NumPy)")
    print("=" * 50)
    print(f"\nCoeficientes reales:   {coefs_reales}")
    print(f"Coeficientes ajustados: {coefs}")
    print(f"\nMétricas sobre test set:")
    print(f"  MAE  = {mae:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  R²   = {r2:.4f}")

    # valores de referencia con SEMILLA=42:
    #   coefs ≈ [5.0, 2.0, -1.0, 0.5]
    #   MAE  ≈ 1.20  (±0.20)
    #   RMSE ≈ 1.50  (±0.20)
    #   R²   ≈ 0.80  (±0.05)

    # -------------------------------------------------------------------------
    # Guardar salidas
    # -------------------------------------------------------------------------
    with open("output/ej3_coeficientes.txt", "w", encoding="utf-8") as f:
        f.write("Regresión Lineal Múltiple — Coeficientes ajustados\n")
        f.write("=" * 50 + "\n")
        nombres = ["Intercepto (β₀)"] + [f"β{i+1} (feature {i+1})" for i in range(n_features)]
        for nombre, valor in zip(nombres, coefs):
            f.write(f"  {nombre}: {valor:.6f}\n")
        f.write("\nCoeficientes reales de referencia:\n")
        for nombre, valor in zip(nombres, coefs_reales):
            f.write(f"  {nombre}: {valor:.6f}\n")

    with open("output/ej3_metricas.txt", "w", encoding="utf-8") as f:
        f.write("Regresión Lineal Múltiple — Métricas de evaluación\n")
        f.write("=" * 50 + "\n")
        f.write(f"  MAE  : {mae:.6f}\n")
        f.write(f"  RMSE : {rmse:.6f}\n")
        f.write(f"  R²   : {r2:.6f}\n")

    graficar_real_vs_predicho(y_test, y_pred)

    print("\nSalidas guardadas en la carpeta output/")
    print("  → output/ej3_coeficientes.txt")
    print("  → output/ej3_metricas.txt")
    print("  → output/ej3_predicciones.png")
