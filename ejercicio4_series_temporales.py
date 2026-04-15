"""
=============================================================================
PRÁCTICA FINAL — EJERCICIO 4
Análisis y Descomposición de Series Temporales
=============================================================================

DESCRIPCIÓN
-----------
En este ejercicio trabajarás con una serie temporal sintética generada con
una semilla fija. Tendrás que:

  1. Visualizar la serie completa.
  2. Descomponerla en sus componentes: Tendencia, Estacionalidad y Residuo.
  3. Analizar cada componente y responder las preguntas del fichero
     Respuestas.md (sección Ejercicio 4).
  4. Evaluar si el ruido (residuo) se ajusta a un ruido ideal (gaussiano
     con media ≈ 0 y varianza constante).

LIBRERÍAS PERMITIDAS
--------------------
  - numpy, pandas
  - matplotlib, seaborn
  - statsmodels   (para seasonal_decompose y adfuller)
  - scipy.stats   (para el test de normalidad del ruido)

SALIDAS ESPERADAS (carpeta output/)
------------------------------------
  - output/ej4_serie_original.png      → Gráfico de la serie completa
  - output/ej4_descomposicion.png      → Los 4 subgráficos de descomposición
  - output/ej4_acf_pacf.png           → Gráfico ACF y PACF del residuo
  - output/ej4_histograma_ruido.png   → Histograma + curva normal del residuo
  - output/ej4_analisis.txt            → Estadísticos numéricos del análisis

=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("output", exist_ok=True)

GRIS   = "#4A4A4A"
AZUL   = "#4C72B0"
VERDE  = "#2ca02c"
ROJO   = "#E05A2B"


# =============================================================================
# GENERACIÓN DE LA SERIE TEMPORAL SINTÉTICA — NO MODIFICAR ESTE BLOQUE
# =============================================================================

def generar_serie_temporal(semilla=42):
    """
    Genera una serie temporal sintética con componentes conocidos.

    La serie tiene:
      - Una tendencia lineal creciente.
      - Estacionalidad anual (periodo 365 días).
      - Ciclos de largo plazo (periodo ~4 años).
      - Ruido gaussiano.

    Parámetros
    ----------
    semilla : int — Semilla aleatoria para reproducibilidad (NO modificar)

    Retorna
    -------
    serie : pd.Series con índice DatetimeIndex diario (2018-01-01 → 2023-12-31)
    """
    rng = np.random.default_rng(semilla)

    # Índice temporal: 6 años de datos diarios
    fechas = pd.date_range(start="2018-01-01", end="2023-12-31", freq="D")
    n = len(fechas)
    t = np.arange(n)

    # --- Componentes ---
    # 1. Tendencia lineal
    tendencia = 0.05 * t + 50

    # 2. Estacionalidad anual (periodo = 365.25 días)
    estacionalidad = 15 * np.sin(2 * np.pi * t / 365.25) \
                   +  6 * np.cos(4 * np.pi * t / 365.25)

    # 3. Ciclo de largo plazo (periodo ~ 4 años = 1461 días)
    ciclo = 8 * np.sin(2 * np.pi * t / 1461)

    # 4. Ruido gaussiano
    ruido = rng.normal(loc=0, scale=3.5, size=n)

    # Serie completa (modelo aditivo)
    valores = tendencia + estacionalidad + ciclo + ruido

    serie = pd.Series(valores, index=fechas, name="valor")
    return serie


# =============================================================================
# TAREA 1 — Visualizar la serie completa
# =============================================================================

def visualizar_serie(serie):
    """
    Genera y guarda un gráfico de la serie temporal completa.

    Salida: output/ej4_serie_original.png

    Parámetros
    ----------
    serie : pd.Series — La serie temporal a visualizar
    """
    fig, ax = plt.subplots(figsize=(14, 4))

    ax.plot(serie.index, serie.values, color=AZUL, linewidth=0.8, alpha=0.9)

    # Media móvil de 90 días para visualizar la tendencia
    media_movil = serie.rolling(window=90, center=True).mean()
    ax.plot(media_movil.index, media_movil.values, color=ROJO,
            linewidth=2.0, linestyle="--", label="Media móvil 90 días")

    ax.set_title("Serie temporal sintética (2018–2023)",
                 fontweight="bold", color=GRIS, pad=10)
    ax.set_xlabel("Fecha", color=GRIS)
    ax.set_ylabel("Valor", color=GRIS)
    ax.tick_params(labelsize=9, colors=GRIS)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("output/ej4_serie_original.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print("  Guardado: output/ej4_serie_original.png")


# =============================================================================
# TAREA 2 — Descomposición de la serie
# =============================================================================

def descomponer_serie(serie):
    """
    Descompone la serie en Tendencia, Estacionalidad y Residuo usando
    statsmodels.tsa.seasonal.seasonal_decompose y guarda el gráfico.

    Salida: output/ej4_descomposicion.png

    Parámetros
    ----------
    serie : pd.Series — La serie temporal

    Retorna
    -------
    resultado : DecomposeResult — Objeto con atributos .trend, .seasonal, .resid
    """
    from statsmodels.tsa.seasonal import seasonal_decompose

    resultado = seasonal_decompose(serie, model="additive", period=365)

    componentes = {
        "Serie original":   serie,
        "Tendencia":        resultado.trend,
        "Estacionalidad":   resultado.seasonal,
        "Residuo":          resultado.resid,
    }
    colores = [AZUL, ROJO, VERDE, GRIS]

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    for ax, (titulo, datos), color in zip(axes, componentes.items(), colores):
        ax.plot(datos.index, datos.values, color=color, linewidth=0.8)
        ax.set_ylabel(titulo, color=GRIS, fontsize=9)
        ax.tick_params(labelsize=8, colors=GRIS)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_title("Descomposición aditiva de la serie temporal (period=365)",
                      fontweight="bold", color=GRIS, pad=10)
    axes[-1].set_xlabel("Fecha", color=GRIS)

    plt.tight_layout()
    plt.savefig("output/ej4_descomposicion.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print("  Guardado: output/ej4_descomposicion.png")

    return resultado


# =============================================================================
# TAREA 3 — Análisis del residuo (ruido)
# =============================================================================

def analizar_residuo(residuo):
    """
    Analiza el componente de residuo para determinar si se parece
    a un ruido ideal (gaussiano, media ≈ 0, varianza constante, sin autocorr.).

    Genera:
      - output/ej4_acf_pacf.png          → ACF y PACF del residuo
      - output/ej4_histograma_ruido.png  → Histograma + curva normal ajustada
      - output/ej4_analisis.txt          → Estadísticos numéricos

    Parámetros
    ----------
    residuo : pd.Series — Componente de residuo de la descomposición
    """
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.tsa.stattools import adfuller
    from scipy.stats import jarque_bera, norm

    # seasonal_decompose deja NaN en los extremos
    residuo_limpio = residuo.dropna()

    media     = residuo_limpio.mean()
    std       = residuo_limpio.std()
    asimetria = residuo_limpio.skew()
    curtosis  = residuo_limpio.kurtosis()

    stat_jb, p_jb = jarque_bera(residuo_limpio)

    resultado_adf = adfuller(residuo_limpio)
    p_adf = resultado_adf[1]

    print(f"\n  Estadísticos del residuo:")
    print(f"    Media     : {media:.4f}  (ideal ≈ 0)")
    print(f"    Std       : {std:.4f}")
    print(f"    Asimetría : {asimetria:.4f}  (ideal ≈ 0)")
    print(f"    Curtosis  : {curtosis:.4f}  (ideal ≈ 0)")
    print(f"    Jarque-Bera p-value : {p_jb:.4f}  ({'normal' if p_jb > 0.05 else 'no normal'})")
    print(f"    ADF p-value         : {p_adf:.4f}  ({'estacionario' if p_adf < 0.05 else 'no estacionario'})")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    plot_acf(residuo_limpio, lags=60, ax=axes[0], color=AZUL, title="")
    axes[0].set_title("ACF del residuo", fontweight="bold", color=GRIS, pad=8)
    axes[0].set_xlabel("Lag (días)", color=GRIS)
    axes[0].tick_params(labelsize=8, colors=GRIS)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    plot_pacf(residuo_limpio, lags=60, ax=axes[1], color=AZUL, title="",
              method="ywm")
    axes[1].set_title("PACF del residuo", fontweight="bold", color=GRIS, pad=8)
    axes[1].set_xlabel("Lag (días)", color=GRIS)
    axes[1].tick_params(labelsize=8, colors=GRIS)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    fig.suptitle("Autocorrelación del residuo — ¿Ruido blanco?",
                 fontsize=12, fontweight="bold", color=GRIS)
    plt.tight_layout()
    plt.savefig("output/ej4_acf_pacf.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print("  Guardado: output/ej4_acf_pacf.png")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(residuo_limpio, bins=50, color=AZUL, edgecolor="white",
            alpha=0.75, density=True, label="Residuo")

    x = np.linspace(residuo_limpio.min(), residuo_limpio.max(), 200)
    ax.plot(x, norm.pdf(x, media, std), color=ROJO, linewidth=2,
            label=f"Normal(μ={media:.2f}, σ={std:.2f})")

    ax.axvline(0, color=GRIS, linestyle="--", linewidth=1.2, alpha=0.6,
               label="μ = 0 (ideal)")
    ax.set_title("Distribución del residuo vs Normal teórica",
                 fontweight="bold", color=GRIS, pad=10)
    ax.set_xlabel("Valor del residuo", color=GRIS)
    ax.set_ylabel("Densidad", color=GRIS)
    ax.legend(fontsize=9, frameon=False)
    ax.tick_params(labelsize=9, colors=GRIS)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.annotate(f"Jarque-Bera p = {p_jb:.3f}", xy=(0.97, 0.93),
                xycoords="axes fraction", ha="right", fontsize=9, color=GRIS)

    plt.tight_layout()
    plt.savefig("output/ej4_histograma_ruido.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print("  Guardado: output/ej4_histograma_ruido.png")

    with open("output/ej4_analisis.txt", "w", encoding="utf-8") as f:
        f.write("Análisis del Residuo — Serie Temporal Sintética\n")
        f.write("=" * 50 + "\n")
        f.write(f"  Media          : {media:.4f}   (ideal ≈ 0)\n")
        f.write(f"  Std            : {std:.4f}\n")
        f.write(f"  Asimetría      : {asimetria:.4f}   (ideal ≈ 0)\n")
        f.write(f"  Curtosis       : {curtosis:.4f}   (ideal ≈ 0)\n")
        f.write(f"  Jarque-Bera p  : {p_jb:.4f}   ({'normal' if p_jb > 0.05 else 'no normal'})\n")
        f.write(f"  ADF p-value    : {p_adf:.4f}   ({'estacionario' if p_adf < 0.05 else 'no estacionario'})\n")
    print("  Guardado: output/ej4_analisis.txt")


# =============================================================================
# MAIN — Ejecuta el pipeline completo
# =============================================================================

if __name__ == "__main__":

    print("=" * 55)
    print("EJERCICIO 4 — Análisis de Series Temporales")
    print("=" * 55)

    # Paso 1: Generar la serie (NO modificar la semilla)
    SEMILLA = 42
    serie = generar_serie_temporal(semilla=SEMILLA)

    print(f"\nSerie generada:")
    print(f"  Periodo:       {serie.index[0].date()} → {serie.index[-1].date()}")
    print(f"  Observaciones: {len(serie)}")
    print(f"  Media:         {serie.mean():.2f}")
    print(f"  Std:           {serie.std():.2f}")
    print(f"  Min / Max:     {serie.min():.2f} / {serie.max():.2f}")

    # Paso 2: Visualizar la serie completa
    print("\n[1/3] Visualizando la serie original...")
    visualizar_serie(serie)

    # Paso 3: Descomponer
    print("[2/3] Descomponiendo la serie...")
    resultado = descomponer_serie(serie)

    # Paso 4: Analizar el residuo
    print("[3/3] Analizando el residuo...")
    if resultado is not None:
        analizar_residuo(resultado.resid)

    # Resumen de salidas
    print("\nSalidas esperadas en output/:")
    salidas = [
        "ej4_serie_original.png",
        "ej4_descomposicion.png",
        "ej4_acf_pacf.png",
        "ej4_histograma_ruido.png",
        "ej4_analisis.txt",
    ]
    for s in salidas:
        existe = os.path.exists(f"output/{s}")
        estado = "✓" if existe else "✗ (pendiente)"
        print(f"  [{estado}] output/{s}")

    print("\n¡Recuerda completar las respuestas en Respuestas.md!")
