"""
================================================================================
PRACTICA FINAL — EJERCICIO 1 Análisis Estadístico Descriptivo
================================================================================
DESCRIPCION: Análisis estadístico descriptivo completo sobre el dataset de
Índices Compuestos de Desarrollo Humano (HDR25, PNUD 2025).
Variable objetivo (target): le_2023 (esperanza de vida en años, 2023)
LIBRERIAS PERMITIDAS: numpy, pandas, matplotlib, seaborn, scipy, scikit-learn
SALIDAS ESPERADAS (carpeta output):
    output/ej1_descriptivo.csv         -> Estadísticos descriptivos variables numéricas
    output/ej1_histogramas.png         -> Histogramas de todas las variables numéricas
    output/ej1_boxplots.png            -> Boxplots de la variable objetivo por categoria
    output/ej1_heatmap_correlacion.png -> Mapa de calor de la matriz de correlaciónes
    output/ej1_categoricas.png         -> Gráficos de frecuencia de variables categóricas
    output/ej1_outliers.txt            -> Outliers detectados y decision de tratamiento
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os

np.random.seed(42)
os.makedirs("output", exist_ok=True)

DATASET_PATH = "data/HDR25_Composite_indices_complete_time_series.csv"

VARS_NUMERICAS = [
    "le_2023",        # TARGET: esperanza de vida
    "hdi_2023",       # Índice de Desarrollo Humano
    "eys_2023",       # Anios de escolaridad esperados
    "mys_2023",       # Anios de escolaridad medios
    "gnipc_2023",     # Ingreso Nacional Bruto per capita
    "gii_2023",       # Índice de desigualdad de género
    "co2_prod_2023",  # Emisiones CO2 per capita
    "ihdi_2023",      # HDI ajustado por desigualdad
]

VARS_CATEGORICAS = ["hdicode", "region"]

TARGET = "le_2023"

NOMBRES = {
    "le_2023":       "Esperanza de vida",
    "hdi_2023":      "IDH",
    "eys_2023":      "Años escolaridad esperados",
    "mys_2023":      "Años escolaridad medios",
    "gnipc_2023":    "INB per capita",
    "gii_2023":      "Índice desigualdad género",
    "co2_prod_2023": "CO2 per capita",
    "ihdi_2023":     "IDH ajustado desigualdad",
    "hdicode":       "Nivel de desarrollo",
    "region":        "Region",
}


def cargar_datos(path):
    """
    Carga el dataset y selecciona las columnas relevantes para el análisis.

    Parametros
    ----------
    path : str
        Ruta al fichero CSV.

    Retorna
    -------
    pd.DataFrame
        DataFrame con las columnas seleccionadas.
    """
    df = pd.read_csv(path, encoding="latin-1")
    cols = ["iso3", "country"] + VARS_CATEGORICAS + VARS_NUMERICAS
    return df[cols].copy()


def resumen_estructural(df):
    """
    Imprime información estructural del dataset: filas, columnas,
    tipos de dato y porcentaje de nulos por columna.

    Parametros
    ----------
    df : pd.DataFrame

    Retorna
    -------
    pd.DataFrame
        Tabla con dtype y % nulos por columna.
    """
    print(f"\n{'='*55}")
    print("A) RESUMEN ESTRUCTURAL")
    print(f"{'='*55}")
    print(f"  Filas    : {df.shape[0]}")
    print(f"  Columnas : {df.shape[1]}")
    print(f"  Memoria  : {df.memory_usage(deep=True).sum() / 1024:.1f} KB")

    resumen = pd.DataFrame({
        "dtype":     df.dtypes,
        "nulos":     df.isnull().sum(),
        "pct_nulos": (df.isnull().sum() / len(df) * 100).round(2),
    })
    print(f"\n{resumen.to_string()}")
    return resumen


def estadisticos_descriptivos(df):
    """
    Calcula media, mediana, moda, desviación típica, varianza, min, max,
    cuartiles, IQR, skewness y curtosis para las variables numéricas.
    Guarda la tabla en output/ej1_descriptivo.csv.

    Parametros
    ----------
    df : pd.DataFrame

    Retorna
    -------
    pd.DataFrame
        Tabla de estadísticos descriptivos.
    """
    print(f"\n{'='*55}")
    print("B) ESTADISTICOS DESCRIPTIVOS")
    print(f"{'='*55}")

    datos = df[VARS_NUMERICAS].dropna()

    tabla = pd.DataFrame({
        "media":     datos.mean(),
        "mediana":   datos.median(),
        "moda":      datos.mode().iloc[0],
        "std":       datos.std(),
        "varianza":  datos.var(),
        "min":       datos.min(),
        "max":       datos.max(),
        "Q1":        datos.quantile(0.25),
        "Q3":        datos.quantile(0.75),
        "IQR":       datos.quantile(0.75) - datos.quantile(0.25),
        "skewness":  datos.skew(),
        "curtosis":  datos.kurtosis(),
    }).round(4)

    print(f"\n{tabla.to_string()}")

    # Destacar target
    print(f"\n  >> TARGET ({TARGET}):")
    print(f"     Skewness : {tabla.loc[TARGET, 'skewness']:.4f}  (|<1| = aproximadamente simétrica)")
    print(f"     Curtosis : {tabla.loc[TARGET, 'curtosis']:.4f}  (0 = mesocurtica)")
    print(f"     IQR      : {tabla.loc[TARGET, 'IQR']:.4f} años")

    tabla.to_csv("output/ej1_descriptivo.csv")
    print("\n  Guardado: output/ej1_descriptivo.csv")
    return tabla


def plot_distribuciones(df):
    """
    Genera histogramas con KDE para todas las variables numéricas.
    Guarda en output/ej1_histogramas.png.

    Parametros
    ----------
    df : pd.DataFrame
    """
    print(f"\n{'='*55}")
    print("C) DISTRIBUCIONES — Histogramas")
    print(f"{'='*55}")

    sns.set_theme(style="white", font_scale=1.0)
    GRIS    = "#4A4A4A"
    AZUL    = "#4C72B0"
    NARANJA = "#E05A2B"

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()

    for i, var in enumerate(VARS_NUMERICAS):
        datos = df[var].dropna()
        ax = axes[i]

        sns.histplot(datos, kde=True, ax=ax, color=AZUL, edgecolor="white",
                     alpha=0.75, linewidth=0.4)

        # Líneas de media y mediana
        ax.axvline(datos.mean(),   color=NARANJA, linestyle="--", linewidth=1.4,
                   label=f"Media {datos.mean():.1f}")
        ax.axvline(datos.median(), color=GRIS,    linestyle=":",  linewidth=1.4,
                   label=f"Mediana {datos.median():.1f}")

        ax.set_title(NOMBRES[var], fontsize=10, fontweight="bold", color=GRIS, pad=6)
        ax.set_xlabel("")
        ax.set_ylabel("Frecuencia", fontsize=8, color=GRIS)
        ax.tick_params(labelsize=8, colors=GRIS)
        ax.legend(fontsize=7, frameon=False)

        # Quitar bordes superiores y derechos
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Distribución de variables numéricas — HDR25 (2023)",
                 fontsize=13, fontweight="bold", color=GRIS, y=1.01)
    plt.tight_layout()
    plt.savefig("output/ej1_histogramas.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print("  Guardado: output/ej1_histogramas.png")


def detectar_outliers(df):
    """
    Detecta outliers en todas las variables numéricas usando el metodo IQR.
    Muestra los paises afectados en el TARGET y guarda el informe.
    Decision: se mantienen los outliers (son paises reales, no errores).
    Guarda en output/ej1_outliers.txt.

    Parametros
    ----------
    df : pd.DataFrame

    Retorna
    -------
    pd.DataFrame
        DataFrame original sin modificar (outliers conservados).
    """
    print(f"\n{'='*55}")
    print("C) OUTLIERS — Metodo IQR")
    print(f"{'='*55}")

    lineas = ["DETECCION DE OUTLIERS — Metodo IQR\n", "=" * 50 + "\n"]

    for var in VARS_NUMERICAS:
        datos = df[var].dropna()
        Q1 = datos.quantile(0.25)
        Q3 = datos.quantile(0.75)
        IQR = Q3 - Q1
        lim_inf = Q1 - 1.5 * IQR
        lim_sup = Q3 + 1.5 * IQR

        outliers = df[(df[var] < lim_inf) | (df[var] > lim_sup)][["country", var]]
        n_out = len(outliers)

        linea = f"\n{NOMBRES[var]} ({var}):\n"
        linea += f"  Limites: [{lim_inf:.3f}, {lim_sup:.3f}]\n"
        linea += f"  Outliers detectados: {n_out}\n"
        if n_out > 0:
            for _, row in outliers.iterrows():
                linea += f"    - {row['country']}: {row[var]:.3f}\n"
        lineas.append(linea)
        print(f"  {NOMBRES[var]}: {n_out} outliers")

    decision = (
        "\n" + "=" * 50 + "\n"
        "DECISION DE TRATAMIENTO:\n"
        "Los outliers detectados corresponden a paises reales con valores\n"
        "extremos justificados (ej: paises con muy alto INB como Singapur,\n"
        "o paises con muy bajo IDH como Sudan del Sur). No son errores de\n"
        "medicion sino variabilidad real del fenomeno. Se CONSERVAN todos.\n"
    )
    lineas.append(decision)
    print(f"\n  Decision: outliers conservados (son valores reales, no errores)")

    with open("output/ej1_outliers.txt", "w", encoding="utf-8") as f:
        f.writelines(lineas)
    print("  Guardado: output/ej1_outliers.txt")

    return df


def plot_boxplots(df):
    """
    Genera boxplots de la variable objetivo (TARGET) segmentados
    por cada variable categórica.
    Guarda en output/ej1_boxplots.png.

    Parametros
    ----------
    df : pd.DataFrame
    """
    print(f"\n{'='*55}")
    print("C) BOXPLOTS por variable categórica")
    print(f"{'='*55}")

    GRIS  = "#4A4A4A"
    sns.set_theme(style="white", font_scale=1.0)

    NOMBRES_REGION = {
        "SSA": "África Subsahariana",
        "LAC": "Latinoamérica",
        "EAP": "Asia-Pacífico",
        "AS":  "Asia Occidental",
        "ECA": "Europa Central/Asia",
        "SA":  "Asia Meridional",
    }

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # --- Boxplot por nivel de desarrollo ---
    orden_hdi = ["Low", "Medium", "High", "Very High"]
    datos_hdi = df[df["hdicode"].isin(orden_hdi)]
    paleta_hdi = ["#d4e6f1", "#7fb3d3", "#2980b9", "#1a5276"]

    sns.boxplot(
        x="hdicode", y=TARGET, hue="hdicode", data=datos_hdi,
        order=orden_hdi, palette=paleta_hdi, ax=axes[0],
        legend=False, linewidth=1.2, flierprops={"marker": "o", "markersize": 4,
        "markerfacecolor": "white", "markeredgecolor": GRIS, "alpha": 0.7}
    )
    # Anotar mediana dentro de cada caja (color blanco en cajas oscuras)
    colores_texto = ["white", "white", "white", GRIS]
    for idx, (nivel, color_txt) in enumerate(zip(orden_hdi, colores_texto)):
        mediana = datos_hdi[datos_hdi["hdicode"] == nivel][TARGET].median()
        axes[0].text(idx, mediana, f"{mediana:.1f}", ha="center",
                     va="center", fontsize=8, color=color_txt, fontweight="bold")

    axes[0].set_title("Esperanza de vida por nivel de desarrollo",
                      fontweight="bold", color=GRIS, pad=10)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Esperanza de vida (años)", color=GRIS)
    axes[0].tick_params(labelsize=9, colors=GRIS)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # --- Boxplot por región (ordenado por mediana) ---
    df_region = df.dropna(subset=["region"]).copy()
    df_region["region_nombre"] = df_region["region"].map(NOMBRES_REGION)
    orden_region = (df_region.groupby("region_nombre")[TARGET]
                    .median().sort_values(ascending=False).index.tolist())

    sns.boxplot(
        x="region_nombre", y=TARGET, hue="region_nombre", data=df_region,
        order=orden_region, palette="Set2", ax=axes[1],
        legend=False, linewidth=1.2, flierprops={"marker": "o", "markersize": 4,
        "markerfacecolor": "white", "markeredgecolor": GRIS, "alpha": 0.7}
    )
    # Anotar mediana
    for idx, region in enumerate(orden_region):
        mediana = df_region[df_region["region_nombre"] == region][TARGET].median()
        axes[1].text(idx, mediana + 0.4, f"{mediana:.1f}", ha="center",
                     va="bottom", fontsize=8, color=GRIS, fontweight="bold")

    axes[1].set_title("Esperanza de vida por región (ordenado por mediana)",
                      fontweight="bold", color=GRIS, pad=10)
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Esperanza de vida (años)", color=GRIS)
    axes[1].tick_params(axis="x", rotation=20, labelsize=8)
    axes[1].tick_params(axis="y", labelsize=9, colors=GRIS)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    fig.suptitle("Distribución de la esperanza de vida por variables categóricas",
                 fontsize=13, fontweight="bold", color=GRIS, y=1.01)
    plt.tight_layout()
    plt.savefig("output/ej1_boxplots.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print("  Guardado: output/ej1_boxplots.png")


def analisis_categoricas(df):
    """
    Calcula frecuencia absoluta y relativa de cada categoria.
    Genera gráficos de barras para cada variable categórica.
    Analiza si alguna categoria domina (desbalance).
    Guarda en output/ej1_categoricas.png.

    Parametros
    ----------
    df : pd.DataFrame
    """
    print(f"\n{'='*55}")
    print("D) VARIABLES CATEGORICAS")
    print(f"{'='*55}")

    GRIS = "#4A4A4A"
    sns.set_theme(style="white", font_scale=1.0)

    NOMBRES_REGION = {
        "SSA": "África\nSubsahariana",
        "LAC": "Latinoamérica",
        "EAP": "Asia-Pacífico",
        "AS":  "Asia\nOccidental",
        "ECA": "Europa\nCentral/Asia",
        "SA":  "Asia\nMeridional",
    }
    NOMBRES_HDI = {
        "Very High":                     "Muy alto",
        "High":                          "Alto",
        "Medium":                        "Medio",
        "Low":                           "Bajo",
        "Other Countries or Territories": "Otros",
    }

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for i, var in enumerate(VARS_CATEGORICAS):
        freq_abs = df[var].value_counts()
        freq_rel = df[var].value_counts(normalize=True) * 100

        print(f"\n  {NOMBRES[var]} ({var}):")
        for cat in freq_abs.index:
            print(f"    {cat:30s}: {freq_abs[cat]:3d}  ({freq_rel[cat]:.1f}%)")

        max_pct = freq_rel.max()
        if max_pct > 50:
            print(f"  ⚠ Desbalance: '{freq_rel.idxmax()}' domina con {max_pct:.1f}%")
        else:
            print(f"  ✓ Sin desbalance notable (max: {max_pct:.1f}%)")

        # Traducir etiquetas
        if var == "region":
            etiquetas = [NOMBRES_REGION.get(c, c) for c in freq_abs.index]
        else:
            etiquetas = [NOMBRES_HDI.get(c, c) for c in freq_abs.index]

        colores = sns.color_palette("Blues_d", len(freq_abs))
        barras = axes[i].bar(etiquetas, freq_abs.values, color=colores,
                             edgecolor="white", linewidth=0.5)

        # Etiquetas encima de cada barra
        for barra, val, pct in zip(barras, freq_abs.values, freq_rel.values):
            axes[i].text(barra.get_x() + barra.get_width() / 2,
                         val + 0.3, f"{val}\n({pct:.1f}%)",
                         ha="center", va="bottom", fontsize=8, color=GRIS)

        axes[i].set_title(NOMBRES[var], fontweight="bold", color=GRIS, pad=10)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Número de países", color=GRIS)
        axes[i].tick_params(labelsize=8, colors=GRIS)
        axes[i].spines["top"].set_visible(False)
        axes[i].spines["right"].set_visible(False)
        axes[i].set_ylim(0, freq_abs.max() * 1.2)

    fig.suptitle("Frecuencia de variables categóricas — HDR25 (2023)",
                 fontsize=13, fontweight="bold", color=GRIS, y=1.01)
    plt.tight_layout()
    plt.savefig("output/ej1_categoricas.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print("\n  Guardado: output/ej1_categoricas.png")


def analisis_correlaciones(df):
    """
    Calcula la matriz de correlaciónes de Pearson entre variables numéricas.
    Genera un heatmap e identifica las top-3 correlaciónes con el TARGET.
    Detecta multicolinealidad (pares con |r| > 0.9).
    Guarda en output/ej1_heatmap_correlacion.png.

    Parametros
    ----------
    df : pd.DataFrame
    """
    print(f"\n{'='*55}")
    print("E) CORRELACIONES")
    print(f"{'='*55}")

    corr = df[VARS_NUMERICAS].corr()

    # Top-3 correlaciónes con el TARGET (excluyendo la propia)
    corr_target = corr[TARGET].drop(TARGET).abs().sort_values(ascending=False)
    print(f"\n  Top-3 correlaciónes con '{TARGET}':")
    for var in corr_target.head(3).index:
        print(f"    {NOMBRES[var]:35s}: r = {corr[TARGET][var]:.4f}")

    # Multicolinealidad: pares con |r| > 0.9
    print(f"\n  Pares con multicolinealidad (|r| > 0.9):")
    encontrado = False
    vars_list = VARS_NUMERICAS
    for i in range(len(vars_list)):
        for j in range(i + 1, len(vars_list)):
            v1, v2 = vars_list[i], vars_list[j]
            r = corr.loc[v1, v2]
            if abs(r) > 0.9:
                print(f"    {NOMBRES[v1]} ↔ {NOMBRES[v2]}: r = {r:.4f}")
                encontrado = True
    if not encontrado:
        print("    Ninguno detectado.")

    GRIS = "#4A4A4A"
    sns.set_theme(style="white", font_scale=1.0)

    # Heatmap — triángulo inferior para evitar redundancia
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True

    fig, ax = plt.subplots(figsize=(10, 8))
    etiquetas = [NOMBRES[v] for v in VARS_NUMERICAS]
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1, vmax=1,
        xticklabels=etiquetas,
        yticklabels=etiquetas,
        ax=ax,
        linewidths=0.5,
        linecolor="white",
        annot_kws={"size": 9},
        square=True,
    )
    ax.set_title("Matriz de correlaciones de Pearson — HDR25 (2023)",
                 fontsize=12, fontweight="bold", color=GRIS, pad=15)
    plt.xticks(rotation=35, ha="right", fontsize=9, color=GRIS)
    plt.yticks(rotation=0, fontsize=9, color=GRIS)
    plt.tight_layout()
    plt.savefig("output/ej1_heatmap_correlacion.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print("\n  Guardado: output/ej1_heatmap_correlacion.png")


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("=" * 55)
    print("EJERCICIO 1 — Análisis Estadístico Descriptivo")
    print("=" * 55)

    df = cargar_datos(DATASET_PATH)
    print(f"\nDataset cargado: {df.shape[0]} paises, {df.shape[1]} columnas")

    resumen_estructural(df)
    estadisticos_descriptivos(df)
    plot_distribuciones(df)
    df = detectar_outliers(df)
    plot_boxplots(df)
    analisis_categoricas(df)
    analisis_correlaciones(df)

    print("\n" + "=" * 55)
    print("Salidas en output/")
    salidas = [
        "ej1_descriptivo.csv",
        "ej1_histogramas.png",
        "ej1_boxplots.png",
        "ej1_heatmap_correlacion.png",
        "ej1_categoricas.png",
        "ej1_outliers.txt",
    ]
    for s in salidas:
        existe = os.path.exists(f"output/{s}")
        estado = "✓" if existe else "✗ (pendiente)"
        print(f"  {estado} output/{s}")
