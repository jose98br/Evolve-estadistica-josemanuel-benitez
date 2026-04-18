# Práctica Final — Estadística para Data Science

Análisis estadístico completo sobre el dataset **HDR25 Composite Indices** del PNUD (Programa de Naciones Unidas para el Desarrollo, 2025). El dataset recoge indicadores socioeconómicos de 206 países para el año 2023.

**Autor:** Jose Manuel Benítez  
**Asignatura:** Estadística  
**Entrega:** Classroom 

---

## Estructura del repositorio

```
├── ejercicio1_descriptivo.py
├── ejercicio2_inferencia.py
├── ejercicio3_regresion_multiple.py
├── ejercicio4_series_temporales.py
├── Respuestas.md
├── data/
│   └── HDR25_Composite_indices_complete_time_series.csv
└── output/
    ├── ej1_*.png / .csv / .txt
    ├── ej2_*.png / .txt
    ├── ej3_*.png / .txt
    └── ej4_*.png / .txt
```

---

## Dataset

**HDR25 Composite Indices — Human Development Report 2025**  
Fuente: [hdr.undp.org](https://hdr.undp.org/data-center/documentation-and-downloads)

- 206 países, 1112 columnas, datos del año 2023
- Variable objetivo: `le_2023` (esperanza de vida en años)
- Variables usadas: IDH, años de escolaridad, INB per cápita, desigualdad de género, emisiones de CO2, región geográfica y nivel de desarrollo humano

---

## Ejercicios

### Ejercicio 1 — Análisis descriptivo
Exploración completa del dataset: estadísticos descriptivos (media, mediana, IQR, skewness, curtosis), detección de outliers por método IQR, análisis de variables categóricas y matriz de correlaciones de Pearson.

### Ejercicio 2 — Regresión con Scikit-Learn
Modelo de regresión lineal sobre el dataset real. Incluye preprocesamiento (codificación, escalado, split 80/20), evaluación con MAE, RMSE y R², y análisis de residuos.

### Ejercicio 3 — Regresión lineal desde cero con NumPy
Implementación manual de la solución OLS (β = (XᵀX)⁻¹ Xᵀy) sin usar scikit-learn. Se valida sobre datos sintéticos con coeficientes conocidos.

### Ejercicio 4 — Series temporales
Análisis de una serie temporal sintética (2018–2023). Descomposición aditiva en tendencia, estacionalidad y residuo. Análisis del residuo con tests de normalidad (Jarque-Bera) y estacionariedad (ADF).

---

## Cómo ejecutar

Requiere Python 3.12+ y las siguientes librerías:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn statsmodels scipy
```

Cada script se ejecuta de forma independiente desde la raíz del proyecto:

```bash
python ejercicio1_descriptivo.py
python ejercicio2_inferencia.py
python ejercicio3_regresion_multiple.py
python ejercicio4_series_temporales.py
```

Los resultados se guardan automáticamente en la carpeta `output/`.
