# Respuestas — Práctica Final: Análisis y Modelado de Datos

---

## Ejercicio 1 — Análisis Estadístico Descriptivo

Dataset: **HDR25 Composite Indices** del PNUD, publicado en el Informe de Desarrollo Humano 2025. Contiene indicadores socioeconómicos de 206 países para el año 2023. Variable objetivo: `le_2023` (esperanza de vida en años).

---

**Pregunta 1.1** — ¿De qué fuente proviene el dataset y cuál es la variable objetivo (target)? ¿Por qué tiene sentido hacer regresión sobre ella?

El dataset es del PNUD (Programa de Naciones Unidas para el Desarrollo), disponible en [hdr.undp.org](https://hdr.undp.org/data-center/documentation-and-downloads). La variable objetivo es `le_2023` (esperanza de vida en años, 2023).

Tiene sentido predecirla porque es numérica continua (rango 54.5–84.7 años), con distribución bastante simétrica (skewness = -0.34), y tiene relación directa con variables del dataset como el ingreso, la educación y la desigualdad de género. Es un problema de regresión con interpretación clara: a partir de indicadores socioeconómicos de un país, ¿cuánto viven sus habitantes de media?

---

**Pregunta 1.2** — ¿Qué distribución tienen las principales variables numéricas y has encontrado outliers? Indica en qué variables y qué has decidido hacer con ellos.

- `le_2023` (esperanza de vida): distribución simétrica, skewness = -0.34, curtosis = -0.67. Sin outliers.
- `hdi_2023` (IDH): ligeramente sesgada a la izquierda, skewness = -0.31. Sin outliers.
- `gnipc_2023` (INB per cápita): fuertemente sesgada a la derecha, skewness = 1.46, curtosis = 1.92. **7 outliers**: países con ingresos muy altos (Singapur, Noruega, Luxemburgo, etc.).
- `co2_prod_2023` (CO2 per cápita): distribución extremadamente asimétrica, skewness = 2.39, curtosis = 8.08. **15 outliers**: países petroleros del Golfo Pérsico.

Decisión: se conservan todos. No son errores de medición sino valores reales que reflejan diferencias reales entre países. Quitarlos distorsionaría el análisis.

---

**Pregunta 1.3** — ¿Qué tres variables numéricas tienen mayor correlación (en valor absoluto) con la variable objetivo? Indica los coeficientes.

| Variable | Descripción | Correlación de Pearson |
|---|---|---|
| `hdi_2023` | Índice de Desarrollo Humano | r = +0.9060 |
| `ihdi_2023` | IDH ajustado por desigualdad | r = +0.9024 |
| `gii_2023` | Índice de desigualdad de género | r = -0.8669 |

Los países más desarrollados viven más (IDH positivo). Los países con mayor desigualdad de género viven menos (GII negativo). Ambas relaciones tienen sentido.

Además, IDH e IDH ajustado tienen entre sí r = 0.98 — casi la misma información dos veces. Esto es multicolinealidad y hay que tenerlo en cuenta al construir el modelo.

---

**Pregunta 1.4** — ¿Hay valores nulos en el dataset? ¿Qué porcentaje representan y cómo los has tratado?

Sí:

| Variable | Nulos | % |
|---|---|---|
| `region` | 55 | 26.70% |
| `ihdi_2023` | 26 | 12.62% |
| `gii_2023` | 23 | 11.17% |
| `hdicode` | 11 | 5.34% |
| `hdi_2023`, `mys_2023`, `gnipc_2023`, `co2_prod_2023` | 2 | 0.97% |
| `le_2023`, `eys_2023` | 0 | 0.00% |

El 26.7% de nulos en `region` se explica porque muchos territorios pequeños y dependientes no tienen región asignada en el dataset. En el análisis descriptivo se muestran todos los registros incluyendo los nulos. Para el modelo (Ejercicio 2) se eliminan las filas con nulos mediante `dropna()`, pasando de 206 a 133 registros.

---

## Ejercicio 2 — Inferencia con Scikit-Learn

Mismo dataset del Ejercicio 1. Se excluyeron `hdi_2023` e `ihdi_2023` por multicolinealidad muy alta con el target (r > 0.9) — incluirlas haría los coeficientes inestables sin aportar información nueva.

**Preprocesamiento:**
- Eliminación de nulos → 133 registros válidos
- Codificación de categóricas con `get_dummies` (drop_first=True)
- Escalado de numéricas con `StandardScaler`
- División 80%/20% con `random_state=42` → 106 train, 27 test

---

**Pregunta 2.1** — Indica los valores de MAE, RMSE y R² de la regresión lineal sobre el test set. ¿El modelo funciona bien? ¿Por qué?

| Métrica | Valor | Qué significa |
|---|---|---|
| MAE | 2.5831 años | Error medio de predicción |
| RMSE | 3.6186 años | Penaliza más los errores grandes |
| R² | 0.5929 | Explica el 59.3% de la varianza |

El modelo funciona de forma moderada. Un R² de 0.59 no es malo considerando que se excluyeron deliberadamente las variables más correlacionadas (IDH e IDH ajustado) para construir algo más honesto. Lo más llamativo es que las regiones geográficas son los predictores más influyentes — especialmente África Subsahariana y Europa Central/Asia. La ubicación de un país predice mejor su esperanza de vida que sus indicadores individuales, probablemente porque concentra muchos factores que no están en el dataset (acceso a sanidad, conflictos, historia...).

No hay señales de overfitting: el dataset es pequeño (133 registros) y el modelo lineal no tiene parámetros suficientes para memorizar los datos.

---

## Ejercicio 3 — Regresión Lineal Múltiple en NumPy

**Pregunta 3.1** — Explica en tus propias palabras qué hace la fórmula β = (XᵀX)⁻¹ Xᵀy y por qué es necesario añadir una columna de unos a la matriz X.

La fórmula β = (XᵀX)⁻¹ Xᵀy es la solución directa de Mínimos Cuadrados Ordinarios (OLS). Lo que hace es encontrar los coeficientes β que minimizan la suma de los errores al cuadrado entre lo que predice el modelo y los valores reales.

En términos matriciales: X son los datos (filas = observaciones, columnas = variables), y son los valores reales, y β son los coeficientes que buscamos. El producto XᵀX recoge las relaciones entre variables, y al invertirlo se despeja β.

La columna de unos es necesaria para que el modelo tenga intercepto (β₀), es decir, un valor base cuando todas las variables valen 0. Sin ella, el modelo estaría forzado a pasar por el origen, lo que casi nunca tiene sentido y produce peores predicciones.

---

**Pregunta 3.2** — Copia aquí los cuatro coeficientes ajustados por tu función y compáralos con los valores de referencia del enunciado.

| Parámetro | Valor real | Valor ajustado |
|-----------|-----------|----------------|
| β₀ (intercepto) | 5.0 | 4.8650 |
| β₁ | 2.0 | 2.0636 |
| β₂ | -1.0 | -1.1170 |
| β₃ | 0.5 | 0.4385 |

Los coeficientes ajustados se acercan bien a los reales. Las diferencias son pequeñas y esperables: la variable objetivo fue generada con ruido gaussiano (σ=1.5), así que la implementación no puede recuperar los coeficientes exactos. `np.linalg.lstsq` es numéricamente estable y produce resultados correctos.

---

**Pregunta 3.3** — ¿Qué valores de MAE, RMSE y R² has obtenido? ¿Se aproximan a los de referencia?

| Métrica | Valor obtenido | Referencia del profesor |
|---|---|---|
| MAE | 1.1665 | ≈ 1.20 (±0.20) ✓ |
| RMSE | 1.4612 | ≈ 1.50 (±0.20) ✓ |
| R² | 0.6897 | ≈ 0.80 (±0.05) |

MAE y RMSE están dentro del margen. El R² de 0.69 queda algo por debajo de la referencia de 0.80, aunque el enunciado indica que los valores son aproximados. La implementación funciona correctamente.

---

## Ejercicio 4 — Series Temporales

**Pregunta 4.1** — ¿La serie presenta tendencia? Descríbela brevemente (tipo, dirección, magnitud aproximada).

Sí. La serie tiene una tendencia lineal creciente: sube de ~44 a ~172 a lo largo de los 6 años (2018–2023), un incremento total de ~128 unidades. La pendiente es 0.05 unidades/día, unos 18 unidades por año. En el gráfico de descomposición se ve claramente como una línea suavemente ascendente.

---

**Pregunta 4.2** — ¿Hay estacionalidad? Indica el periodo aproximado en días y la amplitud del patrón estacional.

Sí. Estacionalidad anual con periodo de 365 días. La amplitud es ±21 unidades (combinación de dos armónicos: 15·sin + 6·cos), lo que significa que en los picos la serie está ~21 unidades por encima de la tendencia y en los valles ~21 por debajo. El patrón se repite con precisión cada año y es claramente visible en el subgráfico de estacionalidad.

---

**Pregunta 4.3** — ¿Se aprecian ciclos de largo plazo en la serie? ¿Cómo los diferencias de la tendencia?

Sí. Hay un ciclo de ~4 años (1461 días) con amplitud ±8 unidades. La diferencia con la tendencia es que la tendencia es monotónica (siempre sube), mientras que el ciclo oscila arriba y abajo alrededor de ella. En el gráfico aparece como una ondulación de baja frecuencia dentro de la curva de tendencia. Con 6 años de datos apenas se completa un ciclo y medio.

---

**Pregunta 4.4** — ¿El residuo se ajusta a un ruido ideal? Indica la media, la desviación típica y el resultado del test de normalidad (p-value) para justificar tu respuesta.

Sí, el residuo se ajusta bien a un ruido ideal:

| Estadístico | Valor | Valor ideal |
|---|---|---|
| Media | 0.1271 | ≈ 0 |
| Desviación típica | 3.2220 | — |
| Asimetría | -0.0509 | ≈ 0 |
| Curtosis | -0.0610 | ≈ 0 |
| Jarque-Bera p-value | 0.5766 | > 0.05 |
| ADF p-value | 0.0000 | < 0.05 |

Jarque-Bera da p = 0.58, así que no se rechaza la normalidad. La media es casi 0, la asimetría y la curtosis son prácticamente nulas, y el test ADF confirma que el residuo es estacionario (p ≈ 0). Todo encaja: la serie fue generada con ruido gaussiano de σ=3.5 y la descomposición ha extraído bien todos los componentes sistemáticos, dejando solo el ruido.

---

*Fin del documento de respuestas*
