# Football Data Mining Analysis

Análisis de datos de fútbol europeo usando técnicas de data mining. El dataset contiene resultados y cuotas de apuestas de las 5 principales ligas europeas durante 7 temporadas.

## Dataset

Fuente: [football-data.co.uk](https://www.football-data.co.uk)

| Parámetro | Valor |
|-----------|-------|
| Ligas | Premier League (E0), La Liga (SP1), Bundesliga (D1), Serie A (I1), Ligue 1 (F1) |
| Temporadas | 2019/20 — 2025/26 |
| Filas | 11,851 partidos |
| Columnas | 29 variables |

## Estructura del repositorio

```
football-data-mining-analysis/
├── Practica 1/
│   ├── data/
│   │   ├── raw/football_raw.csv
│   │   └── clean/football_clean.csv
│   └── data_adquisition_cleaning.py
├── Practica 2/
│   ├── img/
│   └── descriptive_statistics.py
└── README.md
```

## Prácticas

### Práctica 1 — Adquisición y Limpieza de Datos

Descarga automática de CSVs por liga y temporada desde football-data.co.uk. Limpieza y estandarización del dataset.

**Funciones de agregación usadas:**
- Selección de columnas relevantes (proyección)
- Deduplicación por `(Div, Date, HomeTeam, AwayTeam)`
- Imputación con mediana para nulos residuales
- Conversión de tipos: fechas, goles a `Int64`, odds a `float`

**Decisiones de diseño:**
- Se eliminaron columnas de Pinnacle (PSH/PSD/PSA) por ~250 nulos desde julio 2025
- Se conservaron columnas de apertura y cierre para análisis de movimiento de mercado
- Las fechas se mantienen como `datetime` para permitir agrupaciones temporales

---

### Práctica 2 — Estadística Descriptiva

Análisis estadístico completo organizado por temporada → liga → comparación general.

**Funciones de agregación:**

| Función | Aplicación |
|---------|-----------|
| `min` / `max` | Goles máximos, odds extremas por periodo |
| `moda` | Resultado más frecuente por liga |
| `count` | Partidos totales por segmento |
| `sum` | Goles totales, victorias acumuladas |
| `mean` | Promedio de goles, odds, porcentajes |
| `var` / `std` | Variabilidad de goles y movimiento de mercado |
| `skew` | Asimetría de distribuciones de goles y odds |
| `kurt` | Kurtosis para detectar colas pesadas en distribuciones |

**Álgebra relacional aplicada:**

| Operación | Ejemplo en el análisis |
|-----------|----------------------|
| Selección | `df[df["Season"] == temporada]` |
| Proyección | `df[["Div", "Date", "HomeTeam", "FTHG", "FTAG"]]` |
| Agrupación | `df.groupby(["Div", "Season"]).agg(...)` |
| Join | `hg.merge(ag, on="equipo", how="outer")` |
| Unión | `pd.concat([df_liga1, df_liga2])` |
| Transposición | `.unstack()` para pivotar resultados HDA |

**Variables derivadas:**

```python
total_goals   = FTHG + FTAG
btts          = (FTHG > 0) & (FTAG > 0)
over/under    = total_goals > threshold
clean_sheet_h = FTAG == 0
imp_prob_H    = 1 / AvgH
overround     = imp_prob_H + imp_prob_D + imp_prob_A
odds_move_H   = AvgCH - AvgH   # cierre - apertura
is_underdog   = AvgA > 4
```

**Distribuciones simuladas:**

Se usa `normalize_distribution` para comparar la distribución real de goles con una distribución simulada basada en la media observada, permitiendo validar si los datos siguen un comportamiento esperado.

**Análisis incluidos:**

- Goles (local, visitante, total, medio tiempo) por temporada y liga
- Resultados FT y HT con porcentajes
- Métricas de apuestas: btts, over/under 1.5 a 4.5, clean sheets
- Remontadas y comportamiento entre tiempos
- Análisis de odds: apertura, cierre, overround, movimiento de mercado
- Underdogs (AvgA > 4) y underdogs extremos (AvgA > 8)
- Smart money: movimientos significativos de cuota (> 0.1)
- Rankings de equipos: goles, victorias, empates, clean sheets
- Correlaciones entre variables
- Distribución simulada vs real por liga y temporada

**Gráficas generadas (`img/`):**

| Archivo | Descripción |
|---------|-------------|
| `er_diagram.png` | Diagrama entidad-relación del dataset |
| `goles_por_temporada_y_liga.png` | Evolución de goles promedio |
| `over25_por_temporada_y_liga.png` | Tendencia over 2.5 |
| `btts_por_temporada_y_liga.png` | Tendencia ambos marcan |
| `pct_home_win_por_temporada_liga.png` | Ventaja local por temporada |
| `clean_sheet_por_temporada_liga.png` | Porterías a cero por temporada |
| `resultados_pie_por_liga.png` | Distribución H/D/A por liga |
| `btts_over_under_cs_por_liga.png` | Métricas de goles por liga |
| `distribucion_goles_por_liga.png` | Histogramas de goles por liga |
| `distribucion_goles_totales.png` | Histograma global de goles |
| `overround_por_liga.png` | Margen de la casa por liga |
| `underdog_win_vs_lose.png` | Odds de underdogs ganadores vs perdedores |
| `boxplot_goles_por_liga.png` | Boxplot de goles por liga |
| `boxplot_odds_H_por_liga.png` | Boxplot de odds local por liga |
| `boxplot_mov_H_por_liga.png` | Boxplot movimiento de mercado |
| `boxplot_goles_por_temporada.png` | Boxplot de goles por temporada |
| `sim_vs_real_{liga}.png` | Distribución simulada vs real (×5) |
| `odds_apertura_por_liga.png` | Scatter AvgH vs AvgA |
| `goles_local_vs_visitante.png` | Scatter FTHG vs FTAG |
| `movimiento_mercado.png` | Scatter movimiento H vs A |

## Hallazgos principales por el momento

- **Ventaja local estable**: ~43% de victorias locales en todas las temporadas
- **Bundesliga** es la liga más goleadora (3.15 goles/partido), **La Liga** la más defensiva (2.55)
- **Underdogs extremos** (odds > 8) ganan ~15% de las veces — suceden más de lo esperado
- **Smart money hacia local** predice correctamente ~35% de los casos vs ~25% al azar esperado
- El **overround** aumentó en 2025/26 (~1.062) respecto a temporadas anteriores (~1.044)
- Las **remontadas locales** representan ~5-6% de los partidos en todas las ligas


## Imports

```
pandas
numpy
matplotlib
tabulate
scipy
```
