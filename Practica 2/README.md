# Práctica 2 — Estadística Descriptiva

Análisis estadístico completo del dataset de fútbol europeo usando funciones de agregación y álgebra relacional. El análisis está estructurado en un ciclo por temporada → liga → comparación general.

## Objetivo

Extraer estadísticas descriptivas significativas del dataset limpio e identificar patrones por liga y temporada usando funciones de agregación sobre los datos reales.

## Estructura

```
Practica 2/
├── img/
│   ├── er_diagram.png
│   ├── odds_apertura_por_liga.png
│   ├── goles_local_vs_visitante.png
│   ├── odds_apertura_vs_cierre_H.png
│   ├── odds_apertura_vs_cierre_A.png
│   ├── implied_prob_H_vs_A.png
│   ├── movimiento_mercado.png
│   ├── goles_por_temporada_y_liga.png
│   ├── over25_por_temporada_y_liga.png
│   ├── btts_por_temporada_y_liga.png
│   ├── pct_home_win_por_temporada_liga.png
│   ├── clean_sheet_por_temporada_liga.png
│   ├── resultados_pie_por_liga.png
│   ├── btts_over_under_cs_por_liga.png
│   ├── distribucion_goles_por_liga.png
│   ├── distribucion_goles_totales.png
│   ├── overround_por_liga.png
│   └── underdog_win_vs_lose.png
└── descriptive_statistics.py
```

## Marco teórico

### Funciones de agregación usadas

| Función | Aplicación en el análisis |
|---------|--------------------------|
| `min` / `max` | Goles máximos, odds extremas por periodo |
| `moda` | Resultado más frecuente (FTR, HTR) por liga/temporada |
| `count` | Partidos totales por segmento |
| `sum` | Goles acumulados, victorias totales, clean sheets |
| `mean` | Promedio de goles, odds, porcentajes de resultados |
| `var` / `std` | Variabilidad de goles y movimiento de mercado |
| `skew` | Asimetría de distribuciones de goles y odds |
| `kurt` | Kurtosis para detectar colas pesadas |

### Álgebra relacional aplicada

| Operación | Ejemplo en el análisis |
|-----------|----------------------|
| Selección | `df[df["Season"] == temporada]` |
| Proyección | `df[["Div","Date","HomeTeam","FTHG","FTAG"]]` |
| Agrupación | `df.groupby(["Div","Season"]).agg(...)` |
| Join | `hg.merge(ag, on="equipo", how="outer")` |
| Transposición | `.unstack()` para pivotar resultados H/D/A |

### Patrón Map-Reduce

```python
# MAP: transformar cada elemento
df["imp_prob_H"]  = df["AvgH"].map(lambda x: 1/x)
df["total_goals"] = df["FTHG"] + df["FTAG"]

# REDUCE: agregar por grupo
df.groupby("Div")["total_goals"].agg(["sum","mean","count","std"])
df.groupby(["Div","Season"])["home_win"].mean()
```

## Variables derivadas

```python
total_goals    = FTHG + FTAG
ht_goals       = HTHG + HTAG
goal_diff      = FTHG - FTAG
btts           = (FTHG > 0) & (FTAG > 0)
over15/25/35/45 = total_goals > threshold
clean_sheet_h  = FTAG == 0
clean_sheet_a  = FTHG == 0
high_scoring   = total_goals >= 5
goalless       = total_goals == 0
imp_prob_H     = 1 / AvgH
overround      = imp_prob_H + imp_prob_D + imp_prob_A
odds_move_H    = AvgCH - AvgH
is_underdog    = AvgA > 4
```

## Estructura del análisis

### Ciclo por temporada (2019/20 → 2025/26)

Para cada temporada se calculan:

- Estadísticas numéricas completas: mean, median, std, min, max, q25, q75, skew, kurt
- Resultados FT y HT con porcentajes
- Flags: btts, over/under 1.5–4.5, clean sheets, alta anotación, sin goles
- Remontadas (pierde HT, gana FT) y comportamiento entre tiempos
- Odds: apertura, cierre, overround, movimiento de mercado
- Underdogs (AvgA > 4) y underdogs extremos (AvgA > 8) que ganaron
- Smart money: movimientos > 0.1 hacia local, visitante y empate
- Listado de partidos de alta anotación (≥ 5 goles)

### Ciclo por liga dentro de cada temporada (D1, E0, F1, I1, SP1)

Para cada combinación liga × temporada:

- Todos los indicadores anteriores desagregados
- Top 5 goleadores local y visitante
- Top 5 equipos con más clean sheets
- Listado de underdogs extremos que ganaron

### Comparación general al final

- Tablas resumen de todas las temporadas
- Tablas resumen de todas las ligas (todo el periodo)
- Skewness y kurtosis de goles por liga
- Rankings de equipos: goles, victorias, empates, clean sheets
- Matriz de correlación entre variables
- Estadísticas por mes del año

## Diagrama Entidad-Relación

Generado con `matplotlib`. Entidades y relaciones del dataset:

| Entidad | Atributos clave |
|---------|----------------|
| Match | Date, FTHG, FTAG, FTR, HTHG, HTAG, HTR |
| Team | HomeTeam / AwayTeam |
| League | Div (E0, SP1, D1, I1, F1) |
| Season | Season (1920 … 2526) |
| Odds | B365H/D/A, MaxH/D/A, AvgH/D/A, cierres |

| Relación | Cardinalidad |
|----------|-------------|
| Team plays Match | N:1 |
| Match has Odds | 1:1 |
| Match belongs to League | N:1 |
| Match played in Season | N:1 |

## Gráficas generadas

| Archivo | Descripción |
|---------|-------------|
| `er_diagram.png` | Diagrama entidad-relación del dataset |
| `goles_por_temporada_y_liga.png` | Evolución de goles promedio por temporada |
| `over25_por_temporada_y_liga.png` | Tendencia over 2.5 por temporada |
| `btts_por_temporada_y_liga.png` | Tendencia ambos marcan por temporada |
| `pct_home_win_por_temporada_liga.png` | Ventaja local por temporada |
| `clean_sheet_por_temporada_liga.png` | Porterías a cero por temporada |
| `resultados_pie_por_liga.png` | Distribución H/D/A por liga |
| `btts_over_under_cs_por_liga.png` | Métricas de goles agrupadas por liga |
| `distribucion_goles_por_liga.png` | Histogramas de goles por liga |
| `distribucion_goles_totales.png` | Histograma global de goles |
| `overround_por_liga.png` | Margen de la casa por liga |
| `underdog_win_vs_lose.png` | Odds de underdogs ganadores vs perdedores |
| `odds_apertura_por_liga.png` | Scatter AvgH vs AvgA por liga |
| `goles_local_vs_visitante.png` | Scatter FTHG vs FTAG por liga |
| `movimiento_mercado.png` | Scatter movimiento de mercado H vs A |
| `implied_prob_H_vs_A.png` | Scatter probabilidades implícitas H vs A |
| `odds_apertura_vs_cierre_H.png` | Scatter odds apertura vs cierre local |
| `odds_apertura_vs_cierre_A.png` | Scatter odds apertura vs cierre visitante |

## Hallazgos principales

| Hallazgo | Valor |
|----------|-------|
| Victoria local promedio | ~43% en todas las ligas y temporadas |
| Liga más goleadora | Bundesliga (3.15 goles/partido) |
| Liga menos goleadora | La Liga (2.55 goles/partido) |
| Underdogs extremos (AvgA > 8) | Ganan ~15% de las veces |
| Smart money hacia local | Predice correctamente ~35% |
| Overround promedio | ~1.044 (2019–2025), sube a ~1.062 en 2025/26 |
| Remontadas locales | ~5–6% de los partidos |
| Equipo más goleador | Bayern Munich (663 goles totales) |
| Mayor % victorias local | Bayern Munich y Real Madrid (~77%) |

## Imports

```
pandas
numpy
matplotlib
tabulate
```
