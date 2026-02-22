# Práctica 2 — Estadística Descriptiva

Análisis estadístico completo del dataset de fútbol europeo usando funciones de agregación y álgebra relacional. El análisis está estructurado en un ciclo por temporada → liga → comparación general.

## Objetivo

Extraer estadísticas descriptivas significativas del dataset limpio, identificar patrones por liga y temporada, y validar distribuciones reales contra distribuciones simuladas.

## Estructura

```
Practica 2/
├── img/
│   ├── er_diagram.png
│   ├── boxplot_goles_por_liga.png
│   ├── boxplot_odds_H_por_liga.png
│   ├── boxplot_mov_H_por_liga.png
│   ├── boxplot_goles_por_temporada.png
│   ├── sim_vs_real_D1.png
│   ├── sim_vs_real_E0.png
│   ├── sim_vs_real_F1.png
│   ├── sim_vs_real_I1.png
│   ├── sim_vs_real_SP1.png
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
│   ├── underdog_win_vs_lose.png
│   ├── odds_apertura_por_liga.png
│   ├── goles_local_vs_visitante.png
│   └── movimiento_mercado.png
└── descriptive_statistics.py
```

## Marco teórico

### Funciones de agregación usadas

| Función | Aplicación en el análisis |
|---------|--------------------------|
| `min` / `max` | Goles máximos, odds extremas, odds de underdogs |
| `moda` | Resultado más frecuente (FTR, HTR) por liga/temporada |
| `count` | Partidos totales por segmento, underdogs, smart money |
| `sum` | Goles acumulados, victorias totales, clean sheets |
| `mean` | Promedio de goles, odds, porcentajes de resultados |
| `var` / `std` | Variabilidad de goles y movimiento de mercado |
| `skew` | Asimetría de distribuciones de goles y odds |
| `kurt` | Kurtosis para detectar colas pesadas |

### Álgebra relacional aplicada

| Operación | Código |
|-----------|--------|
| Selección | `df[df["Season"] == temporada]` |
| Proyección | `df[["Div","Date","HomeTeam","FTHG","FTAG"]]` |
| Agrupación | `df.groupby(["Div","Season"]).agg(...)` |
| Join | `hg.merge(ag, on="equipo", how="outer")` |
| Transposición | `.unstack()` para pivotar resultados H/D/A |

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

- Estadísticas numéricas completas (mean, median, std, min, max, q25, q75, skew, kurt)
- Distribución simulada vs real de goles
- Resultados FT y HT con porcentajes
- Flags: btts, over/under, clean sheets, alta anotación, sin goles
- Remontadas (pierde HT, gana FT) y comportamiento entre tiempos
- Odds: apertura, cierre, overround, movimiento de mercado
- Underdogs (AvgA > 4) y underdogs extremos (AvgA > 8) que ganaron
- Smart money: movimientos > 0.1 hacia local, visitante y empate

### Ciclo por liga dentro de cada temporada (D1, E0, F1, I1, SP1)

Para cada combinación liga × temporada:

- Todos los indicadores anteriores desagregados
- Top 5 goleadores local y visitante
- Top 5 equipos con más clean sheets
- Listado de underdogs extremos que ganaron

### Comparación general

Al final se comparan todas las temporadas y todas las ligas en tablas resumen, incluyendo tabla de distribución simulada vs real por temporada y por liga.

## Distribuciones simuladas

Se implementan `normalize_distribution` y `create_distribution` para generar distribuciones sintéticas basadas en la media observada y compararlas con los datos reales:

```python
def normalize_distribution(dist: np.ndarray, n: int) -> np.ndarray:
    b = dist - dist.min() + 1e-6
    c = (b / b.sum()) * n
    return np.round(c)

def create_distribution(mean: float, size: int) -> np.ndarray:
    return normalize_distribution(np.random.standard_normal(size), mean * size)
```

Esto permite validar si la distribución real de goles se comporta de forma consistente con lo esperado dado el promedio observado.

## Diagrama Entidad-Relación

Generado con `matplotlib` a partir del dataset. Entidades y relaciones:

| Entidad | Atributos clave |
|---------|----------------|
| Match | Date, FTHG, FTAG, FTR, HTHG, HTAG, HTR |
| Team | HomeTeam / AwayTeam |
| League | Div (E0, SP1, D1, I1, F1) |
| Season | Season (1920 … 2526) |
| Odds | B365H/D/A, MaxH/D/A, AvgH/D/A, B365CH/D/A, MaxCH/D/A, AvgCH/D/A |

| Relación | Cardinalidad |
|----------|-------------|
| Team plays Match | N:1 |
| Match has Odds | 1:1 |
| Match belongs to League | N:1 |
| Match played in Season | N:1 |

## Hallazgos principales

| Hallazgo | Valor |
|----------|-------|
| Victoria local promedio | ~43% en todas las ligas y temporadas |
| Liga más goleadora | Bundesliga (3.15 goles/partido) |
| Liga menos goleadora | La Liga (2.55 goles/partido) |
| Underdogs extremos (AvgA > 8) | Ganan ~15% de las veces |
| Smart money hacia local | Predice correctamente ~35% |
| Overround promedio | ~1.044 (2019-2025), sube a ~1.062 en 2025/26 |
| Remontadas locales | ~5-6% de los partidos |
| Equipo más goleador | Bayern Munich (663 goles totales) |
| Mayor % victorias local | Bayern Munich y Real Madrid (~77%) |

## Imports

```
pandas
numpy
matplotlib
tabulate
scipy
```
