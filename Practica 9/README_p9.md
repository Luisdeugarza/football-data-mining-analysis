# Práctica 9 — Análisis de Texto

Script: `text_analysis.py`  
Dataset: `football_clean.csv` — 11,851 partidos, 5 ligas europeas, temporadas 2019/20 a 2025/26

---

## Objetivo

Construir wordclouds y análisis de frecuencia a partir de las columnas categóricas del dataset — equipos, ligas, resultados y temporadas.

## Cómo correrlo

```bash
cd Practica\ 9
pip install wordcloud
python text_analysis.py
```

---

## Librerías requeridas

```
pandas
matplotlib
collections
wordcloud        ← requiere instalación: pip install wordcloud
tabulate
```

---

## Corpus construido

El texto se generó concatenando columnas categóricas del dataset por partido: nombre del equipo local, nombre del equipo visitante, nombre de la liga, resultado (HomeWin / Draw / AwayWin) y temporada. Para la nube narrativa construí una frase por partido del tipo `"Arsenal victoria Arsenal gana gana gana PremierLeague"` según el resultado y los goles.

| Corpus | Tokens totales | Tokens únicos |
|--------|---------------|---------------|
| Global | 59,255 | 157 |
| Narrativo | 105,549 | 159 |

---

## Hallazgos

- **HomeWin** es el token más frecuente (5,102) — el local gana más que visitante o empate en todas las ligas combinadas
- **Serie A** tiene 31 equipos únicos vs 27-28 del resto — mayor rotación por ascensos y descensos
- Todos los equipos permanentes de cada liga aparecen con ~5% del vocabulario de su liga — distribución uniforme esperada en formato todos-contra-todos
- La Bundesliga tiene el menor número de partidos (2,032) por tener 18 equipos vs 20 de Premier y La Liga

---

## Gráficas generadas

```
img/
├── wordcloud_global.png
├── wordcloud_equipos.png
├── wordcloud_top20_barras.png
├── wordcloud_por_liga.png
├── wordcloud_por_temporada.png
├── wordcloud_narrativo.png
├── wordcloud_freq_equipos_liga.png
└── wordcloud_equipos_freq.png
```
