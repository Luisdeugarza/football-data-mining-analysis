# Práctica 3 — Visualización de datos

Script: `data_visualization.py`  
Dataset: `football_clean.csv` — 11,851 partidos, 5 ligas europeas, temporadas 2019/20 a 2025/26

---
Colores por liga
    "E0":Inglesa  "#3498db",
    "SP1":española "#e74c3c",
    "D1":alemana  "#f39c12",
    "I1":italiana  "#2ecc71",
    "F1":francesa  "#9b59b6",
## Qué hace el script

Lee el CSV limpio generado en la Práctica 1, reconstruye las variables derivadas necesarias y genera alrededor de 45 figuras guardadas en la carpeta `img/`. Las gráficas están organizadas en 8 bloques según el tipo de visualización.

## Cómo correrlo

```bash
cd Practica\ 3
python data_visualization.py
```

Las imágenes se guardan automáticamente en `img/`. Si la carpeta no existe la crea sola.

---

## Gráficas generadas

### Distribuciones univariadas

Histogramas de goles totales con curva KDE encima, uno por liga. Primer tiempo vs segundo tiempo superpuestos en el mismo eje. Distribución de las tres cuotas (local, empate, visitante) usando KDE con una línea por liga. Violin plot del overround para ver dónde se concentra el margen de cada casa. Histograma de la diferencia de goles por partido.

### Boxplots

Dispersión de goles totales entre ligas. Las tres cuotas en un subplot 1×3. Goles según el día de la semana. Overround por temporada para ver si el margen sube o baja con los años. Movimiento de cuota local entre apertura y cierre.

### Barras y pie charts

Pie de resultados FT y HT por liga en subplots 1×5. Barras agrupadas con todos los indicadores booleanos: over1.5, over2.5, over3.5, btts, goalless, high scoring, clean sheet. Evolución de resultados (H/D/A) por temporada. Top 15 marcadores exactos FT y top 10 al descanso en barras horizontales. Primer vs segundo tiempo por liga y temporada. Evolución de over y btts juntos en líneas.

### Series temporales

Promedio de goles por temporada con una línea por liga. Porcentaje de victorias locales por temporada. Overround promedio por temporada. BTTS por temporada. Goles por mes del año empezando en agosto. Ratio goles local/visitante para ver si la ventaja de jugar en casa se está perdiendo.

### Scatter plots

Cuota local apertura vs cierre: puntos sobre la diagonal significan que la cuota subió al cerrar. Lo mismo para la cuota del visitante. Cada partido como un punto con goles local en X y visitante en Y, coloreado por quien ganó. Calibración del mercado: burbujas por rango de cuota comparando la probabilidad implícita con el resultado real. Overround vs goles totales. Movimiento de cuota local vs resultado para analizar el comportamiento del smart money.

### Heatmaps

Matriz de transición HTR → FTR por liga en subplots 1×5. Correlaciones entre todas las variables numéricas clave. BTTS por mes × liga. Promedio de goles por día de semana × liga. Promedio de goles por liga × temporada.

### Dashboards compuestos

Un dashboard 3×3 independiente por cada liga con histograma, pie, boxplot de cuotas, evolución temporal, indicadores, matriz HTR→FTR, scatter apertura/cierre, marcadores exactos y victorias locales por mes. Dashboard comparativo de las 5 ligas en 8 métricas clave. Dashboard de evolución temporal con 6 tendencias en un 3×2.

### Gráficas adicionales

Resultados reales vs implied por rango de cuota local. Cuota de empate por mes del año. Calibración del mercado para el empate con un punto por liga y temporada. Goles apilados HT/ST por liga. Radar chart con el perfil de cada liga en 6 dimensiones. Top 15 equipos goleadores de todo el periodo. Movimiento promedio de cuotas por temporada.

---

## Variables utilizadas

Las siguientes columnas se calculan al inicio del script a partir del CSV original:

| variable | descripción |
|---|---|
| total_goals | FTHG + FTAG |
| ht_goals | HTHG + HTAG |
| second_half_goals | total_goals - ht_goals |
| goal_diff | FTHG - FTAG |
| home_win / draw / away_win | resultado en binario |
| btts | ambos equipos marcaron |
| over15 / over25 / over35 | umbrales de goles |
| goalless / high_scoring | 0 goles / 5 o más |
| clean_sheet_h / clean_sheet_a | portería a cero |
| imp_prob_H / D / A | 1 / cuota |
| overround | suma de probabilidades implícitas |
| odds_move_H / A / D | cierre - apertura |
| Season_label | etiqueta legible de temporada |

---

## Librerías requeridas

```
pandas
matplotlib
numpy
scipy--la única que no se había utilizado
```
