# Práctica 10 — Reporte Final (PIA)

Script: todos los scripts de Práctica 1 a Práctica 9  
Dataset: `football_clean.csv` — 11,851 partidos, 5 ligas europeas, temporadas 2019/20 a 2025/26

---

## Video

[Football Intelligence — Reporte de Inteligencia de Negocios](https://drive.google.com/file/d/16jI0_aU9XlNuLf72b8s1j8CYsb8iPmHT/view?usp=drivesdk)

---

## Resumen ejecutivo

El análisis transforma 11,851 partidos de fútbol europeo en inteligencia de mercado para casas de apuestas y tipsters profesionales. El objetivo no fue predecir resultados individuales sino detectar dónde el mercado está mal calibrado y qué patrones se repiten con respaldo estadístico.

---

## Hallazgos principales

**1. El overround sube en todas las ligas de forma consistente**
El margen de ganancia de las casas crece año tras año en las 5 ligas analizadas. Premier League tiene el margen más bajo (4.38%) pero el crecimiento más predecible — R²=0.29, p-value prácticamente cero. Ligue 1 ya opera al 5.07%. El mercado se vuelve más caro para el apostador promedio cada temporada.

**2. Serie A: la línea de over 2.5 está desactualizada**
El fútbol italiano muestra una caída de −0.078pp por semana en el porcentaje de partidos con más de 2.5 goles (R²=0.10, p<0.001). El modelo predice una caída del 52% histórico al 42% en 16 semanas. El smart money ya corrige las cuotas de local en Serie A de forma sistemática semana tras semana — la casa no ha ajustado del todo.

**3. La casa no ha ajustado las cuotas a la forma reciente de los equipos**
Comparando el win rate rodante de las últimas 5 y 10 fechas contra la probabilidad implícita de la casa, se detectaron brechas significativas: Real Madrid +15.1pp, Barcelona +19.2pp con tendencia creciente, Arsenal +9.9pp en racha. Cuando el gap de las últimas 5 fechas supera al de las últimas 10, la racha es más fuerte ahora que antes y el mercado tiene lag.

**4. Ligue 1 post fecha FIFA: el under 2.5 tiene edge histórico**
La Ligue 1 es la única liga donde las semanas post fecha internacional muestran un efecto significativo: el over 2.5 cae −5.2pp en esas jornadas. El win rate del under sube a 50.3% y el ROI mejora 9.6pp respecto al under en semanas normales. Analizado sobre 885 partidos en ese contexto.

---

## Técnicas utilizadas

| Técnica | Aplicación |
|---------|------------|
| Regresión Lineal OLS | 17 series de tiempo semanales — r² hasta 0.29 |
| Forecasting | Validado con split 80/20 train/test — predicción a 16 semanas |
| KNN Classification | Clasificación de resultados por odds y rendimiento |
| K-Means Clustering | Segmentación de partidos por perfil de mercado |
| Kruskal-Wallis / Mann-Whitney | Validación estadística de diferencias entre ligas |
| Análisis de texto | Frecuencia de vocabulario — 59,255 tokens del dataset |

---

## Limitaciones

- R² bajo en la mayoría de los modelos — el fútbol tiene alta varianza natural
- Las cuotas de under no están disponibles directamente en el dataset
- Lesiones y rotaciones de plantilla no están consideradas
- El lag de mercado puede corregirse rápidamente
- La regresión lineal no captura cambios de régimen como nueva reglamentación (ej. propuesta Arsene Wenger)

---

## Diapositivas

Las diapositivas de la presentación están en `PIA_Football_Intelligence.pptx` dentro de esta carpeta.
