# Práctica 8 — Forecasting con Series de Tiempo

Script: `forecasting.py`  
Dataset: `football_clean.csv` — 11,851 partidos, 5 ligas europeas, temporadas 2019/20 a 2025/26

---

## Objetivo

Aplicar regresión lineal sobre series de tiempo semanales para detectar tendencias de mercado accionables para una casa de apuestas. El análisis está orientado a identificar ineficiencias en las líneas de Pinnacle-style markets.

## Cómo correrlo

```bash
cd Practica\ 8
python forecasting.py
```

Las imágenes se guardan en `img/`.

---

## Librerías requeridas

```
pandas
numpy
matplotlib
statsmodels
scipy
tabulate
```

---

## Metodología

Agrupé los partidos por semana con `resample('W')` para construir series temporales continuas. Cada serie se modeló con **OLS** usando un índice numérico `t` como variable independiente (las fechas no son aceptadas directamente por OLS). Validé cada modelo con un split **80% train / 20% test** calculando MAE y RMSE sobre datos que el modelo nunca vio. Para la predicción futura extendí el índice `t` más allá del último punto observado.

---

## Análisis realizados (17 bloques)

| # | Análisis | Métrica principal |
|---|----------|-------------------|
| 1 | % over 2.5 por liga | tendencia semanal + pred 16w |
| 2 | % victorias local por liga | tendencia semanal + pred 16w |
| 3 | Movimiento de odds apertura→cierre | smart money por liga |
| 4 | Overround semanal por liga | margen de la casa en el tiempo |
| 5 | Victorias local vs visitante por equipo | local y visita comparados |
| 6 | Bias probabilidad implícita vs resultado real | ineficiencia de mercado |
| 7 | Calibración del bias por temporada | ¿aprende el mercado? |
| 8 | Overround por semana del año | márgenes estacionales |
| 9 | Value bet acumulado — apuesta visitante | ROI histórico por liga |
| 10 | Underdog visitante (cuota > 4.0) | subvaluación detectada |
| 11 | Efecto semanas post fecha FIFA | impacto calendario internacional |
| 12 | ROI under 2.5 en Ligue 1 post-FIFA | estrategia específica |
| 13 | Arsenal local — cuota vs win rate rodante | value actual |
| 14 | Rolling gap todos los equipos | value local y visitante |
| 15 | Tabla resumen — mejor liga por tipo | recomendación consolidada |
| 16 | Racha 5 vs 10 — value creciente | detección de lag de mercado |
| 17 | Over/under por equipo de local | señal de goles por equipo |

---

## Hallazgos principales

**Overround creciente en todas las ligas** — el margen de la casa sube de forma estadísticamente significativa en todas las ligas (p=0.000). Premier League tiene el R²=0.29 más alto, lo que indica que su margen crece de forma muy consistente y predecible.

**Serie A: caída significativa en over 2.5** — tendencia de −0.078pp por semana (R²=0.10, p<0.001). En 16 semanas la predicción baja a 41.9%. Es la única liga con señal estadística sólida en esta métrica.

**Ligue 1 post fecha FIFA** — el % over 2.5 cae −5.2pp en semanas post-fecha internacional, el único efecto de calendario significativo detectado. El ROI del under en esas semanas es 9.6pp mejor que el under normal.

**Barcelona local: value creciente** — gap de +7.8pp en últimas 10 fechas y +19.2pp en últimas 5. La tendencia acelerando indica que la casa no ha ajustado las cuotas a la racha actual.

**Real Madrid local** — gap de +15.1pp, ganando 90% de sus últimos 10 partidos de local con imp_prob de solo 75.9%.

**Bias estructural** — todas las ligas sobreestiman al local en 1.5−4.4pp de forma consistente, pero sin tendencia significativa. La casa sabe que sobreestima pero no lo corrige porque es parte de su estrategia de pricing.

---

## Gráficas generadas (17 archivos)

```
img/
├── forecasting_over25_por_liga.png
├── forecasting_homewin_por_liga.png
├── forecasting_odds_move.png
├── forecasting_overround.png
├── forecasting_equipos.png
├── forecasting_bias.png
├── forecasting_bias_temporada.png
├── forecasting_overround_semana.png
├── forecasting_valuebet_visitante.png
├── forecasting_underdog.png
├── forecasting_post_fifa.png
├── forecasting_under_ligue1_postfifa.png
├── forecasting_arsenal_value.png
├── forecasting_rolling_gap_equipos.png
├── forecasting_resumen_final.png
├── forecasting_racha_value.png
└── forecasting_over_por_equipo.png
```
