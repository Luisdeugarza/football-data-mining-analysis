# Práctica 5 — Modelos Lineales y Correlación

Script: `linear_regression.py`  
Resultados: `resultados_lineal_regression.txt`  
Dataset: `football_clean.csv` — 11,851 partidos, 5 ligas europeas, temporadas 2019/20 a 2025/26

---

## Objetivo

Construir modelos de regresión lineal simple y múltiple para predecir resultados de partidos a partir de variables de odds y rendimiento histórico. Evaluar el poder predictivo de las cuotas de mercado.

## Cómo correrlo

```bash
cd Practica\ 5
python linear_regression.py
```

Las imágenes se guardan en `img/` y el output completo en `resultados_lineal_regression.txt`.

---

## Librerías requeridas

```
pandas
numpy
matplotlib
statsmodels
scipy
```

---

## Modelos entrenados

### Regresiones simples

| Modelo | R² | Significativo |
|--------|----|---------------|
| `ht_goals → total_goals` | 0.489 | ✓ |
| `imp_prob_H → home_win` | 0.096 | ✓ |
| `AvgCH (cierre) → home_win` | 0.096 | ✓ |
| `AvgH (apertura) → home_win` | 0.095 | ✓ |
| `odds_move_H → home_win` | 0.005 | ✓ |

### Regresiones múltiples

| Modelo | R² |
|--------|----|
| `AvgH + AvgA + AvgD + overround → total_goals` | 0.054 |
| `imp_prob_H + imp_prob_A + overround → total_goals` | 0.043 |
| `irregularidad + std_goles + racha_previa → home_win` | 0.021 |
| `odds_move_H + odds_move_A → home_win` | ~0.005 |

---

## Hallazgos principales

**El mejor predictor de goles totales es el primer tiempo** — R²=0.489. Un partido con 2 goles al descanso termina con ~3.6 goles en promedio. Es la relación más fuerte de todo el análisis.

**Las cuotas predicen la victoria local con R²~0.095** — apertura y cierre dan resultados prácticamente idénticos (0.095 vs 0.096). El movimiento de cuota entre apertura y cierre agrega muy poco poder predictivo extra.

**El modelo de racha previa × categorías tiene R²=0.919** — pero es un artefacto estadístico por el agrupamiento. No implica que se pueda predecir un partido individual con esa precisión.

**La irregularidad del equipo local importa** — std de goles en últimos 5 partidos tiene coeficiente positivo y significativo (p<0.001). Equipos inconsistentes tienden a sorprender más en casa.

**Multicolinealidad en modelos con cuotas** — VIF > 10 entre `imp_prob_H` e `imp_prob_A`. Se reporta en el análisis como limitación del modelo.

---

## Variables derivadas utilizadas

| Variable | Descripción |
|----------|-------------|
| `imp_prob_H/D/A` | 1 / cuota promedio |
| `overround` | suma de probabilidades implícitas |
| `odds_move_H/A/D` | cuota cierre − cuota apertura |
| `racha_previa` | victorias locales en últimos 5 partidos |
| `std_goles_5` | desviación estándar de goles en últimos 5 |
| `irregularidad` | métrica compuesta de consistencia |

---

## Gráficas generadas

- Scatter con línea de regresión y banda de confianza por cada modelo simple
- Comparativa apertura vs cierre como predictor
- R² por liga en barras
- Diagnóstico de residuales del mejor modelo múltiple
- home_win ~ AvgH por liga en subplots 1×5
