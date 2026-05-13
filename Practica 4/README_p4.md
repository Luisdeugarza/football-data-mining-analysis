# Práctica 4 — Pruebas Estadísticas

Script: `statistic_test.py`  
Dataset: `football_clean.csv` — 11,851 partidos, 5 ligas europeas, temporadas 2019/20 a 2025/26

---

## Objetivo

Demostrar que los grupos del dataset (ligas, temporadas, tipo de partido) son estadísticamente distintos entre sí usando pruebas no paramétricas y paramétricas como contraste.

## Cómo correrlo

```bash
cd Practica\ 4
python statistic_test.py
```

Las imágenes se guardan en `img/`.

---

## Librerías requeridas

```
pandas
numpy
matplotlib
scipy
itertools
```

---

## Metodología

Primero validé los supuestos de normalidad con Shapiro-Wilk (n ≤ 5,000) y D'Agostino-Pearson (n > 5,000). Ningún grupo pasó la prueba de normalidad (p << 0.05 en todos los casos), y el test de Levene confirmó varianzas desiguales entre ligas. Por eso usé **Kruskal-Wallis** como prueba principal y ANOVA solo como contraste paramétrico. Para comparaciones uno a uno apliqué **Mann-Whitney con corrección de Bonferroni**.

---

## Pruebas ejecutadas

| Prueba | Variable | Resultado |
|--------|----------|-----------|
| Kruskal-Wallis | goles totales × liga | ✓ significativo |
| Kruskal-Wallis | overround × liga | ✓ significativo |
| Kruskal-Wallis | BTTS × liga | ✓ significativo |
| Kruskal-Wallis | goles totales × temporada | ✓ significativo |
| Kruskal-Wallis | victoria local × temporada | ✓ significativo |
| Wilcoxon pareado | FTHG > FTAG (ventaja local) | ✓ significativo |
| Wilcoxon pareado | 2do tiempo > 1er tiempo | ✓ significativo |
| Mann-Whitney | COVID vs resto en goles | ✓ significativo |
| Mann-Whitney | COVID vs resto en %local | ✓ significativo |
| Kruskal-Wallis | overround × liga | ✓ significativo |
| Kruskal-Wallis | smart money × resultado | ✓ significativo |

---

## Hallazgos principales

1. Las 5 ligas son estadísticamente distintas en goles, overround y BTTS — Kruskal-Wallis p << 0.001 en todas las variables
2. Los goles promedio varían entre temporadas, con diferencias particulares en la temporada COVID 2019/20
3. La ventaja de local es estadísticamente real — FTHG > FTAG con Wilcoxon pareado p << 0.001
4. El segundo tiempo produce más goles que el primero — Wilcoxon pareado p << 0.001
5. El movimiento de cuota tiene valor predictivo estadístico — cuando la cuota local baja antes del cierre, el local gana más seguido
6. El overround no es igual entre ligas — cada mercado tiene margen distinto para la casa

---

## Gráficas generadas

- p-values Kruskal-Wallis por variable × liga en barras `-log₁₀`
- Boxplot de goles por liga con resultado del test anotado
- Matriz de p-values pairwise Mann-Whitney entre ligas
- Boxplot de overround por liga
- Distribución de goles primer vs segundo tiempo
- Resumen de todas las pruebas en una sola figura
