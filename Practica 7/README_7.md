# Práctica 7 — Clustering con K-means

**Dataset:** 11,851 partidos | 5 ligas europeas | Temporadas 2019/20 – 2025/26

---

## Objetivo

Agrupar los 11,851 partidos en clusters naturales usando K-means sin supervisión — sin usar el resultado (FTR) como guía. El objetivo es descubrir si existe una tipología natural de partidos en el espacio de cuotas y goles, y si esos tipos tienen perfiles de apuesta distintos.

A diferencia de P6 donde clasificamos con etiqueta conocida, aquí el algoritmo encuentra los grupos solo. Luego interpretamos qué significa cada grupo.

---

## Estructura del script

### Parte 1 — K-means implementado desde cero

Implementé manualmente las funciones `calculate_means()` y `calculate_nearest_centroid()`. El algoritmo:

1. Inicializa etiquetas aleatorias para cada punto
2. Calcula el centroide de cada cluster como la media de sus puntos
3. Reasigna cada punto al centroide más cercano
4. Repite hasta converger o llegar al máximo de iteraciones

Genera una imagen por iteración (`img/kmeans_iter_N.png`) para visualizar la convergencia de centroides con datos reales de fútbol.

Features para la demo: `imp_prob_H` vs `total_goals` — 2 dimensiones que permiten visualizar la convergencia de centroides directamente.

### Parte 2 — K-means con sklearn (6 features)

| Feature | Descripción |
|---|---|
| `imp_prob_H` | Dominancia del favorito local según el mercado |
| `imp_prob_A` | Dominancia del favorito visitante |
| `total_goals` | Intensidad goleadora del partido |
| `ht_goals` | Ritmo del primer tiempo |
| `odds_move_H` | Presión del mercado hacia el local antes del cierre |
| `odds_move_A` | Presión del mercado hacia el visitante |

---

## Metodología

1. **K-means manual** — implementación desde cero con 2 features graficables
2. **Normalización** — MinMaxScaler (K-means sensible a escala)
3. **Búsqueda del K óptimo** — método del codo (inercia) + silhouette score
4. **Modelo final** — K-means sklearn con el K óptimo
5. **Perfilado** — estadísticas de cada cluster: goles, cuotas, resultados, btts, over2.5
6. **Etiquetado** — nombre descriptivo por cluster basado en su perfil
7. **Composición** — distribución de clusters por liga y temporada
8. **ROI hipotético** — simulé apostar al resultado más frecuente con test binomial de significancia
8b. **K=4 forzado** — análisis de subclusters para mayor granularidad narrativa
9. **Accuracy del KNN por cluster** — nota sobre F1 bajo en clusters con clase dominante
10. **Irregularidad por cluster** — conecta con hallazgo Tigres de P5

---

## Tipos de partidos encontrados

Los clusters se etiquetan automáticamente según su perfil. Los tipos esperados son:

| Tipo | Descripción | Señal de apuesta |
|---|---|---|
| `dominio_local` | imp_prob_H=0.606, gana H el 58.8% | ROI=+0.98% — no significativo (p=0.21) |
| `dominio_visitante` | imp_prob_A=0.472, gana A el 46.5% | ROI=+5.23% — **significativo (p=0.0003)** |

Con K=4 aparecen subclusters más específicos:

| Cluster4 | Perfil | ROI |
|---|---|---|
| C3 — cerrado | avg_goals=1.61, pct_D=36.3%, pct_btts=34.9% | **+24.1% apostando al empate** |
| C2 — goleador | avg_goals=4.62, pct_over25=100%, pct_btts=88% | +16.35% apostando al local |
| C0 — visitante fav. | avg_iph=0.211, pct_A=59% | +3.1% apostando al visitante |
| C1 — local muy fav. | avg_iph=0.701, pct_H=68.3% | -0.89% — mercado bien calibrado |

---

## Hallazgos principales

### 1. K=2 es la separación natural — el mercado divide en dos tipos
Con silhouette=0.3461, los partidos se dividen en `dominio_local` (n=6,096) y `dominio_visitante` (n=5,755). La distribución es casi 50/50 en todas las ligas y temporadas — el fenómeno es universal, no específico de ninguna competición.

### 2. El mercado es eficiente para el local pero ineficiente para el visitante
Test binomial confirma:
- `dominio_visitante`: ROI=+5.23%, p=0.0003 — **altamente significativo**. El mercado subestima sistemáticamente al visitante favorito.
- `dominio_local`: ROI=+0.98%, p=0.2123 — no significativo. El mercado está bien calibrado para el favorito local.

Esta asimetría es el hallazgo central del PIA.

### 3. K=4 revela el cluster cerrado como el más explotable
El cluster cerrado (avg_goals=1.61, pct_D=36.3%) tiene ROI=+24.1% apostando al empate. El mercado paga 3.41 por el empate pero ocurre el 36.3% de las veces — la probabilidad implícita es 29.3% vs 36.3% real, una diferencia de 7 puntos. El cluster goleador (pct_over25=100%) también tiene ROI=+16.35%.

### 4. El KNN funciona mejor en dominio_local (acc=0.569) que en dominio_visitante (acc=0.468)
El F1 macro bajo en dominio_local (0.295) no indica fallo — con 58.8% de victorias locales el KNN predice casi siempre H y acierta en accuracy pero falla en recall de D y A. En dominio_visitante el F1 es más alto (0.365) porque la distribución es más equilibrada.

### 5. La irregularidad confirma el hallazgo Tigres
`dominio_visitante` tiene avg_racha=-0.509 y 48.1% de rachas negativas del local. Cuando el local es underdog, frecuentemente viene en racha negativa — eso explica por qué el visitante gana más de lo que el mercado anticipa. Conecta directamente con el hallazgo de P5 (irregularidad predice peor rendimiento, p=0.009).

### 6. La distribución por liga es casi uniforme
Todas las ligas tienen ~50% en cada cluster. La excepción menor es La Liga (53.6% dominio_local) y Serie A (50.5% dominio_visitante). Esto confirma que el fenómeno no es un artefacto de una liga específica.

---

## Conexión con el PIA

- **P2:** el mercado tiene sesgos — el edge varía por rango de cuota
- **P4:** confirmado — las cuotas difieren significativamente entre resultados (KW p<0.001)
- **P5:** el sesgo es cuantificable — R²=0.475 en el modelo de edge del mercado
- **P6:** con cuotas de cierre solas clasificamos al 52.4% de accuracy
- **P7:** K-means reveló que el mercado subestima al visitante favorito (ROI=+5.23%, p=0.0003) y sobrevalora los partidos cerrados para el empate (ROI=+24.1% en K=4)

El hallazgo del PIA: *"el mercado de apuestas europeo tiene un sesgo sistemático y cuantificable — subestima al visitante favorito y sobrevalora el local en partidos cerrados. K-means lo hizo explícito sin supervisión y el test binomial lo confirmó estadísticamente con 5,755 y 3,995 partidos respectivamente"*

---

## Gráficas generadas (`img/`)

| Archivo | Descripción |
|---|---|
| `kmeans_espacio_original.png` | Espacio de partidos sin clusters (datos reales) |
| `kmeans_iter_N.png` | Convergencia de centroides iteración a iteración |
| `kmeans_manual_resultado.png` | Resultado final del K-means manual |
| `kmeans_elbow_silhouette.png` | Método del codo e inercia vs K |
| `kmeans_perfil_clusters.png` | Barras de indicadores por cluster |
| `kmeans_pca_scatter.png` | Clusters proyectados en 2D con PCA |
| `kmeans_scatter_directo.png` | Clusters en espacio imp_prob_H vs total_goals |
| `kmeans_composicion_liga_temporada.png` | Distribución de clusters por liga y temporada |
| `kmeans_roi_por_cluster.png` | ROI hipotético por tipo de partido |
| `kmeans_knn_accuracy_por_cluster.png` | Accuracy del KNN de P6 por cluster |
| `kmeans_racha_por_cluster.png` | Estado de racha del local por cluster |

---

## Ejecución

```bash
pip install scikit-learn
cd "Practica 7"
python kmeans_clustering.py > resultados_7.txt
```
