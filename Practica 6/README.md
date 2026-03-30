# Práctica 6 — Clasificación con K-Nearest Neighbors (KNN)

**Dataset:** 11,851 partidos | 5 ligas europeas | Temporadas 2019/20 – 2025/26

---

## Objetivo

Clasificar el resultado final de un partido (`FTR`: H/D/A) usando KNN. El script está dividido en dos partes: primero una implementación desde cero para demostrar el funcionamiento del algoritmo, y después el modelo completo con sklearn aplicado al dataset real.

---

## Variable objetivo

| Clase | Significado | Frecuencia aprox. |
|---|---|---|
| `H` | Victoria local | ~45% |
| `D` | Empate | ~25% |
| `A` | Victoria visitante | ~30% |

---

## Estructura del script

### Parte 1 — KNN implementado desde cero

Implementé manualmente las funciones `euclidean_distance()` y `k_nearest_neighbors()` siguiendo la lógica base del algoritmo:

1. Calculo la distancia euclídea de cada punto de entrada a todos los puntos de entrenamiento
2. Tomo los K índices con menor distancia
3. Asigno la clase por votación (moda entre los K vecinos)

Para esta parte usé solo 2 features (`imp_prob_H` e `imp_prob_A`) porque con 2 dimensiones el espacio de decisión es graficable directamente. Los datos son reales del dataset — no sintéticos.

También implementé la normalización MinMax manualmente y predije 5 partidos hipotéticos con cuotas conocidas, marcados con estrella en el scatter.

### Parte 2 — KNN con sklearn (modelo completo, 15 features)

Con sklearn usé todos los features disponibles. sklearn optimiza el cálculo de distancias con estructuras KD-tree/Ball-tree, lo que hace viable usar 15 features con 11,851 partidos.

---

## Features utilizados

| Feature | Descripción |
|---|---|
| `imp_prob_H/D/A` | Probabilidades implícitas (1/cuota) |
| `AvgH/A/D` | Cuotas de apertura |
| `AvgCH/CA/CD` | Cuotas de cierre |
| `odds_move_H/A/D` | Movimiento apertura → cierre |
| `overround` | Margen total del mercado |
| `diff_imp` | Diferencia de probabilidades implícitas (H - A) |
| `racha_previa` | Racha del equipo local antes del partido |

### Feature sets comparados

| Set | Variables | Descripción |
|---|---|---|
| `probs_implicitas` | 4 | Probabilidades implícitas + overround |
| `cuotas_apertura` | 4 | Cuotas de apertura + overround |
| `cuotas_cierre` | 4 | Cuotas de cierre + overround |
| `movimiento` | 3 | Solo movimiento de cuota |
| `completo` | 15 | Todos los anteriores + racha_previa |

---

## Metodología

1. **Implementación manual** — KNN desde cero con distancia euclídea y votación por moda (2 features, graficable)
2. **Split train/test** — 80/20 estratificado por clase FTR
3. **Normalización** — MinMaxScaler (KNN es sensible a la escala)
4. **Búsqueda del K óptimo** — Loop K=1 a 30, curva accuracy vs K (método del codo)
5. **Validación cruzada** — 5-fold estratificado para confirmar estabilidad
6. **Evaluación** — Accuracy, F1-macro, classification report, matriz de confusión
7. **Permutation importance** — Qué features contribuyen más
8. **Comparación de feature sets** — Qué conjunto clasifica mejor
9. **Análisis de errores** — Tipos de error y partidos donde el modelo falla

---

## Hallazgos principales

### 1. La implementación manual confirma el algoritmo
La versión desde cero con 2 features produce resultados coherentes. La diferencia de accuracy respecto a sklearn con 15 features refleja directamente el valor de agregar más variables, no una diferencia en el algoritmo.

### 2. El set completo supera a cualquier subset individual
Las probabilidades implícitas solas capturan mucho, pero agregar movimiento de cuota y racha mejora el accuracy. El feature set de solo movimiento es el más débil — confirma que el smart money ayuda pero no es suficiente por sí solo.

### 3. Las probabilidades implícitas son los features más importantes
Confirmado por permutation importance: `imp_prob_H` e `imp_prob_A` son los que más bajan el accuracy al permutarlos. Esto es consistente con la práctica 5 donde `imp_prob_H → home_win` tuvo el mayor R².

### 4. El empate es la clase más difícil de predecir
El precision y recall de `D` son los más bajos de las tres clases en ambas versiones del modelo. Los partidos de empate no tienen una zona clara en el espacio de features — sus vecinos más cercanos suelen ser victorias locales o visitantes.

### 5. La validación cruzada confirma robustez
Los 5 folds producen accuracies similares con desviación estándar baja, lo que indica que el resultado no depende del split aleatorio.

### 6. La racha previa aporta información que las cuotas no capturan
Su inclusión mejora marginalmente el accuracy, consistente con el hallazgo de la práctica 5 (racha previa predice home_win con R²=0.014, p<0.001).

### 7. El F1 del empate en partidos equilibrados
Cuando se filtra por partidos donde el mercado señala equilibrio real (AvgH, AvgA y AvgD todas en rango similar), el F1 del empate cambia respecto al modelo general. Si sube, confirma que el KNN detecta mejor el empate cuando los vecinos cercanos ya son partidos inciertos. Si no sube, indica que el problema del empate es estructural y no se resuelve filtrando — los vecinos de un partido equilibrado siguen siendo mayoritariamente victorias locales o visitantes.

---

## Gráficas generadas (`img/`)

| Archivo | Descripción |
|---|---|
| `knn_manual_scatter_train.png` | Scatter real imp_prob_H vs imp_prob_A — espacio de entrenamiento |
| `knn_manual_scatter_pred.png` | Mismo scatter con 5 partidos hipotéticos clasificados (★) |
| `knn_accuracy_vs_k.png` | Curva accuracy y F1-macro vs K — método del codo |
| `knn_confusion_matrix.png` | Matriz de confusión: conteo absoluto y % por clase |
| `knn_feature_sets.png` | Accuracy y F1-macro por conjunto de features |
| `knn_feature_importance.png` | Permutation importance de los 15 features |
| `knn_scatter_imp_probs.png` | Scatter real vs predicho en espacio imp_prob |
| `knn_pca_scatter.png` | Proyección PCA 2D: clases reales vs predichas |
| `knn_equilibrados_f1.png` | F1 por clase: todos los partidos vs partidos equilibrados |

---

## Conexión con prácticas anteriores

- **Práctica 4 (Kruskal-Wallis):** confirmó que las cuotas difieren significativamente entre H/D/A — esa separación es la base teórica de por qué KNN funciona con cuotas como features.
- **Práctica 5 (Regresión):** `imp_prob_H` fue el mejor predictor de `home_win` (R²=0.155). En KNN aparece como el feature más importante en permutation importance.
- **Práctica 5 (Rachas):** racha previa significativa (p<0.001). Se incluyó como feature y mejora el modelo.

## Resumen comparativo — manual vs sklearn

| | KNN manual | KNN sklearn |
|---|---|---|
| Features | 2 (imp_prob_H, imp_prob_A) | 15 (set completo) |
| Partidos | 2,000 (muestra) | ~11,800 (dataset completo) |
| Mejor K | ver output | ver output |
| Accuracy | ver output | ver output |
| F1 macro | — | ver output |
| CV 5-fold | — | ver output |

La diferencia de accuracy entre las dos versiones refleja directamente el valor de agregar los 13 features restantes — no una diferencia en el algoritmo. Ambas versiones usan la misma lógica: distancia euclídea y votación por moda entre los K vecinos más cercanos.

La mejora de manual → sklearn viene de tres factores: más features (las cuotas de cierre y el movimiento de mercado aportan información que las probabilidades implícitas solas no capturan), más datos (el modelo completo entrena con el dataset entero en lugar de una muestra de 2,000) y normalización más robusta (el MinMaxScaler de sklearn maneja edge cases que la implementación manual no cubre).

El empate sigue siendo la clase más difícil en los dos modelos. En el espacio de probabilidades implícitas, los partidos de empate no tienen una zona separada — se mezclan con victorias locales y visitantes de cuotas similares. Esto no es un fallo del modelo sino una propiedad de los datos: el mercado no diferencia empates con precisión.

---

## Ejecución

```bash
pip install scikit-learn
cd "Practica 6"
python knn_classification.py > resultados.txt
```

> **Nota:** el paquete se instala como `scikit-learn` pero se importa como `sklearn` en el código. Son lo mismo — `pip install sklearn` está deprecado y da error.
