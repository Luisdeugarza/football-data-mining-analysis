import sys
sys.stdout.reconfigure(encoding='utf-8')

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import os
from tabulate import tabulate

os.makedirs("img", exist_ok=True)

def print_tabulate(df: pd.DataFrame):
    print(tabulate(df, headers=df.columns, tablefmt="orgtbl"))

LIGAS_NAME     = {"E0":"Premier League","SP1":"La Liga","D1":"Bundesliga",
                  "I1":"Serie A","F1":"Ligue 1"}
LIGA_COLORS    = {"E0":"#3498db","SP1":"#e74c3c","D1":"#f39c12",
                  "I1":"#2ecc71","F1":"#9b59b6"}
SEASON_MAP     = {1920:"2019/20",2021:"2020/21",2122:"2021/22",
                  2223:"2022/23",2324:"2023/24",2425:"2024/25",2526:"2025/26"}
CLUSTER_COLORS = ["#3498db","#e74c3c","#2ecc71","#f39c12","#9b59b6","#1abc9c","#e67e22"]


# implementé el algoritmo K-means desde cero manualmente

def get_cmap(n, name="hsv"):
    return plt.cm.get_cmap(name, n)

def euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """calculé la distancia euclídea entre dos puntos n-dimensionales."""
    return float(np.sqrt(np.sum((p2 - p1) ** 2)))

def calculate_means(points: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """calculé el centroide de cada cluster como la media de sus puntos asignados."""
    means = []
    for cluster in range(k):
        mask = labels == cluster
        if mask.sum() > 0:
            means.append(np.mean(points[mask], axis=0))
        else:
            means.append(points[np.random.randint(len(points))])
    return np.array(means)

def calculate_nearest_centroid(point: np.ndarray, centroids: np.ndarray) -> int:
    """asigné el punto al centroide más cercano por distancia euclídea."""
    distances = [euclidean_distance(point, c) for c in centroids]
    return int(np.argmin(distances))

def k_means_manual(points: np.ndarray, k: int,
                   max_iter: int = 15, random_state: int = 42,
                   x_label: str = "x", y_label: str = "y") -> Tuple[np.ndarray, np.ndarray]:
    """
    implementé K-means desde cero:
      1. inicialicé etiquetas aleatorias para cada punto
      2. calculé el centroide de cada cluster como la media de sus puntos
      3. reasigné cada punto al centroide más cercano
      4. repetí hasta converger o llegar al máximo de iteraciones
    guardé una imagen por iteración para visualizar la convergencia de centroides.
    """
    np.random.seed(random_state)
    x      = np.array(points)
    labels = np.random.randint(0, k, len(x))
    means  = np.zeros((k, x.shape[1]))

    for t in range(max_iter):
        new_means  = calculate_means(x, labels, k)
        new_labels = np.array([calculate_nearest_centroid(p, new_means) for p in x])

        fig, ax = plt.subplots(figsize=(9, 6))
        for cluster in range(k):
            mask = new_labels == cluster
            ax.scatter(x[mask, 0], x[mask, 1],
                       color=CLUSTER_COLORS[cluster % len(CLUSTER_COLORS)],
                       alpha=0.4, s=15, label=f"cluster {cluster}")
        ax.scatter(new_means[:, 0], new_means[:, 1],
                   color="black", marker="X", s=200, zorder=5, label="centroides")
        ax.set_xlabel(x_label); ax.set_ylabel(y_label)
        ax.set_title(f"K-means desde cero — iteración {t+1}/{max_iter}",
                     fontweight="bold")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(f"img/kmeans_iter_{t:02d}.png", dpi=100)
        plt.close()

        if np.array_equal(new_labels, labels):
            print(f"  convergió en iteración {t+1}")
            break
        labels = new_labels
        means  = new_means

    return new_means, new_labels

def scatter_group_by(file_path: str, df: pd.DataFrame,
                     x_col: str, y_col: str, label_col: str,
                     title: str = "", centroids: np.ndarray = None):
    """grafiqué scatter de grupos con colores y opcionalmente centroides marcados."""
    fig, ax = plt.subplots(figsize=(9, 6))
    labels  = sorted(df[label_col].unique())
    for i, lbl in enumerate(labels):
        sub = df[df[label_col] == lbl]
        ax.scatter(sub[x_col], sub[y_col],
                   color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
                   alpha=0.4, s=12, label=str(lbl))
    if centroids is not None:
        ax.scatter(centroids[:, 0], centroids[:, 1],
                   color="black", marker="X", s=200, zorder=5, label="centroides")
    ax.set_xlabel(x_col); ax.set_ylabel(y_col)
    ax.set_title(title or f"{y_col} vs {x_col}", fontweight="bold")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(file_path, dpi=130)
    plt.close()


# cargué el dataset y construí las variables derivadas

df = pd.read_csv("../Practica 1/data/clean/football_clean.csv", parse_dates=["Date"])

df["total_goals"]  = df["FTHG"] + df["FTAG"]
df["ht_goals"]     = df["HTHG"] + df["HTAG"]
df["home_win"]     = (df["FTR"] == "H").astype(int)
df["draw"]         = (df["FTR"] == "D").astype(int)
df["away_win"]     = (df["FTR"] == "A").astype(int)
df["btts"]         = ((df["FTHG"] > 0) & (df["FTAG"] > 0)).astype(int)
df["over25"]       = (df["total_goals"] > 2).astype(int)
df["imp_prob_H"]   = round(1 / df["AvgH"], 4)
df["imp_prob_D"]   = round(1 / df["AvgD"], 4)
df["imp_prob_A"]   = round(1 / df["AvgA"], 4)
df["overround"]    = round(df["imp_prob_H"] + df["imp_prob_D"] + df["imp_prob_A"], 4)
df["odds_move_H"]  = round(df["AvgCH"] - df["AvgH"], 4)
df["odds_move_A"]  = round(df["AvgCA"] - df["AvgA"], 4)
df["odds_move_D"]  = round(df["AvgCD"] - df["AvgD"], 4)
df["diff_imp"]     = df["imp_prob_H"] - df["imp_prob_A"]
df["Season_label"] = df["Season"].map(SEASON_MAP)

ligas = sorted(df["Div"].unique())
print(f"dataset: {len(df):,} partidos | {len(ligas)} ligas")
print(f"distribución FTR: {df['FTR'].value_counts().to_dict()}")


# PARTE 1 — K-MEANS IMPLEMENTADO DESDE CERO

print(f"\n{'='*60}")
print("  PARTE 1 — K-MEANS DESDE CERO")
print(f"{'='*60}")
print("""
  implementé K-means manualmente con distancia euclídea e
  inicialización aleatoria como base del algoritmo.

  features: imp_prob_H (eje X) e total_goals (eje Y)
  2 dimensiones permiten visualizar la convergencia de centroides.
  genera una imagen por iteración en img/kmeans_iter_N.png
""")

FEATURES_2D = ["imp_prob_H", "total_goals"]
data_2d     = df[FEATURES_2D].dropna()
MUESTRA_N   = 2000
sample_2d   = data_2d.sample(n=MUESTRA_N, random_state=42).reset_index(drop=True)

# normalicé manualmente con MinMax para que las escalas sean comparables
min_h = sample_2d["imp_prob_H"].min(); max_h = sample_2d["imp_prob_H"].max()
min_g = sample_2d["total_goals"].min(); max_g = sample_2d["total_goals"].max()

sample_2d_s = sample_2d.copy()
sample_2d_s["imp_prob_H"] = (sample_2d["imp_prob_H"] - min_h) / (max_h - min_h + 1e-9)
sample_2d_s["total_goals"] = (sample_2d["total_goals"] - min_g) / (max_g - min_g + 1e-9)

points_2d = sample_2d_s[FEATURES_2D].values

# grafiqué el espacio antes de clusterizar
scatter_group_by(
    "img/kmeans_espacio_original.png",
    sample_2d.assign(grupo="partidos"),
    "imp_prob_H", "total_goals", "grupo",
    title="Espacio de partidos — imp_prob_H vs total_goals (sin clusters)",
)
print("  grafica guardada: img/kmeans_espacio_original.png")

K_MANUAL = 4
print(f"\n  corriendo K-means manual K={K_MANUAL} sobre {MUESTRA_N:,} partidos...")
centroids_manual, labels_manual = k_means_manual(
    points_2d, k=K_MANUAL,
    x_label="imp_prob_H (norm)", y_label="total_goals (norm)"
)

# mostré los centroides en valores reales (des-normalicé)
print(f"\n  centroides finales (valores reales):")
centroid_rows = []
for i, c in enumerate(centroids_manual):
    iph_real   = c[0] * (max_h - min_h) + min_h
    goals_real = c[1] * (max_g - min_g) + min_g
    n_c        = int((labels_manual == i).sum())
    centroid_rows.append({"cluster": i, "imp_prob_H": round(iph_real,3),
                          "total_goals": round(goals_real,2), "n": n_c})
print_tabulate(pd.DataFrame(centroid_rows))

sample_2d["cluster"] = [str(l) for l in labels_manual]
scatter_group_by(
    "img/kmeans_manual_resultado.png",
    sample_2d, "imp_prob_H", "total_goals", "cluster",
    title=f"K-means desde cero K={K_MANUAL} — resultado final",
    centroids=np.column_stack([
        centroids_manual[:, 0] * (max_h - min_h) + min_h,
        centroids_manual[:, 1] * (max_g - min_g) + min_g,
    ]),
)
print("  grafica guardada: img/kmeans_manual_resultado.png")


# PARTE 2 — K-MEANS CON SKLEARN (6 features, modelo completo)

print(f"\n{'='*60}")
print("  PARTE 2 — K-MEANS CON SKLEARN (modelo completo)")
print(f"{'='*60}")

FEATURES_CLUSTER = [
    "imp_prob_H",   # dominancia del favorito local según el mercado
    "imp_prob_A",   # dominancia del favorito visitante
    "total_goals",  # intensidad goleadora del partido
    "ht_goals",     # ritmo del primer tiempo
    "odds_move_H",  # presión del mercado hacia el local antes del cierre
    "odds_move_A",  # presión del mercado hacia el visitante
]

# incluí las cuotas reales para el cálculo de ROI más adelante
COLS_EXTRA = ["FTR","Div","Season","Season_label","HomeTeam","AwayTeam","Date",
              "home_win","draw","away_win","btts","over25","overround",
              "AvgH","AvgD","AvgA"]
COLS_EXTRA = [c for c in COLS_EXTRA if c in df.columns]

data_cluster = df[FEATURES_CLUSTER + COLS_EXTRA].dropna().copy()
X            = data_cluster[FEATURES_CLUSTER]

scaler   = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

print(f"\n  partidos disponibles: {len(data_cluster):,}")
print(f"  features: {FEATURES_CLUSTER}")


# 1. busqué el K óptimo con método del codo y silhouette score

print(f"\n{'='*60}")
print("  1. BUSQUEDA DEL K OPTIMO")
print(f"{'='*60}")

k_range     = range(2, 11)
inercias    = []
silhouettes = []

for k in k_range:
    km  = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inercias.append(round(km.inertia_, 2))
    sil = silhouette_score(X_scaled, km.labels_, sample_size=3000, random_state=42)
    silhouettes.append(round(sil, 4))

best_k = list(k_range)[np.argmax(silhouettes)]
print(f"  K óptimo por silhouette: {best_k}  (score={max(silhouettes):.4f})")

print_tabulate(pd.DataFrame({
    "K":          list(k_range),
    "inercia":    inercias,
    "silhouette": silhouettes,
}))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(list(k_range), inercias, color="#3498db", linewidth=2, marker="o")
axes[0].set_xlabel("K"); axes[0].set_ylabel("inercia")
axes[0].set_title("Método del codo — inercia vs K", fontweight="bold")
axes[0].grid(alpha=0.3)

axes[1].plot(list(k_range), silhouettes, color="#e74c3c", linewidth=2, marker="s")
axes[1].axvline(best_k, color="#2ecc71", linewidth=1.5, linestyle="--",
                label=f"mejor K={best_k}")
axes[1].set_xlabel("K"); axes[1].set_ylabel("silhouette score")
axes[1].set_title("Silhouette score vs K", fontweight="bold")
axes[1].legend(); axes[1].grid(alpha=0.3)
plt.suptitle("Búsqueda del K óptimo para partidos de fútbol", fontweight="bold")
plt.tight_layout()
plt.savefig("img/kmeans_elbow_silhouette.png", dpi=130)
plt.close()
print("  grafica guardada: img/kmeans_elbow_silhouette.png")


# 2. entrené el modelo final

print(f"\n{'='*60}")
print(f"  2. MODELO FINAL  K={best_k}")
print(f"{'='*60}")

km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
km_final.fit(X_scaled)
data_cluster["cluster"] = km_final.labels_

# mostré los centroides en valores reales (des-normalicé con inverse_transform)
centroids_real = pd.DataFrame(
    scaler.inverse_transform(km_final.cluster_centers_),
    columns=FEATURES_CLUSTER
).round(4)
centroids_real.index.name = "cluster"
print("\n  centroides en valores reales:")
print_tabulate(centroids_real.reset_index())

print("\n  tamaño de cada cluster:")
print_tabulate(
    data_cluster["cluster"].value_counts().sort_index()
    .reset_index()
    .rename(columns={"cluster":"cluster","count":"n_partidos"})
)


# 3. perfilé cada cluster con estadísticas clave

print(f"\n{'='*60}")
print("  3. PERFIL DE CADA CLUSTER")
print(f"{'='*60}")

perfil_rows = []
for cluster in sorted(data_cluster["cluster"].unique()):
    sub = data_cluster[data_cluster["cluster"] == cluster]
    perfil_rows.append({
        "cluster":    cluster,
        "n":          len(sub),
        "avg_goals":  round(sub["total_goals"].mean(), 2),
        "avg_ht":     round(sub["ht_goals"].mean(), 2),
        "avg_iph":    round(sub["imp_prob_H"].mean(), 3),
        "avg_ipa":    round(sub["imp_prob_A"].mean(), 3),
        "avg_mov_h":  round(sub["odds_move_H"].mean(), 4),
        "pct_H":      round(sub["home_win"].mean() * 100, 1),
        "pct_D":      round(sub["draw"].mean() * 100, 1),
        "pct_A":      round(sub["away_win"].mean() * 100, 1),
        "pct_btts":   round(sub["btts"].mean() * 100, 1),
        "pct_over25": round(sub["over25"].mean() * 100, 1),
    })

perfil_df = pd.DataFrame(perfil_rows)
print_tabulate(perfil_df)

# asigné etiquetas descriptivas basadas en el perfil de cada cluster
def etiquetar_cluster(row):
    if row["avg_goals"] >= 3.2:
        return "goleador"
    elif row["avg_iph"] >= 0.56:
        return "dominio_local"
    elif row["avg_ipa"] >= 0.42:
        return "dominio_visitante"
    elif row["avg_goals"] <= 2.0 and row["pct_D"] >= 28:
        return "cerrado_empate"
    else:
        return "equilibrado"

perfil_df["tipo"] = perfil_df.apply(etiquetar_cluster, axis=1)
data_cluster["tipo"] = data_cluster["cluster"].map(
    dict(zip(perfil_df["cluster"], perfil_df["tipo"]))
)

print("\n  etiquetas descriptivas asignadas:")
print_tabulate(perfil_df[["cluster","tipo","n","avg_goals","avg_iph","avg_ipa",
                           "pct_H","pct_D","pct_A","pct_over25"]].reset_index(drop=True))

# gráfica de perfil por cluster con barras de los indicadores clave
indicadores = ["pct_H","pct_D","pct_A","pct_btts","pct_over25"]
fig, axes   = plt.subplots(1, len(indicadores), figsize=(18, 5))
for ax, ind in zip(axes, indicadores):
    vals   = [perfil_df.loc[perfil_df["cluster"]==c, ind].values[0]
              for c in sorted(data_cluster["cluster"].unique())]
    labels = [f"C{c}\n{perfil_df.loc[perfil_df['cluster']==c,'tipo'].values[0]}"
              for c in sorted(data_cluster["cluster"].unique())]
    ax.bar(labels, vals,
           color=[CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
                  for c in sorted(data_cluster["cluster"].unique())],
           alpha=0.85)
    ax.set_title(ind.replace("pct_","% "), fontweight="bold")
    ax.set_ylabel("%"); ax.tick_params(axis="x", rotation=15)
plt.suptitle("Perfil de cada cluster — indicadores principales",
             fontweight="bold", fontsize=12)
plt.tight_layout()
plt.savefig("img/kmeans_perfil_clusters.png", dpi=130)
plt.close()
print("  grafica guardada: img/kmeans_perfil_clusters.png")


# 4. visualicé los clusters en 2D con PCA y directo

print(f"\n{'='*60}")
print("  4. VISUALIZACION 2D")
print(f"{'='*60}")

pca     = PCA(n_components=2, random_state=42)
X_pca   = pca.fit_transform(X_scaled)
var_exp = pca.explained_variance_ratio_
print(f"  varianza explicada PCA — PC1: {var_exp[0]:.3f}  PC2: {var_exp[1]:.3f}  "
      f"total: {sum(var_exp):.3f}")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for ax, (col, title) in zip(axes, [
    ("cluster", f"Clusters K-means (K={best_k})"),
    ("FTR",     "Resultado real (H/D/A)"),
]):
    for i, lbl in enumerate(sorted(data_cluster[col].unique())):
        mask = data_cluster[col] == lbl
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
                   alpha=0.3, s=8, label=str(lbl))
    ax.set_xlabel(f"PC1 ({var_exp[0]:.1%} var.)")
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1%} var.)")
    ax.set_title(title, fontweight="bold"); ax.legend(fontsize=8)
plt.suptitle("Clusters proyectados en 2D (PCA)", fontweight="bold")
plt.tight_layout()
plt.savefig("img/kmeans_pca_scatter.png", dpi=130)
plt.close()
print("  grafica guardada: img/kmeans_pca_scatter.png")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for ax, (col, title) in zip(axes, [
    ("cluster", "Clusters K-means"),
    ("FTR",     "Resultado real"),
]):
    for i, lbl in enumerate(sorted(data_cluster[col].unique())):
        mask = data_cluster[col] == lbl
        ax.scatter(data_cluster.loc[mask, "imp_prob_H"],
                   data_cluster.loc[mask, "total_goals"],
                   color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
                   alpha=0.3, s=8, label=str(lbl))
    ax.set_xlabel("imp_prob_H"); ax.set_ylabel("total_goals")
    ax.set_title(title, fontweight="bold"); ax.legend(fontsize=8)
plt.suptitle("Clusters en espacio imp_prob_H vs total_goals", fontweight="bold")
plt.tight_layout()
plt.savefig("img/kmeans_scatter_directo.png", dpi=130)
plt.close()
print("  grafica guardada: img/kmeans_scatter_directo.png")


# 5. composición por liga y temporada

print(f"\n{'='*60}")
print("  5. COMPOSICION POR LIGA Y TEMPORADA")
print(f"{'='*60}")

liga_cl = (data_cluster.groupby(["Div","cluster"]).size()
           .unstack(fill_value=0))
liga_cl_pct = liga_cl.div(liga_cl.sum(axis=1), axis=0).round(3) * 100
print("\n  distribución de clusters por liga (%):")
print_tabulate(liga_cl_pct.reset_index().rename(columns={"Div":"liga"}))

seas_cl = (data_cluster.groupby(["Season_label","cluster"]).size()
           .unstack(fill_value=0))
seas_cl_pct = seas_cl.div(seas_cl.sum(axis=1), axis=0).round(3) * 100
print("\n  distribución de clusters por temporada (%):")
print_tabulate(seas_cl_pct.reset_index())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
liga_cl_pct.plot(kind="bar", ax=axes[0], color=CLUSTER_COLORS[:best_k], alpha=0.85, width=0.7)
axes[0].set_title("Distribución de clusters por liga", fontweight="bold")
axes[0].set_xlabel(""); axes[0].set_ylabel("% partidos")
axes[0].tick_params(axis="x", rotation=20); axes[0].legend(title="cluster", fontsize=8)

seas_cl_pct.plot(kind="bar", ax=axes[1], color=CLUSTER_COLORS[:best_k], alpha=0.85, width=0.7)
axes[1].set_title("Distribución de clusters por temporada", fontweight="bold")
axes[1].set_xlabel(""); axes[1].set_ylabel("% partidos")
axes[1].tick_params(axis="x", rotation=20); axes[1].legend(title="cluster", fontsize=8)
plt.tight_layout()
plt.savefig("img/kmeans_composicion_liga_temporada.png", dpi=130)
plt.close()
print("  grafica guardada: img/kmeans_composicion_liga_temporada.png")


# 6. ROI hipotético por cluster

print(f"\n{'='*60}")
print("  6. ROI HIPOTETICO POR CLUSTER")
print(f"{'='*60}")
print("""
  simulé apostar siempre al resultado más frecuente de cada cluster
  usando la cuota promedio de apertura real del dataset.
  ROI = (tasa_acierto × cuota_promedio - 1) × 100
  si el ROI es positivo el cluster tiene valor de apuesta histórico.
""")

roi_rows = []
for cluster in sorted(data_cluster["cluster"].unique()):
    sub  = data_cluster[data_cluster["cluster"] == cluster]
    tipo = perfil_df.loc[perfil_df["cluster"] == cluster, "tipo"].values[0]

    pct_h = sub["home_win"].mean()
    pct_d = sub["draw"].mean()
    pct_a = sub["away_win"].mean()

    if pct_h >= pct_d and pct_h >= pct_a:
        res = "H"; aciertos = pct_h; cuota_avg = sub["AvgH"].mean()
    elif pct_a >= pct_h and pct_a >= pct_d:
        res = "A"; aciertos = pct_a; cuota_avg = sub["AvgA"].mean()
    else:
        res = "D"; aciertos = pct_d; cuota_avg = sub["AvgD"].mean()

    roi = round((aciertos * cuota_avg - 1) * 100, 2)
    roi_rows.append({
        "cluster":   cluster,
        "tipo":      tipo,
        "n":         len(sub),
        "res_frec":  res,
        "pct_res":   round(aciertos * 100, 1),
        "avg_cuota": round(cuota_avg, 3),
        "roi_%":     roi,
    })

roi_df = pd.DataFrame(roi_rows).sort_values("roi_%", ascending=False).reset_index(drop=True)
print_tabulate(roi_df)

fig, ax = plt.subplots(figsize=(10, 5))
colors_roi = ["#2ecc71" if v >= 0 else "#e74c3c" for v in roi_df["roi_%"]]
bars = ax.bar(
    [f"C{r['cluster']}\n{r['tipo']}" for _, r in roi_df.iterrows()],
    roi_df["roi_%"], color=colors_roi, alpha=0.85
)
for bar, val in zip(bars, roi_df["roi_%"]):
    ax.text(bar.get_x() + bar.get_width()/2,
            val + (0.3 if val >= 0 else -1.2),
            f"{val:.1f}%", ha="center", fontsize=9)
ax.axhline(0, color="black", linewidth=1)
ax.set_ylabel("ROI hipotético (%)")
ax.set_title("ROI histórico apostando al resultado más frecuente de cada cluster",
             fontweight="bold")
plt.tight_layout()
plt.savefig("img/kmeans_roi_por_cluster.png", dpi=130)
plt.close()
print("  grafica guardada: img/kmeans_roi_por_cluster.png")

# validé la significancia estadística del ROI con test binomial
# un ROI positivo puede ser ruido si la muestra no es suficiente
from scipy.stats import binomtest
print(f"\n  validación estadística del ROI (test binomial):")
print(f"  H0: la tasa de acierto real == tasa implícita por la cuota (mercado perfectamente calibrado)")
sig_rows = []
for _, r in roi_df.iterrows():
    n_total   = int(r["n"])
    n_aciertos = int(round(r["pct_res"] / 100 * n_total))
    p_implicita = round(1 / r["avg_cuota"], 4)
    test      = binomtest(n_aciertos, n_total, p_implicita, alternative="greater")
    sig_rows.append({
        "cluster":    int(r["cluster"]),
        "tipo":       r["tipo"],
        "n":          n_total,
        "aciertos":   n_aciertos,
        "p_implicita": p_implicita,
        "p_real":     round(r["pct_res"]/100, 4),
        "roi_%":      r["roi_%"],
        "p_value":    round(test.pvalue, 4),
        "sig":        "✓ sig" if test.pvalue < 0.05 else "no sig",
    })
print_tabulate(pd.DataFrame(sig_rows))


# 6b. análisis con K=4 forzado para mayor granularidad narrativa

print(f"\n{'='*60}")
print("  6b. K=4 FORZADO — subclusters para mayor granularidad")
print(f"{'='*60}")
print("""
  el silhouette prefiere K=2 — la separación más limpia.
  con K=4 forzado busco si existe más estructura dentro de cada cluster
  que tenga valor interpretativo aunque no sea estadísticamente óptimo.
""")

km4 = KMeans(n_clusters=4, random_state=42, n_init=10)
km4.fit(X_scaled)
data_cluster["cluster4"] = km4.labels_

perfil4_rows = []
for cluster in sorted(data_cluster["cluster4"].unique()):
    sub = data_cluster[data_cluster["cluster4"] == cluster]
    perfil4_rows.append({
        "cluster4":   cluster,
        "n":          len(sub),
        "avg_goals":  round(sub["total_goals"].mean(), 2),
        "avg_iph":    round(sub["imp_prob_H"].mean(), 3),
        "avg_ipa":    round(sub["imp_prob_A"].mean(), 3),
        "pct_H":      round(sub["home_win"].mean() * 100, 1),
        "pct_D":      round(sub["draw"].mean() * 100, 1),
        "pct_A":      round(sub["away_win"].mean() * 100, 1),
        "pct_over25": round(sub["over25"].mean() * 100, 1),
        "pct_btts":   round(sub["btts"].mean() * 100, 1),
    })
perfil4_df = pd.DataFrame(perfil4_rows)
print_tabulate(perfil4_df)

print("\n  ROI por subcluster (K=4):")
roi4_rows = []
for cluster in sorted(data_cluster["cluster4"].unique()):
    sub   = data_cluster[data_cluster["cluster4"] == cluster]
    pct_h = sub["home_win"].mean()
    pct_d = sub["draw"].mean()
    pct_a = sub["away_win"].mean()
    if pct_h >= pct_d and pct_h >= pct_a:
        res = "H"; aciertos = pct_h; cuota_avg = sub["AvgH"].mean()
    elif pct_a >= pct_h and pct_a >= pct_d:
        res = "A"; aciertos = pct_a; cuota_avg = sub["AvgA"].mean()
    else:
        res = "D"; aciertos = pct_d; cuota_avg = sub["AvgD"].mean()
    roi4_rows.append({
        "cluster4":  cluster,
        "n":         len(sub),
        "res_frec":  res,
        "pct_res":   round(aciertos * 100, 1),
        "avg_cuota": round(cuota_avg, 3),
        "roi_%":     round((aciertos * cuota_avg - 1) * 100, 2),
    })
print_tabulate(pd.DataFrame(roi4_rows).sort_values("roi_%", ascending=False).reset_index(drop=True))


print(f"\n{'='*60}")
print("  7. ACCURACY DEL KNN POR CLUSTER — conexion con P6")
print(f"{'='*60}")
print("""
  en P6 el KNN alcanzó accuracy=0.524 global.
  aquí mido si clasifica mejor en algunos tipos de partido.
  nota: el F1 bajo en dominio_local no indica fallo del modelo —
  con 58.8% de victorias locales el KNN tiende a predecir siempre H
  y acierta en accuracy pero falla en recall de D y A (F1 macro bajo).
""")

FEATURES_KNN = [c for c in ["imp_prob_H","imp_prob_D","imp_prob_A",
                              "AvgH","AvgA","AvgD","AvgCH","AvgCA","AvgCD",
                              "odds_move_H","odds_move_A","odds_move_D","overround"]
                if c in df.columns]

data_knn  = df[FEATURES_KNN + ["FTR"]].dropna()
X_knn_all = data_knn[FEATURES_KNN]
y_knn_all = data_knn["FTR"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X_knn_all, y_knn_all, test_size=0.2, random_state=42, stratify=y_knn_all
)
sc_knn   = MinMaxScaler()
X_tr_s   = sc_knn.fit_transform(X_tr)
X_te_s   = sc_knn.transform(X_te)

knn_p6 = KNeighborsClassifier(n_neighbors=39, weights="distance")
knn_p6.fit(X_tr_s, y_tr)

idx_comun = data_cluster.index.intersection(X_te.index)
if len(idx_comun) >= 100:
    y_te_c   = y_te.loc[idx_comun]
    clus_te  = data_cluster.loc[idx_comun, "cluster"]
    tipo_te  = data_cluster.loc[idx_comun, "tipo"]
    X_te_c_s = sc_knn.transform(data_knn.loc[idx_comun, FEATURES_KNN])
    pred_c   = knn_p6.predict(X_te_c_s)

    knn_cl_rows = []
    for cluster in sorted(clus_te.unique()):
        mask  = clus_te == cluster
        if mask.sum() < 20: continue
        acc_c = accuracy_score(y_te_c[mask], pred_c[mask])
        f1_c  = f1_score(y_te_c[mask], pred_c[mask], average="macro")
        tipo_c = tipo_te[mask].iloc[0]
        dist_c = y_te_c[mask].value_counts(normalize=True).to_dict()
        knn_cl_rows.append({
            "cluster":  cluster,
            "tipo":     tipo_c,
            "n_test":   int(mask.sum()),
            "acc_knn":  round(acc_c, 4),
            "f1_macro": round(f1_c, 4),
            "pct_H":    round(dist_c.get("H",0)*100, 1),
            "pct_D":    round(dist_c.get("D",0)*100, 1),
            "pct_A":    round(dist_c.get("A",0)*100, 1),
        })

    knn_cl_df = pd.DataFrame(knn_cl_rows).sort_values("acc_knn", ascending=False)
    print_tabulate(knn_cl_df.reset_index(drop=True))

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        [f"C{int(r['cluster'])}\n{r['tipo']}" for _, r in knn_cl_df.iterrows()],
        knn_cl_df["acc_knn"],
        color=[CLUSTER_COLORS[int(r["cluster"]) % len(CLUSTER_COLORS)]
               for _, r in knn_cl_df.iterrows()],
        alpha=0.85
    )
    for bar, val in zip(bars, knn_cl_df["acc_knn"]):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.003,
                f"{val:.3f}", ha="center", fontsize=9)
    ax.axhline(0.524, color="black", linewidth=1.2, linestyle="--",
               label="accuracy global P6 (0.524)")
    ax.set_ylabel("accuracy del KNN")
    ax.set_title("Accuracy del KNN de P6 por tipo de partido (cluster P7)",
                 fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("img/kmeans_knn_accuracy_por_cluster.png", dpi=130)
    plt.close()
    print("  grafica guardada: img/kmeans_knn_accuracy_por_cluster.png")
else:
    print(f"  solo {len(idx_comun)} partidos en común — omitiendo análisis KNN×cluster")


# 8. top equipos por cluster

print(f"\n{'='*60}")
print("  8. TOP EQUIPOS POR CLUSTER")
print(f"{'='*60}")
print("  qué equipos protagonizan más partidos de cada tipo")

for cluster in sorted(data_cluster["cluster"].unique()):
    sub  = data_cluster[data_cluster["cluster"] == cluster]
    tipo = perfil_df.loc[perfil_df["cluster"] == cluster, "tipo"].values[0]
    equipos = pd.concat([sub["HomeTeam"], sub["AwayTeam"]]).value_counts().head(8)
    print(f"\n  cluster {cluster} ({tipo})  n={len(sub)}:")
    print(f"  {', '.join([f'{eq}({n})' for eq, n in equipos.items()])}")


# 9. irregularidad por cluster — conexión con hallazgo P5

print(f"\n{'='*60}")
print("  9. IRREGULARIDAD POR CLUSTER — conexion con P5")
print(f"{'='*60}")
print("""
  en P5 descubrimos que la irregularidad de resultados previos predice
  peor rendimiento siguiente (p=0.009). aquí veo si algunos clusters
  tienen equipos más irregulares — lo que conectaría con el fenómeno
  Tigres: partidos del cluster 'equilibrado' podrían tener más equipos
  en racha irregular que llevan a resultados sorpresivos.
""")

# construí racha previa del local para todo el dataset
df_sorted  = df.sort_values("Date").copy()
streak_map = {}
streak_list = []
for _, row in df_sorted.iterrows():
    ht = row["HomeTeam"]; at = row["AwayTeam"]
    streak_map.setdefault(ht, 0); streak_map.setdefault(at, 0)
    streak_list.append(streak_map[ht])
    if row["FTR"] == "H":
        streak_map[ht] = max(streak_map[ht], 0) + 1
    elif row["FTR"] == "A":
        streak_map[ht] = min(streak_map[ht], 0) - 1
    else:
        streak_map[ht] = 0
    if row["FTR"] == "A":
        streak_map[at] = max(streak_map[at], 0) + 1
    elif row["FTR"] == "H":
        streak_map[at] = min(streak_map[at], 0) - 1
    else:
        streak_map[at] = 0
df_sorted["racha_previa"] = streak_list

data_cluster["racha_previa"] = df_sorted.loc[data_cluster.index, "racha_previa"]

irr_rows = []
for cluster in sorted(data_cluster["cluster"].unique()):
    sub  = data_cluster[data_cluster["cluster"] == cluster]
    tipo = perfil_df.loc[perfil_df["cluster"] == cluster, "tipo"].values[0]
    irr_rows.append({
        "cluster":        cluster,
        "tipo":           tipo,
        "avg_racha":      round(sub["racha_previa"].mean(), 3),
        "pct_racha_neg":  round((sub["racha_previa"] < 0).mean() * 100, 1),
        "pct_racha_pos":  round((sub["racha_previa"] > 0).mean() * 100, 1),
        "pct_racha_0":    round((sub["racha_previa"] == 0).mean() * 100, 1),
    })

irr_df = pd.DataFrame(irr_rows)
print_tabulate(irr_df)

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(irr_df)); w = 0.28
ax.bar(x - w, irr_df["pct_racha_pos"], w, label="racha positiva", color="#3498db", alpha=0.85)
ax.bar(x,     irr_df["pct_racha_0"],   w, label="neutro",         color="#95a5a6", alpha=0.85)
ax.bar(x + w, irr_df["pct_racha_neg"], w, label="racha negativa", color="#e74c3c", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([f"C{int(r['cluster'])}\n{r['tipo']}" for _, r in irr_df.iterrows()])
ax.set_ylabel("% partidos")
ax.set_title("Estado de racha del local antes del partido por cluster",
             fontweight="bold")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("img/kmeans_racha_por_cluster.png", dpi=130)
plt.close()
print("  grafica guardada: img/kmeans_racha_por_cluster.png")


# números finales

print(f"\n{'='*60}")
print("  NUMEROS FINALES")
print(f"{'='*60}")
print(f"  K-means manual:  K={K_MANUAL}  features=2  muestra={MUESTRA_N:,}")
print(f"  K-means sklearn: K={best_k}  silhouette={max(silhouettes):.4f}  "
      f"features={len(FEATURES_CLUSTER)}  partidos={len(data_cluster):,}")
print(f"  cluster con mayor ROI:         ver tabla sección 6")
print(f"  cluster más fácil de predecir: ver tabla sección 7")
print(f"  cluster más goleador:          {perfil_df.loc[perfil_df['avg_goals'].idxmax(), 'tipo']}")
print(f"  cluster más equilibrado:       {perfil_df.loc[perfil_df['pct_D'].idxmax(), 'tipo']}")

print("\nlisto — todas las graficas guardadas en img/")