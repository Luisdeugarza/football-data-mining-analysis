import sys
sys.stdout.reconfigure(encoding='utf-8')

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from scipy.stats import mode
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score)
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
import os
from tabulate import tabulate

os.makedirs("img", exist_ok=True)

def print_tabulate(df: pd.DataFrame):
    print(tabulate(df, headers=df.columns, tablefmt="orgtbl"))

LIGAS_NAME   = {"E0":"Premier League","SP1":"La Liga","D1":"Bundesliga",
                "I1":"Serie A","F1":"Ligue 1"}
LIGA_COLORS  = {"E0":"#3498db","SP1":"#e74c3c","D1":"#f39c12",
                "I1":"#2ecc71","F1":"#9b59b6"}
LABEL_COLORS = {"H":"#3498db","D":"#f39c12","A":"#e74c3c"}
MARKERS      = {"H":"o","D":"s","A":"^"}


# funciones de la implementación manual del algoritmo KNN

def get_cmap(n, name="hsv"):
    return plt.cm.get_cmap(name, n)

def euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """calculé la distancia euclídea entre dos puntos n-dimensionales."""
    return float(np.sqrt(np.sum((p2 - p1) ** 2)))

def k_nearest_neighbors(
    points:     List[np.ndarray],
    labels:     List[str],
    input_data: List[np.ndarray],
    k:          int,
) -> List[str]:
    """
    implementé KNN desde cero:
      1. calculo distancia euclídea de cada input a todos los puntos de entrenamiento
      2. tomo los K índices con menor distancia
      3. voto por moda entre las etiquetas de esos K vecinos
    """
    labels_arr    = np.array(labels)
    unique_labels = pd.unique(labels_arr)
    label_idx     = {lbl: i for i, lbl in enumerate(unique_labels)}
    idx_label     = {i: lbl for lbl, i in label_idx.items()}

    predictions = []
    for input_point in input_data:
        distances = [euclidean_distance(input_point, pt) for pt in points]
        k_nearest = np.argsort(distances)[:k]
        k_labels  = [label_idx[labels_arr[i]] for i in k_nearest]
        pred_idx  = mode(k_labels, keepdims=True).mode[0]
        predictions.append(idx_label[pred_idx])
    return predictions

def scatter_group_by(
    file_path: str, df: pd.DataFrame,
    x_col: str, y_col: str, label_col: str,
    title: str = "", new_points: List = None,
):
    """grafiqué el scatter de grupos con colores y opcionalmente nuevos puntos."""
    fig, ax = plt.subplots(figsize=(9, 6))
    labels = pd.unique(df[label_col])
    cmap   = get_cmap(len(labels) + 1)
    for i, lbl in enumerate(labels):
        sub = df[df[label_col] == lbl]
        ax.scatter(sub[x_col], sub[y_col], label=lbl,
                   color=LABEL_COLORS.get(lbl, cmap(i)),
                   marker=MARKERS.get(lbl, "o"),
                   alpha=0.45, s=18)
    if new_points:
        for pt, pred in new_points:
            ax.scatter(pt[0], pt[1],
                       color=LABEL_COLORS.get(pred, "black"),
                       marker="*", s=280, edgecolors="black", linewidths=0.8,
                       zorder=5)
            ax.annotate(f"pred:{pred}", xy=(pt[0], pt[1]),
                        xytext=(pt[0]+0.01, pt[1]+0.01), fontsize=8)
    ax.set_xlabel(x_col); ax.set_ylabel(y_col)
    ax.set_title(title or f"{y_col} vs {x_col}", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(file_path, dpi=130)
    plt.close()


# cargué el dataset y construí todas las variables derivadas necesarias

df = pd.read_csv("../Practica 1/data/clean/football_clean.csv", parse_dates=["Date"])

df["imp_prob_H"]  = round(1 / df["AvgH"], 4)
df["imp_prob_D"]  = round(1 / df["AvgD"], 4)
df["imp_prob_A"]  = round(1 / df["AvgA"], 4)
df["overround"]   = round(df["imp_prob_H"] + df["imp_prob_D"] + df["imp_prob_A"], 4)
df["odds_move_H"] = round(df["AvgCH"] - df["AvgH"], 4)
df["odds_move_A"] = round(df["AvgCA"] - df["AvgA"], 4)
df["odds_move_D"] = round(df["AvgCD"] - df["AvgD"], 4)
df["diff_imp"]    = df["imp_prob_H"] - df["imp_prob_A"]

# construí la racha previa del local
df2 = df.sort_values("Date").copy()
streak_map = {}
streak_list = []
for _, row in df2.iterrows():
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
df2["racha_previa"] = streak_list
df = df2.copy()

ligas = sorted(df["Div"].unique())
print(f"dataset: {len(df):,} partidos | {len(ligas)} ligas")
print(f"distribución FTR: {df['FTR'].value_counts().to_dict()}")
print(f"baseline clase mayoritaria: {df['FTR'].value_counts(normalize=True).max():.4f}")



# parte 1: KNN implementado desde cero con datos reales (2 features, graficable)

print(f"\n{'='*60}")
print("  PARTE 1 — KNN DESDE CERO")
print(f"{'='*60}")
print("""
  implementé el algoritmo KNN manualmente usando distancia euclídea
  y votación por moda, igual que el algoritmo base sin librerías.

  features usados: imp_prob_H (eje X) e imp_prob_A (eje Y)
  — dos dimensiones permiten visualizar el espacio de decisión
  — son los features más importantes según la práctica 5

  etiquetas: H (victoria local) | D (empate) | A (victoria visitante)
""")

# preparé el dataset en 2D — solo imp_prob_H e imp_prob_A para poder graficarlo
data_2d = df[["imp_prob_H", "imp_prob_A", "FTR"]].dropna()

# tomé una muestra de 2000 partidos para que la implementación manual sea manejable en tiempo
MUESTRA_N = 2000
sample_df = data_2d.sample(n=MUESTRA_N, random_state=42).reset_index(drop=True)

# dividí la muestra 80/20 a mano sin sklearn
split_idx  = int(MUESTRA_N * 0.8)
train_2d   = sample_df.iloc[:split_idx]
test_2d    = sample_df.iloc[split_idx:]

# normalicé a mano con MinMax usando solo los rangos del train para evitar data leakage
min_h = train_2d["imp_prob_H"].min(); max_h = train_2d["imp_prob_H"].max()
min_a = train_2d["imp_prob_A"].min(); max_a = train_2d["imp_prob_A"].max()

def scale_2d(row):
    return np.array([
        (row["imp_prob_H"] - min_h) / (max_h - min_h + 1e-9),
        (row["imp_prob_A"] - min_a) / (max_a - min_a + 1e-9),
    ])

train_points = [scale_2d(r) for _, r in train_2d.iterrows()]
test_points  = [scale_2d(r) for _, r in test_2d.iterrows()]
train_labels = train_2d["FTR"].tolist()
test_labels  = test_2d["FTR"].tolist()

print(f"  train: {len(train_points)} partidos | test: {len(test_points)} partidos")

# grafiqué el espacio de entrenamiento para ver cómo se distribuyen las tres clases
scatter_group_by(
    "img/knn_manual_scatter_train.png",
    train_2d, "imp_prob_H", "imp_prob_A", "FTR",
    title="Espacio de entrenamiento — imp_prob_H vs imp_prob_A (H/D/A)",
)
print("  grafica guardada: img/knn_manual_scatter_train.png")

# busqué el K óptimo probando de 1 a 15 con la implementación manual
print("\n  buscando K óptimo (K=1 a 15) con implementación manual...")
k_range_manual = range(1, 16)
acc_manual = []
for k in k_range_manual:
    preds = k_nearest_neighbors(train_points, train_labels, test_points, k)
    acc_manual.append(round(sum(p == t for p, t in zip(preds, test_labels)) / len(test_labels), 4))

best_k_manual = list(k_range_manual)[np.argmax(acc_manual)]
best_acc_manual = max(acc_manual)
print(f"  mejor K manual: {best_k_manual}  accuracy: {best_acc_manual:.4f}")

print_tabulate(pd.DataFrame({
    "K":           list(k_range_manual),
    "acc_manual":  acc_manual,
}))

# corrí las predicciones finales con el mejor K encontrado
preds_manual = k_nearest_neighbors(
    train_points, train_labels, test_points, best_k_manual
)
acc_final_manual = sum(p == t for p, t in zip(preds_manual, test_labels)) / len(test_labels)

print(f"\n  accuracy implementación manual K={best_k_manual}: {acc_final_manual:.4f}")
print(f"  distribución predicha: {pd.Series(preds_manual).value_counts().to_dict()}")
print(f"  distribución real:     {pd.Series(test_labels).value_counts().to_dict()}")

# predije 5 partidos hipotéticos con cuotas conocidas para mostrar el modelo en acción
print("\n  predicción de partidos hipotéticos (implementación manual):")
hipoteticos = [
    # avg_h, avg_a, descripcion
    (1.40, 7.50, "favorito local claro"),
    (2.50, 2.80, "partido equilibrado"),
    (5.00, 1.65, "favorito visitante"),
    (1.15, 15.0, "local muy favorito"),
    (3.20, 3.50, "casi moneda al aire"),
]
new_points_plot = []
print(f"  {'descripcion':<28} {'AvgH':>6} {'AvgA':>6} {'pred':>6}")
print("  " + "-"*50)
for avg_h, avg_a, desc in hipoteticos:
    iph = round(1/avg_h, 4); ipa = round(1/avg_a, 4)
    pt  = np.array([
        (iph - min_h) / (max_h - min_h + 1e-9),
        (ipa - min_a) / (max_a - min_a + 1e-9),
    ])
    pred = k_nearest_neighbors(train_points, train_labels, [pt], best_k_manual)[0]
    print(f"  {desc:<28} {avg_h:>6.2f} {avg_a:>6.2f} {pred:>6}")
    new_points_plot.append((np.array([iph, ipa]), pred))

# grafiqué el mismo scatter con los puntos hipotéticos marcados con estrella
scatter_group_by(
    "img/knn_manual_scatter_pred.png",
    train_2d, "imp_prob_H", "imp_prob_A", "FTR",
    title="KNN manual — partidos hipotéticos clasificados (★)",
    new_points=new_points_plot,
)
print("\n  grafica guardada: img/knn_manual_scatter_pred.png")



# parte 2: KNN con sklearn usando los 15 features disponibles en el dataset

print(f"\n{'='*60}")
print("  PARTE 2 — KNN CON SKLEARN (modelo completo, 15 features)")
print(f"{'='*60}")
print("""
  usé sklearn para el modelo completo con todos los features disponibles.
  sklearn optimiza el cálculo de distancias con estructuras KD-tree/Ball-tree
  lo que permite usar 15 features y 11,851 partidos eficientemente.
""")

FEATURE_SETS = {
    "probs_implicitas": ["imp_prob_H","imp_prob_D","imp_prob_A","overround"],
    "cuotas_apertura":  ["AvgH","AvgA","AvgD","overround"],
    "cuotas_cierre":    ["AvgCH","AvgCA","AvgCD","overround"],
    "movimiento":       ["odds_move_H","odds_move_A","odds_move_D"],
    "completo":         ["imp_prob_H","imp_prob_D","imp_prob_A",
                         "AvgH","AvgA","AvgD",
                         "AvgCH","AvgCA","AvgCD",
                         "odds_move_H","odds_move_A","odds_move_D",
                         "overround","diff_imp","racha_previa"],
}
MAIN_FEATURES = FEATURE_SETS["completo"]

data_main = df[MAIN_FEATURES + ["FTR","Div","HomeTeam","AwayTeam","Date"]].dropna()
X_all = data_main[MAIN_FEATURES]
y_all = data_main["FTR"]

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)
idx_test = X_test.index

scaler = MinMaxScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)


# busqué el K óptimo con el método del codo probando K=1 a 30

print(f"\n{'='*60}")
print("  1. BUSQUEDA DEL K OPTIMO — metodo del codo")
print(f"{'='*60}")

k_range      = range(1, 31)
acc_train_l  = []
acc_test_l   = []
f1_test_l    = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_s, y_train)
    p_tr = knn.predict(X_train_s)
    p_te = knn.predict(X_test_s)
    acc_train_l.append(accuracy_score(y_train, p_tr))
    acc_test_l.append(accuracy_score(y_test,  p_te))
    f1_test_l.append(f1_score(y_test, p_te, average="macro"))

best_k   = list(k_range)[np.argmax(acc_test_l)]
best_acc = max(acc_test_l)
print(f"  mejor K: {best_k}  accuracy test: {best_acc:.4f}")

print_tabulate(pd.DataFrame({
    "K":         list(k_range),
    "acc_train": [round(a,4) for a in acc_train_l],
    "acc_test":  [round(a,4) for a in acc_test_l],
    "f1_macro":  [round(a,4) for a in f1_test_l],
}))

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(list(k_range), acc_train_l, label="accuracy train", color="#3498db", linewidth=2)
ax.plot(list(k_range), acc_test_l,  label="accuracy test",  color="#e74c3c", linewidth=2)
ax.plot(list(k_range), f1_test_l,   label="F1 macro test",
        color="#9b59b6", linewidth=1.5, linestyle="--")
ax.axvline(best_k, color="#2ecc71", linewidth=1.5, linestyle="--",
           label=f"mejor K={best_k}")
ax.set_xlabel("K (número de vecinos)"); ax.set_ylabel("métrica")
ax.set_title("Accuracy y F1-macro vs K — método del codo", fontweight="bold")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("img/knn_accuracy_vs_k.png", dpi=130)
plt.close()
print("  grafica guardada: img/knn_accuracy_vs_k.png")


# validé con 5-fold estratificado para confirmar que el accuracy no depende del split

print(f"\n{'='*60}")
print("  2. VALIDACION CRUZADA 5-FOLD")
print(f"{'='*60}")

X_all_s = scaler.fit_transform(X_all)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    KNeighborsClassifier(n_neighbors=best_k),
    X_all_s, y_all, cv=cv, scoring="accuracy"
)
print(f"  scores por fold: {[round(s,4) for s in cv_scores]}")
print(f"  media:           {cv_scores.mean():.4f}")
print(f"  desv. estándar:  {cv_scores.std():.4f}")
print(f"  intervalo 95%:   [{cv_scores.mean()-2*cv_scores.std():.4f}, "
      f"{cv_scores.mean()+2*cv_scores.std():.4f}]")


# entrené el modelo final con el mejor K y calculé todas las métricas

print(f"\n{'='*60}")
print(f"  3. MODELO FINAL  K={best_k}")
print(f"{'='*60}")

scaler2   = MinMaxScaler()
X_train_s2 = scaler2.fit_transform(X_train)
X_test_s2  = scaler2.transform(X_test)

knn_final = KNeighborsClassifier(n_neighbors=best_k)
knn_final.fit(X_train_s2, y_train)
y_pred    = knn_final.predict(X_test_s2)

acc_final = accuracy_score(y_test, y_pred)
f1_final  = f1_score(y_test, y_pred, average="macro")

print(f"\n  accuracy test:  {acc_final:.4f}")
print(f"  F1 macro test:  {f1_final:.4f}")
print(f"  partidos test:  {len(y_test):,}")
print(f"  distribución real:  H={sum(y_test=='H'):4d}  D={sum(y_test=='D'):4d}  A={sum(y_test=='A'):4d}")
print(f"  distribución pred:  H={sum(y_pred=='H'):4d}  D={sum(y_pred=='D'):4d}  A={sum(y_pred=='A'):4d}")
print("\n  classification report:")
print(classification_report(y_test, y_pred, target_names=["H","D","A"]))

report_dict = classification_report(y_test, y_pred, target_names=["H","D","A"],
                                     output_dict=True)
print_tabulate(pd.DataFrame([{
    "clase":     c,
    "precision": round(report_dict[c]["precision"], 4),
    "recall":    round(report_dict[c]["recall"], 4),
    "f1":        round(report_dict[c]["f1-score"], 4),
    "support":   int(report_dict[c]["support"]),
} for c in ["H","D","A"]]))


# grafiqué la matriz de confusión en conteo absoluto y porcentaje por clase

print(f"\n{'='*60}")
print("  4. MATRIZ DE CONFUSION")
print(f"{'='*60}")

cm     = confusion_matrix(y_test, y_pred, labels=["H","D","A"])
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, mat, title, fmt in zip(
    axes,
    [cm, cm_pct],
    ["Conteo absoluto", "% por clase real (recall)"],
    [".0f", ".1f"]
):
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks([0,1,2]); ax.set_yticks([0,1,2])
    ax.set_xticklabels(["H","D","A"]); ax.set_yticklabels(["H","D","A"])
    ax.set_xlabel("predicción"); ax.set_ylabel("real")
    ax.set_title(title, fontweight="bold")
    for i in range(3):
        for j in range(3):
            v   = mat[i,j]
            suf = "%" if fmt == ".1f" else ""
            ax.text(j, i, f"{v:{fmt}}{suf}", ha="center", va="center",
                    fontsize=11, color="white" if v > mat.max()*0.5 else "black")
    plt.colorbar(im, ax=ax)
fig.suptitle(f"Matriz de confusión — KNN K={best_k}", fontweight="bold", fontsize=12)
plt.tight_layout()
plt.savefig("img/knn_confusion_matrix.png", dpi=130)
plt.close()
print("  grafica guardada: img/knn_confusion_matrix.png")

print_tabulate(pd.DataFrame(cm, index=["real H","real D","real A"],
                             columns=["pred H","pred D","pred A"]).reset_index())


# comparé el accuracy de cinco conjuntos distintos de features para ver cuál clasifica mejor

print(f"\n{'='*60}")
print("  5. COMPARACION DE FEATURE SETS")
print(f"{'='*60}")

fs_results = []
for nombre, features in FEATURE_SETS.items():
    cols_ok = [f for f in features if f in df.columns]
    data_fs = df[cols_ok + ["FTR"]].dropna()
    X_fs = data_fs[cols_ok]; y_fs = data_fs["FTR"]
    Xtr, Xte, ytr, yte = train_test_split(
        X_fs, y_fs, test_size=0.2, random_state=42, stratify=y_fs
    )
    sc      = MinMaxScaler()
    Xtr_s   = sc.fit_transform(Xtr)
    Xte_s   = sc.transform(Xte)
    bst_a   = 0; bst_k = 1
    for k in range(1, 21):
        knn_t = KNeighborsClassifier(n_neighbors=k)
        knn_t.fit(Xtr_s, ytr)
        a = accuracy_score(yte, knn_t.predict(Xte_s))
        if a > bst_a: bst_a = a; bst_k = k
    knn_b = KNeighborsClassifier(n_neighbors=bst_k)
    knn_b.fit(Xtr_s, ytr)
    f1_fs = f1_score(yte, knn_b.predict(Xte_s), average="macro")
    fs_results.append({
        "feature_set": nombre,
        "n_features":  len(cols_ok),
        "mejor_k":     bst_k,
        "accuracy":    round(bst_a, 4),
        "f1_macro":    round(f1_fs, 4),
    })

fs_df = pd.DataFrame(fs_results).sort_values("accuracy", ascending=False).reset_index(drop=True)
print_tabulate(fs_df)

fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(fs_df)); w = 0.35
b1 = ax.bar(x - w/2, fs_df["accuracy"], w, label="accuracy", color="#3498db", alpha=0.85)
b2 = ax.bar(x + w/2, fs_df["f1_macro"], w, label="F1 macro",  color="#9b59b6", alpha=0.85)
for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f"{bar.get_height():.3f}", ha="center", fontsize=8)
ax.axhline(1/3, color="red", linewidth=1, linestyle="--", label="baseline azar")
ax.set_xticks(x); ax.set_xticklabels(fs_df["feature_set"], rotation=15)
ax.set_ylabel("métrica")
ax.set_title("Accuracy y F1 por conjunto de features — KNN sklearn", fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig("img/knn_feature_sets.png", dpi=130)
plt.close()
print("  grafica guardada: img/knn_feature_sets.png")


# calculé la importancia de cada feature permutándolos y midiendo cuánto cae el accuracy

print(f"\n{'='*60}")
print("  6. IMPORTANCIA DE FEATURES (permutation importance)")
print(f"{'='*60}")

perm = permutation_importance(knn_final, X_test_s2, y_test,
                               n_repeats=10, random_state=42, scoring="accuracy")
imp_df = pd.DataFrame({
    "feature":     MAIN_FEATURES,
    "importancia": perm.importances_mean.round(4),
    "std":         perm.importances_std.round(4),
}).sort_values("importancia", ascending=False).reset_index(drop=True)
print_tabulate(imp_df)

fig, ax = plt.subplots(figsize=(11, 7))
colors_imp = ["#e74c3c" if v > 0 else "#95a5a6" for v in imp_df["importancia"]]
ax.barh(imp_df["feature"][::-1], imp_df["importancia"][::-1],
        xerr=imp_df["std"][::-1], color=colors_imp[::-1], alpha=0.85, capsize=3)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("reducción en accuracy al permutar (mayor = más importante)")
ax.set_title("Importancia de features — KNN permutation importance", fontweight="bold")
plt.tight_layout()
plt.savefig("img/knn_feature_importance.png", dpi=130)
plt.close()
print("  grafica guardada: img/knn_feature_importance.png")


# grafiqué el espacio real vs predicho usando los dos features más importantes directamente

print(f"\n{'='*60}")
print("  7. SCATTER imp_prob_H vs imp_prob_A — real vs predicho")
print(f"{'='*60}")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for ax, (labels, titulo) in zip(axes, [
    (y_test.values, "Clases REALES"),
    (y_pred,        "Clases PREDICHAS por KNN sklearn"),
]):
    for lbl in ["H","D","A"]:
        mask = labels == lbl
        sub  = X_test[mask]
        ax.scatter(sub["imp_prob_H"], sub["imp_prob_A"],
                   c=LABEL_COLORS[lbl], marker=MARKERS[lbl],
                   alpha=0.35, s=12, label=f"{lbl} (n={mask.sum()})")
    ax.set_xlabel("imp_prob_H  (prob. implícita local)")
    ax.set_ylabel("imp_prob_A  (prob. implícita visitante)")
    ax.set_title(titulo, fontweight="bold")
    ax.legend(fontsize=8)
fig.suptitle("Espacio de decisión: prob. implícita local vs visitante",
             fontweight="bold", fontsize=12)
plt.tight_layout()
plt.savefig("img/knn_scatter_imp_probs.png", dpi=130)
plt.close()
print("  grafica guardada: img/knn_scatter_imp_probs.png")


# proyecté los 15 features a 2D con PCA para visualizar cómo separa las clases el modelo

print(f"\n{'='*60}")
print("  8. VISUALIZACION PCA 2D")
print(f"{'='*60}")

pca       = PCA(n_components=2, random_state=42)
X_te_pca  = pca.fit_transform(X_test_s2)
var_exp   = pca.explained_variance_ratio_
print(f"  varianza explicada — PC1: {var_exp[0]:.3f}  PC2: {var_exp[1]:.3f}  "
      f"total: {sum(var_exp):.3f}")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for ax, (labels, titulo) in zip(axes, [
    (y_test.values, "Clases REALES"),
    (y_pred,        "Clases PREDICHAS por KNN sklearn"),
]):
    for lbl in ["H","D","A"]:
        mask = labels == lbl
        ax.scatter(X_te_pca[mask, 0], X_te_pca[mask, 1],
                   c=LABEL_COLORS[lbl], marker=MARKERS[lbl],
                   alpha=0.4, s=12, label=lbl)
    ax.set_xlabel(f"PC1 ({var_exp[0]:.1%} var.)")
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1%} var.)")
    ax.set_title(titulo, fontweight="bold")
    ax.legend()
fig.suptitle("Espacio de decisión KNN proyectado en 2D (PCA)",
             fontweight="bold", fontsize=12)
plt.tight_layout()
plt.savefig("img/knn_pca_scatter.png", dpi=130)
plt.close()
print("  grafica guardada: img/knn_pca_scatter.png")


# desglosé el accuracy por liga para ver si el modelo funciona igual en todas las competiciones

print(f"\n{'='*60}")
print("  9. ACCURACY POR LIGA")
print(f"{'='*60}")

test_div = data_main.loc[idx_test, "Div"]
liga_acc = []
for liga in ligas:
    mask = test_div == liga
    if mask.sum() < 10: continue
    acc_l = accuracy_score(y_test[mask], y_pred[mask])
    f1_l  = f1_score(y_test[mask], y_pred[mask], average="macro")
    dist  = y_test[mask].value_counts().to_dict()
    liga_acc.append({
        "liga":     LIGAS_NAME[liga],
        "n_test":   int(mask.sum()),
        "accuracy": round(acc_l, 4),
        "f1_macro": round(f1_l, 4),
        "pct_H":    round(dist.get("H",0)/mask.sum()*100,1),
        "pct_D":    round(dist.get("D",0)/mask.sum()*100,1),
        "pct_A":    round(dist.get("A",0)/mask.sum()*100,1),
    })
print_tabulate(pd.DataFrame(liga_acc))

fig, ax = plt.subplots(figsize=(9, 5))
la_df = pd.DataFrame(liga_acc)
bars  = ax.bar(la_df["liga"], la_df["accuracy"],
               color=[LIGA_COLORS[l] for l in ligas], alpha=0.85)
for bar, val in zip(bars, la_df["accuracy"]):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.002,
            f"{val:.4f}", ha="center", fontsize=9)
ax.axhline(1/3, color="red", linewidth=1, linestyle="--", label="baseline azar (33.3%)")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy del KNN por liga", fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig("img/knn_accuracy_por_liga.png", dpi=130)
plt.close()
print("  grafica guardada: img/knn_accuracy_por_liga.png")


# analicé qué tipo de errores cometió el modelo y en qué partidos fue más confiado al equivocarse

print(f"\n{'='*60}")
print("  10. ANALISIS DE ERRORES")
print(f"{'='*60}")

test_meta = data_main.loc[idx_test, ["HomeTeam","AwayTeam","Date","Div"]].copy()
test_meta["real"]     = y_test.values
test_meta["pred"]     = y_pred
test_meta["correcto"] = (test_meta["real"] == test_meta["pred"]).astype(int)

errores = test_meta[test_meta["correcto"] == 0]
print(f"\n  total errores: {len(errores):,} de {len(test_meta):,} "
      f"({len(errores)/len(test_meta)*100:.1f}%)")

confusiones = (errores.groupby(["real","pred"]).size()
               .reset_index(name="n")
               .assign(pct=lambda d: round(d["n"]/len(errores)*100, 2))
               .sort_values("n", ascending=False)
               .reset_index(drop=True))
print("\n  tipos de error (real → predicho):")
print_tabulate(confusiones)

test_meta_full = test_meta.join(X_test[["imp_prob_H","imp_prob_A","AvgH","AvgA"]])
test_meta_full["max_imp"] = test_meta_full[["imp_prob_H","imp_prob_A"]].max(axis=1)
worst = (test_meta_full[test_meta_full["correcto"]==0]
         .sort_values("max_imp", ascending=False)
         .head(10)
         [["HomeTeam","AwayTeam","Date","real","pred","AvgH","AvgA","max_imp"]]
         .reset_index(drop=True))
print("\n  10 errores en partidos con mayor desequilibrio de cuota:")
print_tabulate(worst)


# definí una función que recibe cuotas de un partido nuevo y devuelve la predicción con votos

print(f"\n{'='*60}")
print("  11. PREDICCION DE NUEVOS PARTIDOS")
print(f"{'='*60}")

def predecir_partido(avg_h, avg_d, avg_a, avg_ch, avg_cd, avg_ca, racha=0):
    """predice FTR dado cuotas de apertura y cierre usando el KNN entrenado."""
    iph  = round(1/avg_h, 4); ipd = round(1/avg_d, 4); ipa = round(1/avg_a, 4)
    over = round(iph+ipd+ipa, 4)
    mvh  = round(avg_ch-avg_h, 4); mva = round(avg_ca-avg_a, 4); mvd = round(avg_cd-avg_d, 4)
    diff = round(iph-ipa, 4)
    ent  = pd.DataFrame([[iph,ipd,ipa,avg_h,avg_a,avg_d,avg_ch,avg_ca,avg_cd,
                          mvh,mva,mvd,over,diff,racha]], columns=MAIN_FEATURES)
    ent_s = scaler2.transform(ent)
    pred  = knn_final.predict(ent_s)[0]
    probs = knn_final.predict_proba(ent_s)[0]
    cls   = list(knn_final.classes_)
    return {
        "pred":   pred,
        "prob_H": round(probs[cls.index("H")]*100, 1),
        "prob_D": round(probs[cls.index("D")]*100, 1),
        "prob_A": round(probs[cls.index("A")]*100, 1),
    }

ejemplos = [
    {"desc":"favorito local claro",  "p":(1.40,4.50,7.50,1.38,4.60,7.80, 2)},
    {"desc":"partido equilibrado",   "p":(2.50,3.20,2.80,2.55,3.15,2.75, 0)},
    {"desc":"favorito visitante",    "p":(5.00,4.00,1.65,5.10,3.90,1.60,-1)},
    {"desc":"local muy favorito",    "p":(1.15,7.00,15.0,1.14,7.10,15.5, 3)},
    {"desc":"cuota local sube",      "p":(2.20,3.30,3.10,2.45,3.20,3.00, 1)},
]
print(f"\n  {'descripcion':<25} {'pred':>5} {'%H':>6} {'%D':>6} {'%A':>6}")
print("  " + "-"*52)
for ej in ejemplos:
    r = predecir_partido(*ej["p"])
    print(f"  {ej['desc']:<25} {r['pred']:>5}  {r['prob_H']:>5.1f}%  "
          f"{r['prob_D']:>5.1f}%  {r['prob_A']:>5.1f}%")



# imprimeción de ambos modelos

print(f"\n{'='*60}")
print("  NUMEROS FINALES")
print(f"{'='*60}")
print(f"  KNN manual:   K={best_k_manual}  acc={acc_final_manual:.4f}  features=2  muestra={MUESTRA_N:,}")
print(f"  KNN sklearn:  K={best_k}  acc={acc_final:.4f}  features={len(MAIN_FEATURES)}  partidos={len(data_main):,}")
print(f"  F1 macro:     {f1_final:.4f}")
print(f"  CV 5-fold:    {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"  mejor set:    {fs_df.iloc[0]['feature_set']} (acc={fs_df.iloc[0]['accuracy']:.4f})")

print("\nlisto — todas las graficas guardadas en img/")