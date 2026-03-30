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
print(f"  distribución predicha: {pd.Series([str(p) for p in preds_manual]).value_counts().to_dict()}")
print(f"  distribución real:     {pd.Series([str(t) for t in test_labels]).value_counts().to_dict()}")

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

data_main = df[MAIN_FEATURES + ["FTR","Div","HomeTeam","AwayTeam","Date","Season"]].dropna()
X_all = data_main[MAIN_FEATURES]
y_all = data_main["FTR"]

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)
idx_test = X_test.index

scaler = MinMaxScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)


# busqué el K óptimo con el método del codo probando K=1 a 50
# también comparé votación uniforme vs ponderada por distancia inversa

print(f"\n{'='*60}")
print("  1. BUSQUEDA DEL K OPTIMO — metodo del codo")
print(f"{'='*60}")

k_range      = range(1, 51)
acc_train_l  = []
acc_test_l   = []
f1_test_l    = []
acc_test_wd  = []  # weighted distance

for k in k_range:
    knn_u = KNeighborsClassifier(n_neighbors=k, weights="uniform")
    knn_d = KNeighborsClassifier(n_neighbors=k, weights="distance")
    knn_u.fit(X_train_s, y_train)
    knn_d.fit(X_train_s, y_train)
    p_tr = knn_u.predict(X_train_s)
    p_te = knn_u.predict(X_test_s)
    p_wd = knn_d.predict(X_test_s)
    acc_train_l.append(accuracy_score(y_train, p_tr))
    acc_test_l.append(accuracy_score(y_test,  p_te))
    f1_test_l.append(f1_score(y_test, p_te, average="macro"))
    acc_test_wd.append(accuracy_score(y_test, p_wd))

best_k     = list(k_range)[np.argmax(acc_test_l)]
best_acc   = max(acc_test_l)
best_k_wd  = list(k_range)[np.argmax(acc_test_wd)]
best_acc_wd = max(acc_test_wd)

print(f"  votación uniforme  — mejor K: {best_k}   accuracy: {best_acc:.4f}")
print(f"  votación distancia — mejor K: {best_k_wd}  accuracy: {best_acc_wd:.4f}")

mejor_weights = "distance" if best_acc_wd > best_acc else "uniform"
mejor_k_final = best_k_wd if best_acc_wd > best_acc else best_k
mejor_acc_final = max(best_acc_wd, best_acc)
print(f"  ganador: weights={mejor_weights}  K={mejor_k_final}  acc={mejor_acc_final:.4f}")

print_tabulate(pd.DataFrame({
    "K":              list(k_range),
    "acc_uniform":    [round(a, 4) for a in acc_test_l],
    "acc_distance":   [round(a, 4) for a in acc_test_wd],
    "f1_macro":       [round(a, 4) for a in f1_test_l],
}))

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(list(k_range), acc_train_l,  label="accuracy train",     color="#3498db", linewidth=2)
ax.plot(list(k_range), acc_test_l,   label="accuracy test (uniform)",  color="#e74c3c", linewidth=2)
ax.plot(list(k_range), acc_test_wd,  label="accuracy test (distance)", color="#f39c12", linewidth=2, linestyle="--")
ax.plot(list(k_range), f1_test_l,    label="F1 macro (uniform)",
        color="#9b59b6", linewidth=1.5, linestyle=":")
ax.axvline(mejor_k_final, color="#2ecc71", linewidth=1.5, linestyle="--",
           label=f"mejor K={mejor_k_final} ({mejor_weights})")
ax.set_xlabel("K (número de vecinos)"); ax.set_ylabel("métrica")
ax.set_title("Accuracy y F1-macro vs K — uniform vs distance weights", fontweight="bold")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("img/knn_accuracy_vs_k.png", dpi=130)
plt.close()
print("  grafica guardada: img/knn_accuracy_vs_k.png")


# validé con 5-fold estratificado usando el mejor weights encontrado

print(f"\n{'='*60}")
print("  2. VALIDACION CRUZADA 5-FOLD")
print(f"{'='*60}")
print(f"  weights={mejor_weights}  K={mejor_k_final}")

X_all_s = scaler.fit_transform(X_all)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    KNeighborsClassifier(n_neighbors=mejor_k_final, weights=mejor_weights),
    X_all_s, y_all, cv=cv, scoring="accuracy"
)
print(f"  scores por fold: {[round(s,4) for s in cv_scores]}")
print(f"  media:           {cv_scores.mean():.4f}")
print(f"  desv. estándar:  {cv_scores.std():.4f}")
print(f"  intervalo 95%:   [{cv_scores.mean()-2*cv_scores.std():.4f}, "
      f"{cv_scores.mean()+2*cv_scores.std():.4f}]")


# entrené el modelo final con el mejor K y weights, y calculé todas las métricas

print(f"\n{'='*60}")
print(f"  3. MODELO FINAL  K={mejor_k_final}  weights={mejor_weights}")
print(f"{'='*60}")

scaler2    = MinMaxScaler()
X_train_s2 = scaler2.fit_transform(X_train)
X_test_s2  = scaler2.transform(X_test)

knn_final = KNeighborsClassifier(n_neighbors=mejor_k_final, weights=mejor_weights)
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
fig.suptitle(f"Matriz de confusión — KNN K={mejor_k_final} weights={mejor_weights}", fontweight="bold", fontsize=12)
plt.tight_layout()
plt.savefig("img/knn_confusion_matrix.png", dpi=130)
plt.close()
print("  grafica guardada: img/knn_confusion_matrix.png")

print_tabulate(pd.DataFrame(cm, index=["real H","real D","real A"],
                             columns=["pred H","pred D","pred A"]).reset_index())


# comparé el accuracy de cinco conjuntos distintos de features con ambos tipos de weights

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
    sc    = MinMaxScaler()
    Xtr_s = sc.fit_transform(Xtr)
    Xte_s = sc.transform(Xte)
    bst_a = 0; bst_k = 1; bst_w = "uniform"
    for w in ["uniform", "distance"]:
        for k in range(1, 31):
            knn_t = KNeighborsClassifier(n_neighbors=k, weights=w)
            knn_t.fit(Xtr_s, ytr)
            a = accuracy_score(yte, knn_t.predict(Xte_s))
            if a > bst_a:
                bst_a = a; bst_k = k; bst_w = w
    knn_b = KNeighborsClassifier(n_neighbors=bst_k, weights=bst_w)
    knn_b.fit(Xtr_s, ytr)
    f1_fs = f1_score(yte, knn_b.predict(Xte_s), average="macro")
    fs_results.append({
        "feature_set": nombre,
        "n_features":  len(cols_ok),
        "mejor_k":     bst_k,
        "weights":     bst_w,
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
ax.set_title("Accuracy y F1 por conjunto de features — mejor K y weights por set", fontweight="bold")
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



# analicé el rendimiento del modelo en partidos equilibrados donde el empate es más probable
# un partido equilibrado tiene cuotas H, D y A todas en rango similar

print(f"\n{'='*60}")
print("  12. PARTIDOS EQUILIBRADOS — donde el empate deberia ser predecible")
print(f"{'='*60}")
print("""
  el empate tuvo F1=0.17 en el modelo general porque en el espacio de
  features los empates no tienen zona propia. pero cuando el mercado
  señala equilibrio (cuotas similares para los tres resultados) los
  empates deberían concentrarse más y el modelo debería detectarlos mejor.
""")

mask_eq_test = (
    (X_test["AvgH"].between(2.2, 4.0)) &
    (X_test["AvgA"].between(2.2, 4.0)) &
    (X_test["AvgD"].between(2.8, 3.8))
)
n_eq = mask_eq_test.sum()

if n_eq >= 20:
    acc_eq  = accuracy_score(y_test[mask_eq_test], y_pred[mask_eq_test])
    f1_eq   = f1_score(y_test[mask_eq_test], y_pred[mask_eq_test], average="macro")
    dist_eq = y_test[mask_eq_test].value_counts().to_dict()
    pred_eq = pd.Series(y_pred[mask_eq_test]).value_counts().to_dict()
    cm_eq   = confusion_matrix(y_test[mask_eq_test], y_pred[mask_eq_test], labels=["H","D","A"])

    print(f"  partidos equilibrados en test: {n_eq}")
    print(f"  distribución real:   {dist_eq}")
    print(f"  distribución pred:   {pred_eq}")
    print(f"  accuracy:            {acc_eq:.4f}  (vs {acc_final:.4f} general)")
    print(f"  F1 macro:            {f1_eq:.4f}  (vs {f1_final:.4f} general)")

    print("\n  classification report en partidos equilibrados:")
    print(classification_report(
        y_test[mask_eq_test], y_pred[mask_eq_test],
        target_names=["H","D","A"]
    ))

    print("  matriz de confusión en partidos equilibrados:")
    print_tabulate(pd.DataFrame(
        cm_eq,
        index=["real H","real D","real A"],
        columns=["pred H","pred D","pred A"]
    ).reset_index())

    # grafica comparativa: F1 por clase general vs equilibrados
    report_gen = classification_report(y_test, y_pred, target_names=["H","D","A"],
                                        output_dict=True)
    report_eq  = classification_report(y_test[mask_eq_test], y_pred[mask_eq_test],
                                        target_names=["H","D","A"], output_dict=True)
    clases = ["H","D","A"]
    f1_gen_vals = [report_gen[c]["f1-score"] for c in clases]
    f1_eq_vals  = [report_eq[c]["f1-score"]  for c in clases]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(3); w = 0.35
    ax.bar(x - w/2, f1_gen_vals, w, label="todos los partidos",
           color="#3498db", alpha=0.85)
    ax.bar(x + w/2, f1_eq_vals,  w, label="partidos equilibrados",
           color="#2ecc71", alpha=0.85)
    for i, (vg, ve) in enumerate(zip(f1_gen_vals, f1_eq_vals)):
        ax.text(i - w/2, vg + 0.005, f"{vg:.2f}", ha="center", fontsize=9)
        ax.text(i + w/2, ve + 0.005, f"{ve:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(["H (local)","D (empate)","A (visitante)"])
    ax.set_ylabel("F1-score por clase")
    ax.set_title("F1 por clase: todos los partidos vs partidos equilibrados",
                 fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig("img/knn_equilibrados_f1.png", dpi=130)
    plt.close()
    print("  grafica guardada: img/knn_equilibrados_f1.png")

    if report_eq["D"]["f1-score"] > report_gen["D"]["f1-score"]:
        print(f"\n  el F1 del empate sube de {report_gen['D']['f1-score']:.3f} a "
              f"{report_eq['D']['f1-score']:.3f} en partidos equilibrados")
        print("  confirma que el KNN detecta mejor el empate cuando el mercado ya señala incertidumbre")
    else:
        print(f"\n  el F1 del empate no mejora en partidos equilibrados "
              f"({report_eq['D']['f1-score']:.3f} vs {report_gen['D']['f1-score']:.3f})")
        print("  el problema del empate no se resuelve filtrando por equilibrio de cuotas")
else:
    print(f"  solo {n_eq} partidos equilibrados en test — muestra insuficiente para comparar")


# ajusté el threshold de probabilidad de voto para el empate buscando el punto
# donde el F1 del empate mejora sin destrozar el accuracy general

print(f"\n{'='*60}")
print("  12b. THRESHOLD DEL EMPATE — ajuste de sensibilidad")
print(f"{'='*60}")
print("""
  el KNN con distance vota D solo 130 veces sobre 2371 partidos.
  bajando el threshold de decisión para D — si la probabilidad de empate
  supera X% lo clasificamos como D en lugar de requerir mayoría absoluta —
  podemos mejorar el recall del empate a costa de algo de precision.
""")

probs_test = knn_final.predict_proba(X_test_s2)
cls_order  = list(knn_final.classes_)
idx_D      = cls_order.index("D")
idx_H      = cls_order.index("H")
idx_A      = cls_order.index("A")

# calculé el F1 del empate del modelo base (sin ningún threshold) para usarlo como referencia
report_base = classification_report(y_test, y_pred, target_names=["H","D","A"],
                                     output_dict=True, zero_division=0)
f1_D_base  = round(report_base["D"]["f1-score"], 4)
acc_base   = round(accuracy_score(y_test, y_pred), 4)
n_D_base   = int(sum(y_pred == "D"))

thresholds = [0.20, 0.25, 0.28, 0.30, 0.33, 0.35, 0.38, 0.40]
thr_rows   = []
for thr in thresholds:
    y_thr = []
    for prob_row in probs_test:
        if prob_row[idx_D] >= thr:
            y_thr.append("D")
        else:
            # descarto D — elijo entre H y A directamente
            y_thr.append("H" if prob_row[idx_H] >= prob_row[idx_A] else "A")
    y_thr = np.array(y_thr)
    rep   = classification_report(y_test, y_thr, target_names=["H","D","A"],
                                   output_dict=True, zero_division=0)
    thr_rows.append({
        "threshold_D": thr,
        "acc":         round(accuracy_score(y_test, y_thr), 4),
        "f1_H":        round(rep["H"]["f1-score"], 4),
        "f1_D":        round(rep["D"]["f1-score"], 4),
        "f1_A":        round(rep["A"]["f1-score"], 4),
        "f1_macro":    round(rep["macro avg"]["f1-score"], 4),
        "n_pred_D":    int(sum(y_thr == "D")),
    })

thr_df = pd.DataFrame(thr_rows)

# agrego fila del modelo base al inicio para comparación visual
base_row = pd.DataFrame([{
    "threshold_D": "base (sin thr)",
    "acc":         acc_base,
    "f1_H":        round(report_base["H"]["f1-score"], 4),
    "f1_D":        f1_D_base,
    "f1_A":        round(report_base["A"]["f1-score"], 4),
    "f1_macro":    round(report_base["macro avg"]["f1-score"], 4),
    "n_pred_D":    n_D_base,
}])
print_tabulate(pd.concat([base_row, thr_df], ignore_index=True))

mejor_thr_idx = thr_df["f1_D"].idxmax()
mejor_thr     = thr_df.loc[mejor_thr_idx, "threshold_D"]
print(f"\n  F1 empate sin threshold (modelo base): {f1_D_base:.4f}  (n_pred_D={n_D_base})")
print(f"  threshold que maximiza F1 del empate:   {mejor_thr}")
print(f"  F1 empate con threshold={mejor_thr}:       {thr_df.loc[mejor_thr_idx, 'f1_D']:.4f}")
print(f"  accuracy con ese threshold:              {thr_df.loc[mejor_thr_idx, 'acc']:.4f}  (vs {acc_base:.4f} sin threshold)")
delta_f1 = thr_df.loc[mejor_thr_idx, 'f1_D'] - f1_D_base
delta_acc = thr_df.loc[mejor_thr_idx, 'acc'] - acc_base
print(f"  trade-off: F1 empate {'+' if delta_f1>=0 else ''}{delta_f1:.4f}  "
      f"accuracy {'+' if delta_acc>=0 else ''}{delta_acc:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].axhline(f1_D_base, color="#f39c12", linewidth=1.2, linestyle=":",
                label=f"F1 empate base ({f1_D_base:.3f})")
axes[0].plot(thr_df["threshold_D"], thr_df["f1_D"],  color="#f39c12", linewidth=2,
             marker="o", label="F1 empate con threshold")
axes[0].plot(thr_df["threshold_D"], thr_df["f1_H"],  color="#3498db", linewidth=1.5,
             marker="s", linestyle="--", label="F1 local")
axes[0].plot(thr_df["threshold_D"], thr_df["f1_A"],  color="#e74c3c", linewidth=1.5,
             marker="^", linestyle="--", label="F1 visitante")
axes[0].plot(thr_df["threshold_D"], thr_df["f1_macro"], color="#9b59b6", linewidth=1.5,
             linestyle=":", label="F1 macro")
axes[0].axvline(mejor_thr, color="green", linewidth=1.2, linestyle="--",
                label=f"mejor thr={mejor_thr}")
axes[0].set_xlabel("threshold probabilidad D"); axes[0].set_ylabel("F1-score")
axes[0].set_title("F1 por clase según threshold del empate", fontweight="bold")
axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

axes[1].axhline(acc_base, color="#2ecc71", linewidth=1.2, linestyle=":",
                label=f"accuracy base ({acc_base:.4f})")
axes[1].plot(thr_df["threshold_D"], thr_df["acc"], color="#2ecc71", linewidth=2,
             marker="o", label="accuracy con threshold")
ax2 = axes[1].twinx()
ax2.plot(thr_df["threshold_D"], thr_df["n_pred_D"],
         color="#f39c12", linewidth=1.5, marker="s", linestyle="--")
ax2.axhline(n_D_base, color="#f39c12", linewidth=1, linestyle=":")
ax2.set_ylabel("n predichos como D", color="#f39c12")
ax2.tick_params(axis="y", labelcolor="#f39c12")
axes[1].axvline(mejor_thr, color="green", linewidth=1.2, linestyle="--")
axes[1].set_xlabel("threshold probabilidad D")
axes[1].set_ylabel("accuracy")
axes[1].set_title("Accuracy y partidos predichos como D", fontweight="bold")
axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

fig.suptitle("Ajuste de threshold para mejorar detección del empate",
             fontweight="bold", fontsize=11)
plt.tight_layout()
plt.savefig("img/knn_threshold_empate.png", dpi=130)
plt.close()
print("  grafica guardada: img/knn_threshold_empate.png")


# desglosé el accuracy por temporada para ver si el modelo es más estable en algunos años

print(f"\n{'='*60}")
print("  13. ACCURACY POR TEMPORADA")
print(f"{'='*60}")

SEASON_MAP = {1920:"2019/20",2021:"2020/21",2122:"2021/22",
              2223:"2022/23",2324:"2023/24",2425:"2024/25",2526:"2025/26"}
test_season = data_main.loc[idx_test, "Season"] if "Season" in data_main.columns else None

if test_season is not None:
    seasons_sorted = sorted(data_main["Season"].unique())
    seas_acc = []
    for s in seasons_sorted:
        mask_s = test_season == s
        if mask_s.sum() < 20: continue
        acc_s = accuracy_score(y_test[mask_s], y_pred[mask_s])
        f1_s  = f1_score(y_test[mask_s], y_pred[mask_s], average="macro")
        dist_s = y_test[mask_s].value_counts().to_dict()
        seas_acc.append({
            "temporada": SEASON_MAP.get(s, str(s)),
            "n_test":    int(mask_s.sum()),
            "accuracy":  round(acc_s, 4),
            "f1_macro":  round(f1_s, 4),
            "pct_H":     round(dist_s.get("H",0)/mask_s.sum()*100, 1),
            "pct_D":     round(dist_s.get("D",0)/mask_s.sum()*100, 1),
            "pct_A":     round(dist_s.get("A",0)/mask_s.sum()*100, 1),
        })
    print_tabulate(pd.DataFrame(seas_acc))

    fig, ax = plt.subplots(figsize=(11, 5))
    sa_df = pd.DataFrame(seas_acc)
    ax.plot(sa_df["temporada"], sa_df["accuracy"],
            color="#3498db", linewidth=2, marker="o", label="accuracy")
    ax.plot(sa_df["temporada"], sa_df["f1_macro"],
            color="#9b59b6", linewidth=1.5, marker="s", linestyle="--", label="F1 macro")
    ax.axhline(1/3, color="red", linewidth=1, linestyle=":", label="baseline azar")
    ax.axhline(acc_final, color="#2ecc71", linewidth=1, linestyle="--",
               label=f"accuracy global ({acc_final:.4f})")
    ax.set_ylabel("métrica")
    ax.set_title("Accuracy y F1 por temporada", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig("img/knn_accuracy_por_temporada.png", dpi=130)
    plt.close()
    print("  grafica guardada: img/knn_accuracy_por_temporada.png")
else:
    print("  columna Season no disponible en data_main")


# imprimeción modelos

print(f"\n{'='*60}")
print("  NUMEROS FINALES")
print(f"{'='*60}")
print(f"  KNN manual:   K={best_k_manual}  acc={acc_final_manual:.4f}  features=2  muestra={MUESTRA_N:,}")
print(f"  KNN sklearn:  K={mejor_k_final}  weights={mejor_weights}  acc={acc_final:.4f}  features={len(MAIN_FEATURES)}  partidos={len(data_main):,}")
print(f"  F1 macro:     {f1_final:.4f}")
print(f"  CV 5-fold:    {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"  mejor set:    {fs_df.iloc[0]['feature_set']} (acc={fs_df.iloc[0]['accuracy']:.4f})")

print("\nlisto — todas las graficas guardadas en img/")