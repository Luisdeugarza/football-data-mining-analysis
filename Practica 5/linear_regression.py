import sys
sys.stdout.reconfigure(encoding='utf-8')

import warnings
warnings.filterwarnings("ignore", message="divide by zero encountered in scalar divide")
warnings.filterwarnings("ignore", message="omni_normtest is not valid")
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os
from tabulate import tabulate

os.makedirs("img", exist_ok=True)

LIGAS_NAME = {
    "E0": "Premier League",
    "SP1": "La Liga",
    "D1":  "Bundesliga",
    "I1":  "Serie A",
    "F1":  "Ligue 1",
}
LIGA_COLORS = {
    "E0":  "#3498db",
    "SP1": "#e74c3c",
    "D1":  "#f39c12",
    "I1":  "#2ecc71",
    "F1":  "#9b59b6",
}
SEASON_MAP = {
    1920: "2019/20", 2021: "2020/21", 2122: "2021/22",
    2223: "2022/23", 2324: "2023/24", 2425: "2024/25", 2526: "2025/26",
}

def print_tabulate(df: pd.DataFrame):
    print(tabulate(df, headers=df.columns, tablefmt="orgtbl"))


# cargué el dataset y calculé las variables derivadas necesarias para los modelos

df = pd.read_csv("../Practica 1/data/clean/football_clean.csv", parse_dates=["Date"])

df["total_goals"]       = df["FTHG"] + df["FTAG"]
df["ht_goals"]          = df["HTHG"] + df["HTAG"]
df["second_half_goals"] = df["total_goals"] - df["ht_goals"]
df["home_win"]          = (df["FTR"] == "H").astype(int)
df["draw"]              = (df["FTR"] == "D").astype(int)
df["away_win"]          = (df["FTR"] == "A").astype(int)
df["btts"]              = ((df["FTHG"] > 0) & (df["FTAG"] > 0)).astype(int)
df["over25"]            = (df["total_goals"] > 2).astype(int)
df["imp_prob_H"]        = round(1 / df["AvgH"], 4)
df["imp_prob_D"]        = round(1 / df["AvgD"], 4)
df["imp_prob_A"]        = round(1 / df["AvgA"], 4)
df["overround"]         = round(df["imp_prob_H"] + df["imp_prob_D"] + df["imp_prob_A"], 4)
df["odds_move_H"]       = round(df["AvgCH"] - df["AvgH"], 4)
df["odds_move_A"]       = round(df["AvgCA"] - df["AvgA"], 4)
df["odds_move_D"]       = round(df["AvgCD"] - df["AvgD"], 4)
df["diff_imp"]          = df["imp_prob_H"] - df["imp_prob_A"]
df["Season_label"]      = df["Season"].map(SEASON_MAP)

ligas   = sorted(df["Div"].unique())
seasons = sorted(df["Season"].unique())

print(f"dataset cargado: {len(df):,} partidos | {len(ligas)} ligas | {len(seasons)} temporadas")


# definí las funciones de regresión, VIF y diagnóstico de residuales

def linear_regression(df_in: pd.DataFrame, x: str, y: str,
                       title: str = "", save_path: str = None,
                       color: str = "#3498db") -> dict:
    datos = df_in[[x, y]].dropna()
    X = sm.add_constant(datos[x].values)
    Y = datos[y].values

    model = sm.OLS(Y, X).fit()
    print(f"\n{'='*60}")
    print(f"  {title or f'{y} ~ {x}'}")
    print(f"{'='*60}")
    print(model.summary())

    const = model.params[0]
    slope = model.params[1]

    if save_path:
        fig, ax = plt.subplots(figsize=(9, 6))
        datos.plot(x=x, y=y, kind="scatter", ax=ax,
                   color=color, alpha=0.3, s=8)
        x_line = np.linspace(datos[x].min(), datos[x].max(), 300)
        ax.plot(x_line, slope * x_line + const, color="red", linewidth=1.8,
                label=f"R²={model.rsquared:.3f}")
        ax.set_title(title or f"{y} ~ {x}", fontsize=11, fontweight="bold")
        ax.set_xlabel(x); ax.set_ylabel(y)
        ax.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=130)
        plt.close()
        print(f"  grafica guardada: {save_path}")

    return {
        "x": x, "y": y,
        "r2":      round(model.rsquared, 4),
        "r2_adj":  round(model.rsquared_adj, 4),
        "coef":    round(slope, 6),
        "const":   round(const, 6),
        "pval":    round(model.pvalues[1], 6),
        "n":       int(model.nobs),
        "model":   model,
    }


def linear_regression_multiple(df_in: pd.DataFrame, features: list, y: str,
                                 title: str = "") -> dict:
    cols = features + [y]
    datos = df_in[cols].dropna()
    X = sm.add_constant(datos[features])
    Y = datos[y]

    model = sm.OLS(Y, X).fit()
    print(f"\n{'='*60}")
    print(f"  {title or f'{y} ~ ' + ' + '.join(features)}")
    print(f"{'='*60}")
    print(model.summary())

    return {
        "features": features, "y": y,
        "r2":      round(model.rsquared, 4),
        "r2_adj":  round(model.rsquared_adj, 4),
        "n":       int(model.nobs),
        "aic":     round(model.aic, 2),
        "bic":     round(model.bic, 2),
        "model":   model,
        "X":       X,
        "Y":       Y,
    }


def calc_vif(df_in: pd.DataFrame, features: list) -> pd.DataFrame:
    datos = df_in[features].dropna()
    X = sm.add_constant(datos)
    rows = []
    for i, col in enumerate(X.columns):
        if col == "const":
            continue
        vif = variance_inflation_factor(X.values, i)
        rows.append({"variable": col, "VIF": round(vif, 3)})
    return pd.DataFrame(rows).sort_values("VIF", ascending=False).reset_index(drop=True)


def plot_residuals(model, title: str, save_path: str):
    fitted    = model.fittedvalues
    residuals = model.resid

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].scatter(fitted, residuals, alpha=0.2, s=6, color="#3498db")
    axes[0].axhline(0, color="red", linewidth=1.2, linestyle="--")
    axes[0].set_xlabel("valores ajustados")
    axes[0].set_ylabel("residuales")
    axes[0].set_title("Residuales vs ajustados", fontweight="bold")

    sm.qqplot(residuals, line="s", ax=axes[1], alpha=0.3, markersize=3)
    axes[1].set_title("QQ plot de residuales", fontweight="bold")

    fig.suptitle(f"Diagnostico de residuales — {title}", fontweight="bold", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close()
    print(f"  grafica guardada: {save_path}")


# calculé la matriz de correlación entre todas las variables numéricas clave

print(f"\n{'='*60}")
print("  1. MATRIZ DE CORRELACION")
print(f"{'='*60}")

corr_cols = [
    "total_goals", "ht_goals", "second_half_goals",
    "home_win", "draw", "away_win",
    "AvgH", "AvgA", "AvgD",
    "AvgCH", "AvgCA",
    "imp_prob_H", "imp_prob_A", "imp_prob_D",
    "overround",
    "odds_move_H", "odds_move_A", "odds_move_D",
    "btts", "over25",
]

corr_mat = df[corr_cols].corr().round(3)
print("\n  correlaciones con total_goals (ordenadas por valor absoluto):")
print_tabulate(
    corr_mat["total_goals"]
    .drop("total_goals")
    .sort_values(key=abs, ascending=False)
    .reset_index()
    .rename(columns={"index": "variable", "total_goals": "corr_con_total_goals"})
)

print("\n  correlaciones con home_win (ordenadas por valor absoluto):")
print_tabulate(
    corr_mat["home_win"]
    .drop("home_win")
    .sort_values(key=abs, ascending=False)
    .reset_index()
    .rename(columns={"index": "variable", "home_win": "corr_con_home_win"})
)

fig, ax = plt.subplots(figsize=(14, 12))
im = ax.imshow(corr_mat.values, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr_cols)))
ax.set_xticklabels(corr_cols, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(corr_cols)))
ax.set_yticklabels(corr_cols, fontsize=8)
for r in range(len(corr_cols)):
    for c in range(len(corr_cols)):
        val = corr_mat.values[r, c]
        ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                fontsize=6, color="white" if abs(val) > 0.5 else "black")
plt.colorbar(im, ax=ax)
ax.set_title("Heatmap de correlaciones — variables numericas principales",
             fontweight="bold", fontsize=11)
plt.tight_layout()
plt.savefig("img/lr_heatmap_correlaciones.png", dpi=130)
plt.close()
print("\n  grafica guardada: img/lr_heatmap_correlaciones.png")


# ajusté los modelos de regresión simple para cada par de variables

print(f"\n{'='*60}")
print("  2. REGRESIONES SIMPLES")
print(f"{'='*60}")

resultados_simples = []

r = linear_regression(
    df, x="ht_goals", y="total_goals",
    title="total_goals ~ ht_goals",
    save_path="img/lr_total_goals_ht_goals.png",
    color="#3498db"
)
resultados_simples.append(r)

r = linear_regression(
    df, x="AvgH", y="home_win",
    title="home_win ~ AvgH (apertura)",
    save_path="img/lr_home_win_AvgH.png",
    color="#e74c3c"
)
resultados_simples.append(r)

r = linear_regression(
    df, x="AvgCH", y="home_win",
    title="home_win ~ AvgCH (cierre)",
    save_path="img/lr_home_win_AvgCH.png",
    color="#e67e22"
)
resultados_simples.append(r)

r = linear_regression(
    df, x="overround", y="total_goals",
    title="total_goals ~ overround",
    save_path="img/lr_total_goals_overround.png",
    color="#9b59b6"
)
resultados_simples.append(r)

r = linear_regression(
    df, x="imp_prob_H", y="home_win",
    title="home_win ~ imp_prob_H",
    save_path="img/lr_home_win_imp_prob_H.png",
    color="#2ecc71"
)
resultados_simples.append(r)

r = linear_regression(
    df, x="diff_imp", y="total_goals",
    title="total_goals ~ diff_imp (imp_H - imp_A)",
    save_path="img/lr_total_goals_diff_imp.png",
    color="#f39c12"
)
resultados_simples.append(r)

print(f"\n{'='*60}")
print("  resumen regresiones simples")
print(f"{'='*60}")
print_tabulate(pd.DataFrame(resultados_simples)[["x","y","n","r2","r2_adj","coef","pval"]])


# ajusté los modelos de regresión múltiple y calculé el VIF para detectar multicolinealidad

print(f"\n{'='*60}")
print("  3. REGRESION MULTIPLE — total_goals")
print(f"{'='*60}")

resultados_multiple = []

features_goles = ["imp_prob_H", "imp_prob_A", "overround"]
r_mult1 = linear_regression_multiple(
    df, features=features_goles, y="total_goals",
    title="total_goals ~ imp_prob_H + imp_prob_A + overround"
)
resultados_multiple.append(r_mult1)

# este modelo tuvo el mejor R² (0.053) pero disparó warning de multicolinealidad
features_odds = ["AvgH", "AvgA", "AvgD", "overround"]
r_mult2 = linear_regression_multiple(
    df, features=features_odds, y="total_goals",
    title="total_goals ~ AvgH + AvgA + AvgD + overround"
)
resultados_multiple.append(r_mult2)

print(f"\n{'='*60}")
print("  VIF — variance inflation factor")
print(f"{'='*60}")
print("  VIF > 10 indica multicolinealidad problematica entre features")
print("\n  modelo probabilidades implicitas:")
print_tabulate(calc_vif(df, features_goles))
print("\n  modelo cuotas directas (el que disparo condition number 1680):")
print_tabulate(calc_vif(df, features_odds))

print("\n  resumen regresiones multiples:")
print_tabulate(pd.DataFrame(resultados_multiple)[["features","y","n","r2","r2_adj","aic","bic"]])

print("\n  diagnostico de residuales — mejor modelo multiple:")
plot_residuals(
    r_mult2["model"],
    title="AvgH + AvgA + AvgD + overround -> total_goals",
    save_path="img/lr_residuales_modelo_goles.png"
)


# probé el movimiento de cuota como predictor de resultado y de goles

print(f"\n{'='*60}")
print("  4. MOVIMIENTO DE CUOTA COMO PREDICTOR")
print(f"{'='*60}")

r_mov_hw = linear_regression(
    df, x="odds_move_H", y="home_win",
    title="home_win ~ odds_move_H",
    save_path="img/lr_home_win_odds_move_H.png",
    color="#3498db"
)
resultados_simples.append(r_mov_hw)

# modelo descartado: odds_move_H -> total_goals salió no significativo (p=0.590)
print(f"\n{'='*60}")
print("  odds_move_H -> total_goals: modelo NO SIGNIFICATIVO — descartado")
print(f"{'='*60}")
print("  p-value del F-statistic = 0.590, R²=0.000")
print("  el movimiento de cuota del local no predice cuantos goles habra")
print("  el smart money apunta al resultado del partido, no al numero de goles")

r_moves = linear_regression_multiple(
    df,
    features=["odds_move_H", "odds_move_A", "odds_move_D"],
    y="total_goals",
    title="total_goals ~ odds_move_H + odds_move_A + odds_move_D"
)
resultados_multiple.append(r_moves)

r_mov_multi = linear_regression_multiple(
    df,
    features=["odds_move_H", "odds_move_A"],
    y="home_win",
    title="home_win ~ odds_move_H + odds_move_A"
)
resultados_multiple.append(r_mov_multi)


# modelé el edge del mercado como variable dependiente para detectar sesgos sistemáticos

print(f"\n{'='*60}")
print("  5. EDGE COMO VARIABLE DEPENDIENTE")
print(f"{'='*60}")
print("\n  el edge es el error sistematico del mercado: real_% - implied_%")
print("  R²=0.475 en el modelo local significa que casi la mitad de la variacion")
print("  del error del mercado se explica solo con el rango de cuota")
print("  no es ruido — es un sesgo con estructura: el mercado sobrevalora favoritos")
print("  grandes y subestima rangos intermedios de forma predecible")

bins   = [1.0, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 6.0, 25.0]
labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)]
d_edge = df.copy()
d_edge["rng_H"] = pd.cut(d_edge["AvgH"], bins=bins, labels=labels, right=False)

edge_rows = []
for rng in labels:
    sub = d_edge[d_edge["rng_H"] == rng]
    if len(sub) < 10:
        continue
    mid  = (bins[labels.index(rng)] + bins[labels.index(rng) + 1]) / 2
    imp  = 1 / mid * 100
    real = sub["home_win"].mean() * 100
    edge_rows.append({
        "rango_AvgH": rng,
        "mid_cuota":  round(mid, 3),
        "n":          len(sub),
        "imp_%":      round(imp, 2),
        "real_%":     round(real, 2),
        "edge":       round(real - imp, 2),
    })

df_edge = pd.DataFrame(edge_rows)
print("\n  edge por rango de cuota local:")
print_tabulate(df_edge)

r_edge = linear_regression(
    df_edge, x="mid_cuota", y="edge",
    title="edge ~ mid_cuota (sesgo del mercado por rango de cuota local)",
    save_path="img/lr_edge_cuota.png",
    color="#9b59b6"
)
print(f"\n  R²={r_edge['r2']} — la cuota explica {r_edge['r2']*100:.1f}% de la variacion del error")
print(f"  coef={r_edge['coef']:.4f}: a mayor cuota el edge cambia sistematicamente")

d_edge["rng_A"] = pd.cut(d_edge["AvgA"], bins=bins, labels=labels, right=False)
edge_rows_a = []
for rng in labels:
    sub = d_edge[d_edge["rng_A"] == rng]
    if len(sub) < 10:
        continue
    mid  = (bins[labels.index(rng)] + bins[labels.index(rng) + 1]) / 2
    imp  = 1 / mid * 100
    real = sub["away_win"].mean() * 100
    edge_rows_a.append({
        "rango_AvgA": rng,
        "mid_cuota":  round(mid, 3),
        "n":          len(sub),
        "imp_%":      round(imp, 2),
        "real_%":     round(real, 2),
        "edge":       round(real - imp, 2),
    })
df_edge_a = pd.DataFrame(edge_rows_a)
print("\n  edge por rango de cuota visitante:")
print_tabulate(df_edge_a)
r_edge_a = linear_regression(
    df_edge_a, x="mid_cuota", y="edge",
    title="edge_visitante ~ mid_cuota",
    save_path="img/lr_edge_cuota_visitante.png",
    color="#e74c3c"
)


# comparé apertura vs cierre como predictores de victoria local

print(f"\n{'='*60}")
print("  6. APERTURA VS CIERRE: CUAL PREDICE MEJOR LA VICTORIA LOCAL")
print(f"{'='*60}")

r_ap = linear_regression(
    df, x="AvgH", y="home_win",
    title="home_win ~ AvgH apertura",
    save_path="img/lr_apertura_vs_cierre_H.png",
    color="#3498db"
)
r_ci = linear_regression(
    df, x="AvgCH", y="home_win",
    title="home_win ~ AvgCH cierre",
    save_path="img/lr_apertura_vs_cierre_CH.png",
    color="#e67e22"
)

print("\n  comparacion apertura vs cierre:")
print_tabulate(pd.DataFrame([
    {"tipo": "apertura (AvgH)",  "r2": r_ap["r2"], "r2_adj": r_ap["r2_adj"],
     "coef": r_ap["coef"], "pval": r_ap["pval"]},
    {"tipo": "cierre  (AvgCH)", "r2": r_ci["r2"], "r2_adj": r_ci["r2_adj"],
     "coef": r_ci["coef"], "pval": r_ci["pval"]},
]))
mejor = "cierre" if r_ci["r2"] > r_ap["r2"] else "apertura"
print(f"\n  mejor predictor: {mejor} (R² {max(r_ap['r2'],r_ci['r2']):.4f} vs {min(r_ap['r2'],r_ci['r2']):.4f})")
print("  la diferencia es minima — el movimiento de cuota no agrega poder predictivo")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (x_col, titulo, color) in zip(axes, [
    ("AvgH",  f"apertura  R²={r_ap['r2']:.3f}", "#3498db"),
    ("AvgCH", f"cierre    R²={r_ci['r2']:.3f}", "#e67e22"),
]):
    datos = df[[x_col, "home_win"]].dropna()
    datos.plot(x=x_col, y="home_win", kind="scatter",
               ax=ax, color=color, alpha=0.15, s=6)
    m = sm.OLS(datos["home_win"].values,
               sm.add_constant(datos[x_col].values)).fit()
    x_line = np.linspace(datos[x_col].min(), datos[x_col].max(), 200)
    ax.plot(x_line, m.params[1] * x_line + m.params[0], color="red", linewidth=1.8)
    ax.set_title(titulo, fontweight="bold")
    ax.set_xlabel(x_col); ax.set_ylabel("home_win")
fig.suptitle("Cuota apertura vs cierre como predictor de victoria local",
             fontweight="bold", fontsize=12)
plt.tight_layout()
plt.savefig("img/lr_ap_vs_ci_comparativa.png", dpi=130)
plt.close()
print("  grafica guardada: img/lr_ap_vs_ci_comparativa.png")


# repliqué el modelo por cada liga para ver si la relación varía entre competiciones

print(f"\n{'='*60}")
print("  7. REGRESION POR LIGA  (total_goals ~ imp_prob_H + imp_prob_A + overround)")
print(f"{'='*60}")

liga_results = []
for liga in ligas:
    sub   = df[df["Div"] == liga]
    cols  = ["imp_prob_H", "imp_prob_A", "overround", "total_goals"]
    datos = sub[cols].dropna()
    X     = sm.add_constant(datos[["imp_prob_H", "imp_prob_A", "overround"]])
    model = sm.OLS(datos["total_goals"], X).fit()
    liga_results.append({
        "liga":       LIGAS_NAME[liga],
        "n":          int(model.nobs),
        "r2":         round(model.rsquared, 4),
        "r2_adj":     round(model.rsquared_adj, 4),
        "coef_imp_H": round(model.params.get("imp_prob_H", np.nan), 4),
        "coef_imp_A": round(model.params.get("imp_prob_A", np.nan), 4),
        "coef_over":  round(model.params.get("overround", np.nan), 4),
        "pval_F":     round(model.f_pvalue, 6),
    })

print_tabulate(pd.DataFrame(liga_results))

fig, ax = plt.subplots(figsize=(9, 5))
liga_df = pd.DataFrame(liga_results)
bars = ax.bar(liga_df["liga"], liga_df["r2"],
              color=[LIGA_COLORS[l] for l in ligas], alpha=0.85)
for bar, val in zip(bars, liga_df["r2"]):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.001,
            f"{val:.3f}", ha="center", fontsize=9)
ax.set_ylabel("R²")
ax.set_title("R² del modelo de goles por liga\n(total_goals ~ imp_prob_H + imp_prob_A + overround)",
             fontweight="bold")
ax.set_ylim(0, max(liga_df["r2"]) * 1.2)
plt.tight_layout()
plt.savefig("img/lr_r2_por_liga.png", dpi=130)
plt.close()
print("  grafica guardada: img/lr_r2_por_liga.png")

fig, axes = plt.subplots(1, 5, figsize=(22, 5), sharey=True)
for i, liga in enumerate(ligas):
    sub = df[df["Div"] == liga][["AvgH", "home_win"]].dropna()
    axes[i].scatter(sub["AvgH"], sub["home_win"],
                    alpha=0.15, s=6, color=LIGA_COLORS[liga])
    m = sm.OLS(sub["home_win"].values,
               sm.add_constant(sub["AvgH"].values)).fit()
    x_line = np.linspace(sub["AvgH"].min(), sub["AvgH"].max(), 200)
    axes[i].plot(x_line, m.params[1] * x_line + m.params[0],
                 color="red", linewidth=1.8)
    axes[i].set_title(f"{LIGAS_NAME[liga]}\nR²={m.rsquared:.3f}",
                      fontsize=9, fontweight="bold")
    axes[i].set_xlabel("AvgH")
    if i == 0:
        axes[i].set_ylabel("home_win")
fig.suptitle("home_win ~ AvgH por liga", fontweight="bold", fontsize=12)
plt.tight_layout()
plt.savefig("img/lr_home_win_por_liga.png", dpi=130)
plt.close()
print("  grafica guardada: img/lr_home_win_por_liga.png")


# modelé la tendencia temporal de goles agrupando por fecha y usando el índice como X

print(f"\n{'='*60}")
print("  8. SERIE TEMPORAL — promedio de goles por fecha")
print(f"{'='*60}")

df_ts = df.groupby("Date")["total_goals"].mean().reset_index()
df_ts.columns = ["Fecha", "avg_goals"]
df_ts = df_ts.sort_values("Fecha").reset_index(drop=True)
df_ts["t"] = range(len(df_ts))

print(f"\n  {len(df_ts)} fechas | desde {df_ts['Fecha'].min().date()} hasta {df_ts['Fecha'].max().date()}")
print_tabulate(df_ts.head(10))

X_ts    = sm.add_constant(df_ts["t"])
model_ts = sm.OLS(df_ts["avg_goals"], X_ts).fit()
print(model_ts.summary())

const_ts = model_ts.params["const"]
slope_ts = model_ts.params["t"]
print(f"\n  pendiente: {slope_ts:.6f} goles por jornada")
print(f"  R²: {model_ts.rsquared:.4f}  p-value: {model_ts.pvalues['t']:.3f}")
print("  no hay tendencia lineal significativa — el futbol europeo no cambio")
print("  de forma lineal en numero de goles entre 2019 y 2026")

fig, ax = plt.subplots(figsize=(14, 5))
ax.scatter(df_ts["Fecha"], df_ts["avg_goals"],
           alpha=0.4, s=10, color="#3498db", label="avg goles por fecha")
ax.plot(df_ts["Fecha"],
        slope_ts * df_ts["t"].values + const_ts,
        color="red", linewidth=2,
        label=f"tendencia lineal  R²={model_ts.rsquared:.4f}  p={model_ts.pvalues['t']:.3f}")
ax.set_xlabel("Fecha"); ax.set_ylabel("avg goles por partido")
ax.set_title("Serie temporal: promedio de goles por fecha con regresion lineal",
             fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig("img/lr_serie_temporal_goles.png", dpi=130)
plt.close()
print("  grafica guardada: img/lr_serie_temporal_goles.png")

fig, ax = plt.subplots(figsize=(13, 6))
for liga in ligas:
    sub = df[df["Div"] == liga].groupby("Date")["total_goals"].mean().reset_index()
    sub = sub.sort_values("Date").reset_index(drop=True)
    sub["t"] = range(len(sub))
    m = sm.OLS(sub["total_goals"].values,
               sm.add_constant(sub["t"].values)).fit()
    ax.plot(sub["Date"], m.params[1] * sub["t"] + m.params[0],
            label=f"{LIGAS_NAME[liga]}  R²={m.rsquared:.3f}",
            color=LIGA_COLORS[liga], linewidth=2)
ax.set_xlabel("Fecha"); ax.set_ylabel("avg goles (tendencia)")
ax.set_title("Tendencia temporal de goles por liga", fontweight="bold")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("img/lr_tendencia_goles_por_liga.png", dpi=130)
plt.close()
print("  grafica guardada: img/lr_tendencia_goles_por_liga.png")


# validé el efecto de la racha previa del local sobre el resultado con regresión lineal

print(f"\n{'='*60}")
print("  9. RACHAS — VALIDACION CON REGRESION LINEAL")
print(f"{'='*60}")
print("""
  hipotesis: la racha previa del equipo local predice el resultado
  y el fenomeno racha mala -> goleada -> siguiente partido pierde
  tiene evidencia estadistica.

  variables construidas:
    racha_previa_local : entero secuencial (positivo=victorias, negativo=derrotas)
    post_goleada_mala  : 1 si el equipo local venia de racha <= -2 y marco 4+ goles
""")

# construí la racha previa del local partido a partido recorriendo el dataset en orden cronológico
df2 = df.sort_values("Date").copy()
streak_map = {}
home_streak_list = []
for _, row in df2.iterrows():
    ht = row["HomeTeam"]
    at = row["AwayTeam"]
    streak_map.setdefault(ht, 0)
    streak_map.setdefault(at, 0)
    home_streak_list.append(streak_map[ht])
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

df2["racha_previa"] = home_streak_list

r_racha = linear_regression(
    df2, x="racha_previa", y="home_win",
    title="home_win ~ racha_previa del local",
    save_path="img/lr_home_win_racha.png",
    color="#3498db"
)
print(f"\n  racha_previa -> home_win:")
print(f"  R²={r_racha['r2']}  coef={r_racha['coef']}  p={r_racha['pval']}")
if r_racha["pval"] < 0.05:
    dir_r = "positivo" if r_racha["coef"] > 0 else "negativo"
    print(f"  SIGNIFICATIVO — coeficiente {dir_r}: la racha previa predice victoria local")

r_racha_g = linear_regression(
    df2, x="racha_previa", y="total_goals",
    title="total_goals ~ racha_previa del local",
    save_path="img/lr_total_goals_racha.png",
    color="#e74c3c"
)
print(f"\n  racha_previa -> total_goals:")
print(f"  R²={r_racha_g['r2']}  coef={r_racha_g['coef']}  p={r_racha_g['pval']}")

# fenomeno: racha mala + goleada -> siguiente partido
print(f"\n{'='*60}")
print("  9b. FENOMENO: racha mala -> goleada -> siguiente partido pierde")
print(f"{'='*60}")

# estado por equipo: racha actual y si el partido anterior fue una goleada
# tras racha mala (racha <= -2 antes de ese partido) ganando con 4+ goles
# la estructura guarda: racha_actual, goles_marcados_ultimo, racha_antes_del_ultimo, gano_ultimo
team_state = {}  # {equipo: {"racha": int, "goles_marcados": int, "racha_previa": int, "gano": bool}}

def get_state(team):
    if team not in team_state:
        team_state[team] = {"racha": 0, "goles_marcados": 0, "racha_previa": 0, "gano": False}
    return team_state[team]

post_goleada_flags = []
df2 = df2.sort_values("Date").copy()

for _, row in df2.iterrows():
    ht = row["HomeTeam"]
    at = row["AwayTeam"]
    st_ht = get_state(ht)
    st_at = get_state(at)

    # activé el flag si en el partido ANTERIOR el local:
    #   1) tenía racha <= -2 ANTES de ese partido (racha_previa)
    #   2) ganó ese partido (gano=True)
    #   3) marcó 4+ goles en ese partido (goles_marcados)
    es_post_goleada = int(
        st_ht["racha_previa"] <= -2 and
        st_ht["gano"] and
        st_ht["goles_marcados"] >= 4
    )
    post_goleada_flags.append(es_post_goleada)

    # actualicé el estado del local para el próximo partido
    racha_actual_ht = st_ht["racha"]
    gano_ht = row["FTR"] == "H"
    team_state[ht] = {
        "racha_previa":    racha_actual_ht,
        "goles_marcados":  int(row["FTHG"]),
        "gano":            gano_ht,
        "racha": (max(racha_actual_ht, 0) + 1) if gano_ht
                 else (min(racha_actual_ht, 0) - 1) if row["FTR"] == "A"
                 else 0,
    }

    # actualicé el estado del visitante
    racha_actual_at = st_at["racha"]
    gano_at = row["FTR"] == "A"
    team_state[at] = {
        "racha_previa":   racha_actual_at,
        "goles_marcados": int(row["FTAG"]),
        "gano":           gano_at,
        "racha": (max(racha_actual_at, 0) + 1) if gano_at
                 else (min(racha_actual_at, 0) - 1) if row["FTR"] == "H"
                 else 0,
    }

df2["post_goleada_mala"] = post_goleada_flags

n_eventos = int(df2["post_goleada_mala"].sum())
print(f"\n  partidos donde el local venia de racha<=-2 y marco 4+ el partido anterior: {n_eventos}")
if n_eventos > 0:
    pct_pierde_post = df2[df2["post_goleada_mala"]==1]["away_win"].mean() * 100
    pct_pierde_resto = df2[df2["post_goleada_mala"]==0]["away_win"].mean() * 100
    print(f"  % que pierde hoy (post goleada):  {pct_pierde_post:.1f}%")
    print(f"  % que pierde en el resto:         {pct_pierde_resto:.1f}%")
    print(f"  diferencia: {pct_pierde_post - pct_pierde_resto:.1f} puntos porcentuales")
else:
    print("  sin eventos detectados con estos filtros")

r_pg = linear_regression(
    df2, x="post_goleada_mala", y="home_win",
    title="home_win ~ post_goleada_mala (racha mala + goleada previa)",
    save_path="img/lr_home_win_post_goleada.png",
    color="#9b59b6"
)
print(f"\n  post_goleada_mala -> home_win:")
print(f"  R²={r_pg['r2']}  coef={r_pg['coef']}  p={r_pg['pval']}")
if r_pg["pval"] < 0.05 and r_pg["coef"] < 0:
    print("  SIGNIFICATIVO y coeficiente NEGATIVO — confirma el fenomeno:")
    print("  cuando el local venia de racha mala y metio goleada, gana menos en el siguiente")
elif r_pg["pval"] < 0.05:
    print("  SIGNIFICATIVO pero coeficiente positivo — el fenomeno no se confirma")
else:
    print("  no significativo — el fenomeno no tiene evidencia de relacion lineal fuerte")
    print("  puede ser util descriptivamente pero el R² sera bajo por ser variable binaria rara")

# victoria local por categoria de racha
racha_cats = [
    ("3+vic",  df2["racha_previa"] >= 3,  3),
    ("2vic",   df2["racha_previa"] == 2,  2),
    ("1vic",   df2["racha_previa"] == 1,  1),
    ("neutro", df2["racha_previa"] == 0,  0),
    ("1der",   df2["racha_previa"] == -1, -1),
    ("2der",   df2["racha_previa"] == -2, -2),
    ("3+der",  df2["racha_previa"] <= -3, -3),
]
racha_rows = []
for lbl, mask, num in racha_cats:
    sub = df2[mask]
    if len(sub) < 10:
        continue
    racha_rows.append({
        "racha_cat":  lbl,
        "racha_num":  num,
        "n":          len(sub),
        "pct_H":      round(sub["home_win"].mean()    * 100, 2),
        "pct_D":      round(sub["draw"].mean()        * 100, 2),
        "pct_A":      round(sub["away_win"].mean()    * 100, 2),
        "avg_goles":  round(sub["total_goals"].mean(), 3),
    })
print("\n  resultado por categoria de racha previa del local:")
print_tabulate(pd.DataFrame(racha_rows).drop(columns="racha_num"))

df_racha_plot = pd.DataFrame(racha_rows)
r_racha_cat = linear_regression(
    df_racha_plot, x="racha_num", y="pct_H",
    title="pct_H ~ racha_num (por categoria de racha previa)",
    save_path="img/lr_pct_H_racha_continua.png",
    color="#3498db"
)
print(f"\n  racha_num -> pct_H (por categoria agregada):")
print(f"  R²={r_racha_cat['r2']}  coef={r_racha_cat['coef']}  p={r_racha_cat['pval']}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(df_racha_plot["racha_cat"], df_racha_plot["pct_H"],
       color=["#e74c3c" if x < 0 else "#95a5a6" if x == 0 else "#3498db"
              for x in df_racha_plot["racha_num"]], alpha=0.85)
ax.axhline(df2["home_win"].mean() * 100, color="black", linewidth=1.2,
           linestyle="--", label=f"promedio global ({df2['home_win'].mean()*100:.1f}%)")
ax.set_ylabel("% victoria local")
ax.set_title("% victoria local segun racha previa del equipo local\n(rojo=racha mala, azul=racha buena)",
             fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig("img/lr_pct_H_por_racha.png", dpi=130)
plt.close()
print("  grafica guardada: img/lr_pct_H_por_racha.png")


# construí el índice de irregularidad y la desviación estándar de goles para cada equipo local

print(f"\n{'='*60}")
print("  9c. IRREGULARIDAD — equipos que alternan goleadas y derrotas")
print(f"{'='*60}")
print("""
  el fenomeno observado en Tigres UANL no es una racha mala pura,
  sino irregularidad: el equipo golea un partido, pierde el siguiente,
  golea de nuevo, y asi. ese patron de alta varianza en resultados
  puede predecir peor rendimiento en el partido siguiente.

  dos variables nuevas calculadas sobre los ultimos 5 partidos del equipo local:
    std_goles_5    : desviacion estandar de goles marcados (alta = impredecible)
    irregularidad  : cuantas veces alterna W/L/D en ultimos 5 partidos (0-4 cambios)
                     ej: G-P-G-P-G = 4 cambios (maxima irregularidad)
                     ej: G-G-G-G-G = 0 cambios (muy estable)

  hipotesis: equipos con alta irregularidad ganan menos en el siguiente partido
""")

# construí el historial por equipo ordenado cronológicamente
df3 = df.sort_values("Date").copy()

# acumulé los últimos resultados y goles marcados por equipo
historial = {}  # {equipo: [(fecha, resultado_propio, goles_marcados)]}

std_goles_list   = []
irregularidad_list = []

for _, row in df3.iterrows():
    ht = row["HomeTeam"]
    at = row["AwayTeam"]

    historial.setdefault(ht, [])
    historial.setdefault(at, [])

    # calculé las métricas para el local ANTES de este partido
    hist_ht = historial[ht][-5:]  # tomé los últimos 5 partidos previos

    if len(hist_ht) >= 3:
        goles_prev = [g for _, _, g in hist_ht]
        res_prev   = [r for _, r, _ in hist_ht]
        std_g = round(float(pd.Series(goles_prev).std()), 4)
        # conté cuántas veces cambió el resultado entre partidos consecutivos
        cambios = sum(1 for i in range(1, len(res_prev)) if res_prev[i] != res_prev[i-1])
        irreg = cambios
    else:
        std_g = 0.0
        irreg = 0

    std_goles_list.append(std_g)
    irregularidad_list.append(irreg)

    # actualicé el historial del local
    res_ht = "W" if row["FTR"] == "H" else "L" if row["FTR"] == "A" else "D"
    historial[ht].append((row["Date"], res_ht, int(row["FTHG"])))

    # actualicé el historial del visitante
    res_at = "W" if row["FTR"] == "A" else "L" if row["FTR"] == "H" else "D"
    historial[at].append((row["Date"], res_at, int(row["FTAG"])))

df3["std_goles_5"]    = std_goles_list
df3["irregularidad"]  = irregularidad_list

# filtré solo partidos con historial suficiente y agregué racha_previa desde df2
# agregar racha_previa desde df2 para usarla en el modelo combinado
df3 = df3.merge(
    df2[["Div","Date","HomeTeam","AwayTeam","racha_previa"]],
    on=["Div","Date","HomeTeam","AwayTeam"],
    how="left"
)

df3_valid = df3[df3["std_goles_5"] > 0].copy()
print(f"  partidos con historial suficiente (>=3 previos): {len(df3_valid):,}")

# mostré la distribución del índice de irregularidad con el % de victorias por nivel
print("\n  distribucion del indice de irregularidad (0=estable, 4=maxima alternancia):")
irr_dist = df3_valid.groupby("irregularidad").agg(
    partidos  = ("home_win", "count"),
    pct_H     = ("home_win", "mean"),
    avg_goles = ("total_goals", "mean"),
).round(3).reset_index()
irr_dist["pct_H"] = round(irr_dist["pct_H"] * 100, 2)
print_tabulate(irr_dist)

# ajusté regresión: std_goles_5 como predictor de victoria local
r_std = linear_regression(
    df3_valid, x="std_goles_5", y="home_win",
    title="home_win ~ std_goles_5 (varianza goles ultimos 5 partidos del local)",
    save_path="img/lr_home_win_std_goles.png",
    color="#e67e22"
)
print(f"\n  std_goles_5 -> home_win:")
print(f"  R²={r_std['r2']}  coef={r_std['coef']}  p={r_std['pval']}")
if r_std["pval"] < 0.05:
    dir_s = "negativo" if r_std["coef"] < 0 else "positivo"
    print(f"  SIGNIFICATIVO — coef {dir_s}: mayor varianza en goles marcados {'reduce' if r_std['coef']<0 else 'aumenta'} la probabilidad de victoria local")
else:
    print("  no significativo")

# regresion: irregularidad predice victoria local
r_irr = linear_regression(
    df3_valid, x="irregularidad", y="home_win",
    title="home_win ~ irregularidad (alternancia W/L en ultimos 5 partidos del local)",
    save_path="img/lr_home_win_irregularidad.png",
    color="#e74c3c"
)
print(f"\n  irregularidad -> home_win:")
print(f"  R²={r_irr['r2']}  coef={r_irr['coef']}  p={r_irr['pval']}")
if r_irr["pval"] < 0.05:
    dir_i = "negativo" if r_irr["coef"] < 0 else "positivo"
    print(f"  SIGNIFICATIVO — coef {dir_i}: equipos mas irregulares {'ganan menos' if r_irr['coef']<0 else 'ganan mas'} en el siguiente partido")
    if r_irr["coef"] < 0:
        print("  esto valida el fenomeno Tigres: irregularidad alta -> peor rendimiento siguiente")
else:
    print("  no significativo")

# ajusté el modelo múltiple combinando irregularidad + std + racha
r_irr_multi = linear_regression_multiple(
    df3_valid,
    features=["irregularidad", "std_goles_5", "racha_previa"],
    y="home_win",
    title="home_win ~ irregularidad + std_goles_5 + racha_previa"
)
print(f"\n  modelo combinado irregularidad + std + racha:")
print(f"  R²={r_irr_multi['r2']}  AIC={r_irr_multi['aic']}")

# grafiqué el % de victorias por nivel de irregularidad y por rango de varianza de goles
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# panel izquierdo: pct_H por nivel de irregularidad
axes[0].bar(irr_dist["irregularidad"].astype(str),
            irr_dist["pct_H"],
            color=["#2ecc71","#f39c12","#e67e22","#e74c3c","#8e44ad"],
            alpha=0.85)
axes[0].axhline(df3_valid["home_win"].mean() * 100, color="black",
                linewidth=1.2, linestyle="--",
                label=f"promedio ({df3_valid['home_win'].mean()*100:.1f}%)")
axes[0].set_xlabel("irregularidad (cambios de resultado en ultimos 5)")
axes[0].set_ylabel("% victoria local")
axes[0].set_title("% victoria local segun irregularidad previa\n(0=estable, 4=alterna siempre)",
                  fontweight="bold")
axes[0].legend()

# panel derecho: pct_H por rango de std de goles
bins_std = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 10.0]
lbl_std  = [f"{bins_std[i]}-{bins_std[i+1]}" for i in range(len(bins_std)-1)]
df3_valid["rng_std"] = pd.cut(df3_valid["std_goles_5"], bins=bins_std,
                               labels=lbl_std, right=False)
std_agg = df3_valid.groupby("rng_std", observed=True).agg(
    n     = ("home_win", "count"),
    pct_H = ("home_win", "mean"),
).reset_index()
std_agg["pct_H"] = round(std_agg["pct_H"] * 100, 2)
axes[1].bar(std_agg["rng_std"].astype(str), std_agg["pct_H"],
            color="#3498db", alpha=0.85)
axes[1].axhline(df3_valid["home_win"].mean() * 100, color="black",
                linewidth=1.2, linestyle="--")
axes[1].set_xlabel("std goles marcados (ultimos 5 partidos)")
axes[1].set_ylabel("% victoria local")
axes[1].set_title("% victoria local segun varianza de goles marcados",
                  fontweight="bold")
axes[1].tick_params(axis="x", rotation=20)

fig.suptitle("Irregularidad del equipo local como predictor de victoria\n(fenomeno Tigres-Juarez)",
             fontweight="bold", fontsize=11)
plt.tight_layout()
plt.savefig("img/lr_irregularidad_local.png", dpi=130)
plt.close()
print("\n  grafica guardada: img/lr_irregularidad_local.png")

# consolidé todos los modelos en una tabla resumen ordenada por R²

print(f"\n{'='*60}")
print("  RESUMEN FINAL — TODOS LOS MODELOS")
print(f"{'='*60}")

todos = pd.DataFrame([
    {"modelo": "ht_goals -> total_goals",              "tipo": "simple",        "r2": resultados_simples[0]["r2"], "sig": "si"},
    {"modelo": "mid_cuota -> edge (local)",             "tipo": "edge",          "r2": r_edge["r2"],               "sig": "si"},
    {"modelo": "mid_cuota -> edge (visitante)",         "tipo": "edge",          "r2": r_edge_a["r2"],             "sig": "si"},
    {"modelo": "imp_prob_H -> home_win",                "tipo": "simple",        "r2": resultados_simples[4]["r2"], "sig": "si"},
    {"modelo": "AvgCH (cierre) -> home_win",            "tipo": "simple",        "r2": resultados_simples[2]["r2"], "sig": "si"},
    {"modelo": "AvgH -> home_win",                      "tipo": "simple",        "r2": resultados_simples[1]["r2"], "sig": "si"},
    {"modelo": "AvgH+AvgA+AvgD+or -> goals",           "tipo": "multiple",      "r2": r_mult2["r2"],              "sig": "si"},
    {"modelo": "imp_H+imp_A+overround -> goals",        "tipo": "multiple",      "r2": r_mult1["r2"],              "sig": "si"},
    {"modelo": "moves_H+A+D -> total_goals",            "tipo": "multiple",      "r2": r_moves["r2"],              "sig": "si"},
    {"modelo": "moves_H+A -> home_win",                 "tipo": "multiple",      "r2": r_mov_multi["r2"],          "sig": "si"},
    {"modelo": "odds_move_H -> home_win",               "tipo": "simple",        "r2": r_mov_hw["r2"],             "sig": "si"},
    {"modelo": "racha_previa -> home_win",              "tipo": "racha",         "r2": r_racha["r2"],              "sig": "si"},
    {"modelo": "racha_num -> pct_H (categorias)",       "tipo": "racha",         "r2": r_racha_cat["r2"],          "sig": "?"},
    {"modelo": "irregularidad+std+racha -> home_win",   "tipo": "irregularidad", "r2": r_irr_multi["r2"],          "sig": "?"},
    {"modelo": "std_goles_5 -> home_win",               "tipo": "irregularidad", "r2": r_std["r2"],                "sig": "?"},
    {"modelo": "irregularidad -> home_win",             "tipo": "irregularidad", "r2": r_irr["r2"],                "sig": "?"},
    {"modelo": "post_goleada_mala -> home_win",         "tipo": "racha",         "r2": r_pg["r2"],                 "sig": "?"},
    {"modelo": "racha_previa -> total_goals",           "tipo": "racha",         "r2": r_racha_g["r2"],            "sig": "?"},
    {"modelo": "overround -> total_goals",              "tipo": "simple",        "r2": resultados_simples[3]["r2"], "sig": "si"},
    {"modelo": "diff_imp -> total_goals",               "tipo": "simple",        "r2": resultados_simples[5]["r2"], "sig": "si"},
    {"modelo": "odds_move_H -> total_goals",            "tipo": "simple",        "r2": 0.0,                        "sig": "NO"},
    {"modelo": "t -> avg_goals (temporal)",             "tipo": "temporal",      "r2": round(model_ts.rsquared, 4),"sig": "NO"},
])

print_tabulate(todos.sort_values("r2", ascending=False).reset_index(drop=True))

print("\nlisto — todas las graficas guardadas en img/")