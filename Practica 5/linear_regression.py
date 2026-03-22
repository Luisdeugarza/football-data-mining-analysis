import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
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


# carga y variables derivadas

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
df["Season_label"]      = df["Season"].map(SEASON_MAP)

ligas   = sorted(df["Div"].unique())
seasons = sorted(df["Season"].unique())

print(f"dataset cargado: {len(df):,} partidos | {len(ligas)} ligas | {len(seasons)} temporadas")


# funcion principal de regresion lineal (mismo estilo que el profe)

def linear_regression(df_in: pd.DataFrame, x: str, y: str,
                       title: str = "", save_path: str = None,
                       color: str = "#3498db") -> dict:
    """Ajusta OLS simple con sm.add_constant, imprime el summary completo
    y guarda scatter con la linea de regresion en rojo."""
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
    }


def linear_regression_multiple(df_in: pd.DataFrame, features: list, y: str,
                                 title: str = "") -> dict:
    """OLS multiple: varios features para una sola variable dependiente."""
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
    }


# 1. matriz de correlacion

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
print("\n  correlaciones con total_goals (ordenadas):")
print_tabulate(
    corr_mat["total_goals"]
    .drop("total_goals")
    .sort_values(key=abs, ascending=False)
    .reset_index()
    .rename(columns={"index": "variable", "total_goals": "corr_con_total_goals"})
)

print("\n  correlaciones con home_win (ordenadas):")
print_tabulate(
    corr_mat["home_win"]
    .drop("home_win")
    .sort_values(key=abs, ascending=False)
    .reset_index()
    .rename(columns={"index": "variable", "home_win": "corr_con_home_win"})
)

# heatmap de correlaciones
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


# 2. regresiones simples

print(f"\n{'='*60}")
print("  2. REGRESIONES SIMPLES")
print(f"{'='*60}")

resultados_simples = []

# goles del primer tiempo predicen los goles totales
r = linear_regression(
    df, x="ht_goals", y="total_goals",
    title="total_goals ~ ht_goals",
    save_path="img/lr_total_goals_ht_goals.png",
    color="#3498db"
)
resultados_simples.append(r)

# cuota local predice victoria local
r = linear_regression(
    df, x="AvgH", y="home_win",
    title="home_win ~ AvgH (apertura)",
    save_path="img/lr_home_win_AvgH.png",
    color="#e74c3c"
)
resultados_simples.append(r)

# cuota local de CIERRE predice victoria local
r = linear_regression(
    df, x="AvgCH", y="home_win",
    title="home_win ~ AvgCH (cierre)",
    save_path="img/lr_home_win_AvgCH.png",
    color="#e67e22"
)
resultados_simples.append(r)

# overround predice goles totales
r = linear_regression(
    df, x="overround", y="total_goals",
    title="total_goals ~ overround",
    save_path="img/lr_total_goals_overround.png",
    color="#9b59b6"
)
resultados_simples.append(r)

# probabilidad implicita del local predice victoria local
r = linear_regression(
    df, x="imp_prob_H", y="home_win",
    title="home_win ~ imp_prob_H",
    save_path="img/lr_home_win_imp_prob_H.png",
    color="#2ecc71"
)
resultados_simples.append(r)

# diferencia de probabilidades implicitas predice goles
df["diff_imp"] = df["imp_prob_H"] - df["imp_prob_A"]
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


# 3. regresion multiple: total_goals

print(f"\n{'='*60}")
print("  3. REGRESION MULTIPLE — total_goals")
print(f"{'='*60}")

resultados_multiple = []

features_goles = ["imp_prob_H", "imp_prob_A", "overround"]
r = linear_regression_multiple(
    df, features=features_goles, y="total_goals",
    title="total_goals ~ imp_prob_H + imp_prob_A + overround"
)
resultados_multiple.append(r)

features_goles2 = ["imp_prob_H", "imp_prob_A", "overround",
                    "odds_move_H", "odds_move_A"]
r = linear_regression_multiple(
    df, features=features_goles2, y="total_goals",
    title="total_goals ~ imp_prob_H + imp_prob_A + overround + odds_move_H + odds_move_A"
)
resultados_multiple.append(r)

# con cuotas directas en vez de probabilidades implicitas
features_odds = ["AvgH", "AvgA", "AvgD", "overround"]
r = linear_regression_multiple(
    df, features=features_odds, y="total_goals",
    title="total_goals ~ AvgH + AvgA + AvgD + overround"
)
resultados_multiple.append(r)

print(f"\n  resumen regresiones multiples (total_goals):")
print_tabulate(pd.DataFrame(resultados_multiple)[["features","y","n","r2","r2_adj","aic","bic"]])


# 4. movimiento de cuota como predictor

print(f"\n{'='*60}")
print("  4. MOVIMIENTO DE CUOTA COMO PREDICTOR")
print(f"{'='*60}")

print("\n  hipotesis: si la cuota local baja antes del cierre, predice victoria del local")

r = linear_regression(
    df, x="odds_move_H", y="home_win",
    title="home_win ~ odds_move_H (movimiento cuota local)",
    save_path="img/lr_home_win_odds_move_H.png",
    color="#3498db"
)
resultados_simples.append(r)

r = linear_regression(
    df, x="odds_move_H", y="total_goals",
    title="total_goals ~ odds_move_H",
    save_path="img/lr_total_goals_odds_move_H.png",
    color="#e74c3c"
)
resultados_simples.append(r)

# movimiento combinado H y A predice goles
r = linear_regression_multiple(
    df,
    features=["odds_move_H", "odds_move_A", "odds_move_D"],
    y="total_goals",
    title="total_goals ~ odds_move_H + odds_move_A + odds_move_D"
)
resultados_multiple.append(r)

# movimiento combinado predice victoria local
r = linear_regression_multiple(
    df,
    features=["odds_move_H", "odds_move_A"],
    y="home_win",
    title="home_win ~ odds_move_H + odds_move_A"
)
resultados_multiple.append(r)


# 5. edge como variable dependiente

print(f"\n{'='*60}")
print("  5. EDGE COMO VARIABLE DEPENDIENTE")
print(f"{'='*60}")
print("\n  el edge mide cuanto se equivoca el mercado: real_% - implied_%")
print("  si hay patron sistematico con la cuota, el mercado tiene sesgo")

bins   = [1.0, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 6.0, 25.0]
labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)]
d_edge = df.copy()
d_edge["rng_H"] = pd.cut(d_edge["AvgH"], bins=bins, labels=labels, right=False)

edge_rows = []
for rng in labels:
    sub = d_edge[d_edge["rng_H"] == rng]
    if len(sub) < 10:
        continue
    mid = (bins[labels.index(rng)] + bins[labels.index(rng) + 1]) / 2
    imp = 1 / mid * 100
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

# regresion: cuota (mid) predice edge
r_edge = linear_regression(
    df_edge, x="mid_cuota", y="edge",
    title="edge ~ mid_cuota  (sesgo del mercado por rango de cuota)",
    save_path="img/lr_edge_cuota.png",
    color="#9b59b6"
)
print(f"\n  R² del modelo edge ~ cuota: {r_edge['r2']}")
print("  si R² > 0 y coeficiente != 0, el mercado tiene sesgo sistematico en ese rango")

# lo mismo para visitante
d_edge["rng_A"] = pd.cut(d_edge["AvgA"], bins=bins, labels=labels, right=False)
edge_rows_a = []
for rng in labels:
    sub = d_edge[d_edge["rng_A"] == rng]
    if len(sub) < 10:
        continue
    mid = (bins[labels.index(rng)] + bins[labels.index(rng) + 1]) / 2
    imp = 1 / mid * 100
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
r_edge_a = linear_regression(
    df_edge_a, x="mid_cuota", y="edge",
    title="edge_visitante ~ mid_cuota",
    save_path="img/lr_edge_cuota_visitante.png",
    color="#e74c3c"
)


# 6. apertura vs cierre como predictor: cual tiene mejor R2

print(f"\n{'='*60}")
print("  6. APERTURA VS CIERRE: CUAL TIENE MEJOR R2 PREDICIENDO HOME_WIN")
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
comp_ac = pd.DataFrame([
    {"tipo": "apertura (AvgH)",  "r2": r_ap["r2"], "r2_adj": r_ap["r2_adj"],
     "coef": r_ap["coef"], "pval": r_ap["pval"]},
    {"tipo": "cierre  (AvgCH)", "r2": r_ci["r2"], "r2_adj": r_ci["r2_adj"],
     "coef": r_ci["coef"], "pval": r_ci["pval"]},
])
print_tabulate(comp_ac)
mejor = "cierre" if r_ci["r2"] > r_ap["r2"] else "apertura"
print(f"\n  mejor predictor: {mejor} (R² {max(r_ap['r2'], r_ci['r2']):.4f} vs {min(r_ap['r2'], r_ci['r2']):.4f})")

# grafica comparativa lado a lado
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
    ax.plot(x_line, m.params[1] * x_line + m.params[0],
            color="red", linewidth=1.8)
    ax.set_title(titulo, fontweight="bold")
    ax.set_xlabel(x_col); ax.set_ylabel("home_win")
fig.suptitle("Cuota apertura vs cierre como predictor de victoria local",
             fontweight="bold", fontsize=12)
plt.tight_layout()
plt.savefig("img/lr_ap_vs_ci_comparativa.png", dpi=130)
plt.close()
print("  grafica guardada: img/lr_ap_vs_ci_comparativa.png")


# 7. regresion por liga

print(f"\n{'='*60}")
print("  7. REGRESION POR LIGA  (total_goals ~ imp_prob_H + imp_prob_A + overround)")
print(f"{'='*60}")

liga_results = []
for liga in ligas:
    sub = df[df["Div"] == liga]
    cols = ["imp_prob_H", "imp_prob_A", "overround", "total_goals"]
    datos = sub[cols].dropna()
    X = sm.add_constant(datos[["imp_prob_H", "imp_prob_A", "overround"]])
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

# grafica R2 por liga
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

# scatter por liga con linea de regresion simple (AvgH vs home_win)
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
fig.suptitle("home_win ~ AvgH por liga (con linea de regresion)",
             fontweight="bold", fontsize=12)
plt.tight_layout()
plt.savefig("img/lr_home_win_por_liga.png", dpi=130)
plt.close()
print("  grafica guardada: img/lr_home_win_por_liga.png")


# 8. serie temporal: promedio de goles por fecha

print(f"\n{'='*60}")
print("  8. SERIE TEMPORAL — avg goles por fecha")
print(f"{'='*60}")
print("  mismo enfoque que el profe: agrupar por fecha y correr OLS sobre el indice temporal")

df_ts = df.groupby("Date")["total_goals"].mean().reset_index()
df_ts.columns = ["Fecha", "avg_goals"]
df_ts = df_ts.sort_values("Fecha").reset_index(drop=True)
df_ts["t"] = range(len(df_ts))

print(f"\n  {len(df_ts)} fechas con partidos | desde {df_ts['Fecha'].min().date()} hasta {df_ts['Fecha'].max().date()}")
print_tabulate(df_ts.head(10))

# OLS sobre el indice temporal
X_ts = sm.add_constant(df_ts["t"])
model_ts = sm.OLS(df_ts["avg_goals"], X_ts).fit()
print(model_ts.summary())

const_ts = model_ts.params["const"]
slope_ts = model_ts.params["t"]
print(f"\n  pendiente: {slope_ts:.6f} goles por jornada")
print(f"  R²: {model_ts.rsquared:.4f}")
if slope_ts > 0:
    print("  tendencia: los partidos son marginalmente mas goleadores con el tiempo")
else:
    print("  tendencia: los partidos son marginalmente menos goleadores con el tiempo")

fig, ax = plt.subplots(figsize=(14, 5))
ax.scatter(df_ts["Fecha"], df_ts["avg_goals"],
           alpha=0.4, s=10, color="#3498db", label="avg goles por fecha")
x_line_ts = df_ts["t"].values
ax.plot(df_ts["Fecha"],
        slope_ts * x_line_ts + const_ts,
        color="red", linewidth=2,
        label=f"tendencia lineal  R²={model_ts.rsquared:.4f}")
ax.set_xlabel("Fecha"); ax.set_ylabel("avg goles por partido")
ax.set_title("Serie temporal: promedio de goles por fecha con regresion lineal",
             fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig("img/lr_serie_temporal_goles.png", dpi=130)
plt.close()
print("  grafica guardada: img/lr_serie_temporal_goles.png")

# por liga: tendencia temporal de goles
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
ax.set_title("Tendencia temporal de goles por liga (regresion lineal)",
             fontweight="bold")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("img/lr_tendencia_goles_por_liga.png", dpi=130)
plt.close()
print("  grafica guardada: img/lr_tendencia_goles_por_liga.png")


# resumen final de todos los modelos

print(f"\n{'='*60}")
print("  RESUMEN FINAL — TODOS LOS MODELOS")
print(f"{'='*60}")

todos = pd.DataFrame([
    {"modelo": "ht_goals → total_goals",        "tipo": "simple",   "r2": resultados_simples[0]["r2"]},
    {"modelo": "AvgH → home_win",               "tipo": "simple",   "r2": resultados_simples[1]["r2"]},
    {"modelo": "AvgCH (cierre) → home_win",     "tipo": "simple",   "r2": resultados_simples[2]["r2"]},
    {"modelo": "overround → total_goals",        "tipo": "simple",   "r2": resultados_simples[3]["r2"]},
    {"modelo": "imp_prob_H → home_win",         "tipo": "simple",   "r2": resultados_simples[4]["r2"]},
    {"modelo": "diff_imp → total_goals",        "tipo": "simple",   "r2": resultados_simples[5]["r2"]},
    {"modelo": "odds_move_H → home_win",        "tipo": "simple",   "r2": resultados_simples[6]["r2"]},
    {"modelo": "odds_move_H → total_goals",     "tipo": "simple",   "r2": resultados_simples[7]["r2"]},
    {"modelo": "imp_H+imp_A+overround → goals", "tipo": "multiple", "r2": resultados_multiple[0]["r2"]},
    {"modelo": "imp+overround+moves → goals",   "tipo": "multiple", "r2": resultados_multiple[1]["r2"]},
    {"modelo": "AvgH+AvgA+AvgD+or → goals",    "tipo": "multiple", "r2": resultados_multiple[2]["r2"]},
    {"modelo": "moves_H+A+D → total_goals",     "tipo": "multiple", "r2": resultados_multiple[3]["r2"]},
    {"modelo": "moves_H+A → home_win",          "tipo": "multiple", "r2": resultados_multiple[4]["r2"]},
    {"modelo": "mid_cuota → edge (local)",      "tipo": "edge",     "r2": r_edge["r2"]},
    {"modelo": "mid_cuota → edge (visitante)",  "tipo": "edge",     "r2": r_edge_a["r2"]},
    {"modelo": "t → avg_goals (temporal)",      "tipo": "temporal", "r2": round(model_ts.rsquared, 4)},
])

print_tabulate(todos.sort_values("r2", ascending=False).reset_index(drop=True))

print(f"\n  mejor modelo simple:   {todos[todos['tipo']=='simple'].sort_values('r2',ascending=False).iloc[0]['modelo']}")
print(f"  mejor modelo multiple: {todos[todos['tipo']=='multiple'].sort_values('r2',ascending=False).iloc[0]['modelo']}")
print(f"\n  nota: R² bajos son esperados en datos de futbol — el resultado tiene mucha")
print(f"  varianza aleatoria. Lo importante es que los coeficientes sean significativos")
print(f"  (p < 0.05) y que las direcciones de los efectos tengan sentido economico.")

print("\nlisto — todas las graficas guardadas en img/")