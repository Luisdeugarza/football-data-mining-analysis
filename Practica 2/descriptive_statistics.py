import pandas as pd
from tabulate import tabulate
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
from typing import List

def print_tabulate(df: pd.DataFrame):
    print(tabulate(df, headers=df.columns, tablefmt="orgtbl"))

def get_cmap(n, name="hsv"):
    return matplotlib.colormaps.get_cmap(name).resampled(n)

def describe_numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    stats = []
    for col in columns:
        if col in df.columns:
            s = df[col].dropna()
            stats.append({
                "column": col,
                "mean": round(s.mean(), 3),
                "median": round(s.median(), 3),
                "std": round(s.std(), 3),
                "min": round(s.min(), 3),
                "max": round(s.max(), 3),
                "q25": round(s.quantile(0.25), 3),
                "q75": round(s.quantile(0.75), 3),
            })
    return pd.DataFrame(stats)

def describe_categorical(df: pd.DataFrame, column: str) -> pd.DataFrame:
    counts = df[column].value_counts().reset_index()
    counts.columns = [column, "count"]
    counts["pct"] = round(counts["count"] / len(df) * 100, 2)
    return counts

def scatter_group_by(file_path: str, df: pd.DataFrame, x_column: str, y_column: str, label_column: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = pd.unique(df[label_column])
    cmap = get_cmap(len(labels) + 1)
    for i, label in enumerate(labels):
        filter_df = df[df[label_column] == label]
        ax.scatter(filter_df[x_column], filter_df[y_column], label=label, color=cmap(i), alpha=0.5, s=10)
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.legend()
    plt.savefig(file_path)
    plt.close()

def draw_er_diagram(file_path: str):
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis("off")

    def entity(ax, x, y, title, fields, w=2.2, h=2.6):
        ax.add_patch(mpatches.FancyBboxPatch((x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.1", linewidth=2,
            edgecolor="#2c3e50", facecolor="#d6eaf8"))
        ax.text(x, y + h/2 - 0.25, title, ha="center", va="center",
                fontsize=10, fontweight="bold", color="#2c3e50")
        ax.plot([x - w/2, x + w/2], [y + h/2 - 0.45, y + h/2 - 0.45],
                color="#2c3e50", linewidth=1)
        for i, f in enumerate(fields):
            ax.text(x, y + h/2 - 0.7 - i * 0.32, f, ha="center", va="center",
                    fontsize=7.5, color="#1a252f")

    def relation(ax, x, y, label, w=1.3, h=0.55):
        diamond = plt.Polygon(
            [[x, y + h], [x + w/2, y], [x, y - h], [x - w/2, y]],
            closed=True, linewidth=1.5,
            edgecolor="#2c3e50", facecolor="#fef9e7")
        ax.add_patch(diamond)
        ax.text(x, y, label, ha="center", va="center", fontsize=8, color="#2c3e50")

    def arrow(ax, x1, y1, x2, y2, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#2c3e50", lw=1.5))
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx + 0.1, my + 0.1, label, fontsize=7, color="#555")

    # entidades
    entity(ax, 7,   4.5, "Match",  ["Date", "FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR"])
    entity(ax, 2,   7,   "Team",   ["HomeTeam / AwayTeam"])
    entity(ax, 12,  7,   "Odds",   ["B365H/D/A", "MaxH/D/A", "AvgH/D/A", "B365CH/D/A", "MaxCH/D/A", "AvgCH/D/A"])
    entity(ax, 2,   2,   "League", ["Div (E0/SP1/D1/I1/F1)"])
    entity(ax, 12,  2,   "Season", ["Season (1920..2526)"])

    # relaciones
    relation(ax, 4.3, 6.1, "plays")
    relation(ax, 9.7, 6.1, "has odds")
    relation(ax, 4.3, 3.2, "belongs to")
    relation(ax, 9.7, 3.2, "played in")

    # flechas entidad - relacion - match
    arrow(ax, 2,   6.7,  3.5, 6.3)
    arrow(ax, 5.1, 6.1,  5.9, 5.1)
    arrow(ax, 12,  6.7,  10.5, 6.3)
    arrow(ax, 8.8, 5.5,  10.3, 6.1, "1:1")
    arrow(ax, 2,   2.7,  3.5, 3.1)
    arrow(ax, 5.1, 3.2,  5.9, 3.9, "N:1")
    arrow(ax, 12,  2.7,  10.5, 3.1)
    arrow(ax, 8.8, 3.9,  10.3, 3.2, "N:1")

    # cardinalidades
    ax.text(3.0, 6.6, "N", fontsize=8, color="#e74c3c", fontweight="bold")
    ax.text(5.5, 5.6, "1", fontsize=8, color="#e74c3c", fontweight="bold")
    ax.text(11.0, 6.6, "1", fontsize=8, color="#e74c3c", fontweight="bold")
    ax.text(3.0, 2.9, "N", fontsize=8, color="#e74c3c", fontweight="bold")
    ax.text(11.0, 2.9, "N", fontsize=8, color="#e74c3c", fontweight="bold")

    ax.set_title("Diagrama Entidad-Relación — European Football Dataset", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(file_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"diagrama ER guardado en {file_path}")

os.makedirs("img", exist_ok=True)

df = pd.read_csv("../Practica 1/data/clean/football_clean.csv", parse_dates=["Date"])

# columnas derivadas
df["total_goals"] = df["FTHG"] + df["FTAG"]
df["goal_diff"] = df["FTHG"] - df["FTAG"]
df["home_win"] = (df["FTR"] == "H").astype(int)
df["draw"] = (df["FTR"] == "D").astype(int)
df["away_win"] = (df["FTR"] == "A").astype(int)
df["btts"] = ((df["FTHG"] > 0) & (df["FTAG"] > 0)).astype(int)
df["over15"] = (df["total_goals"] > 1).astype(int)
df["over25"] = (df["total_goals"] > 2).astype(int)
df["over35"] = (df["total_goals"] > 3).astype(int)
df["over45"] = (df["total_goals"] > 4).astype(int)
df["under15"] = (df["total_goals"] <= 1).astype(int)
df["under25"] = (df["total_goals"] <= 2).astype(int)
df["under35"] = (df["total_goals"] <= 3).astype(int)
df["imp_prob_H"] = round(1 / df["AvgH"], 4)
df["imp_prob_D"] = round(1 / df["AvgD"], 4)
df["imp_prob_A"] = round(1 / df["AvgA"], 4)
df["overround"] = round(df["imp_prob_H"] + df["imp_prob_D"] + df["imp_prob_A"], 4)
df["odds_move_H"] = round(df["AvgCH"] - df["AvgH"], 4)
df["odds_move_A"] = round(df["AvgCA"] - df["AvgA"], 4)
df["odds_move_D"] = round(df["AvgCD"] - df["AvgD"], 4)
df["is_underdog_away"] = (df["AvgA"] > 4).astype(int)
df["is_underdog_home"] = (df["AvgH"] > 4).astype(int)
df["month"] = df["Date"].dt.month
df["year"] = df["Date"].dt.year

ligas = sorted(df["Div"].unique())
temporadas = sorted(df["Season"].unique())


print("=== shape ===")
print(df.shape)

print("\n=== dtypes ===")
print(df.dtypes)


print("\n\n--- ESTADISTICAS GENERALES ---")

print("\n=== goles ===")
print_tabulate(describe_numeric(df, ["FTHG", "FTAG", "HTHG", "HTAG", "total_goals", "goal_diff"]))

print("\n=== odds apertura ===")
print_tabulate(describe_numeric(df, ["B365H", "B365D", "B365A", "MaxH", "MaxD", "MaxA", "AvgH", "AvgD", "AvgA"]))

print("\n=== odds cierre ===")
print_tabulate(describe_numeric(df, ["B365CH", "B365CD", "B365CA", "MaxCH", "MaxCD", "MaxCA", "AvgCH", "AvgCD", "AvgCA"]))

print("\n=== movimiento de mercado ===")
print_tabulate(describe_numeric(df, ["odds_move_H", "odds_move_D", "odds_move_A"]))

print("\n=== implied probability y overround ===")
print_tabulate(describe_numeric(df, ["imp_prob_H", "imp_prob_D", "imp_prob_A", "overround"]))

print("\n=== distribucion resultados ===")
print_tabulate(describe_categorical(df, "FTR"))

print("\n=== btts / over / under generales ===")
for label, col in [("btts", "btts"), ("over 1.5", "over15"), ("over 2.5", "over25"),
                   ("over 3.5", "over35"), ("over 4.5", "over45"),
                   ("under 1.5", "under15"), ("under 2.5", "under25"), ("under 3.5", "under35")]:
    print(f"{label}: {df[col].sum()} ({round(df[col].mean()*100,2)}%)")


print("\n\n--- POR LIGA ---")

print("\n=== goles por liga ===")
gl = df.groupby("Div").agg(
    partidos=("total_goals", "count"),
    avg_local=("FTHG", "mean"),
    avg_visitante=("FTAG", "mean"),
    avg_total=("total_goals", "mean"),
    std_total=("total_goals", "std"),
    max_goles=("total_goals", "max"),
).round(3).reset_index()
print_tabulate(gl)

print("\n=== btts / over / under por liga ===")
goal_flags = ["btts", "over15", "over25", "over35", "over45", "under15", "under25", "under35"]
gf_liga = df.groupby("Div")[goal_flags].mean().round(4) * 100
gf_liga = gf_liga.round(2).reset_index()
print_tabulate(gf_liga)

print("\n=== resultados H/D/A por liga ===")
ftr_l = df.groupby(["Div", "FTR"]).size().unstack(fill_value=0).reset_index()
ftr_l["total"] = ftr_l[["A", "D", "H"]].sum(axis=1)
ftr_l["pct_H"] = round(ftr_l["H"] / ftr_l["total"] * 100, 2)
ftr_l["pct_D"] = round(ftr_l["D"] / ftr_l["total"] * 100, 2)
ftr_l["pct_A"] = round(ftr_l["A"] / ftr_l["total"] * 100, 2)
print_tabulate(ftr_l)

print("\n=== odds promedio por liga ===")
ol = df.groupby("Div")[["AvgH", "AvgD", "AvgA", "AvgCH", "AvgCD", "AvgCA", "overround"]].mean().round(3).reset_index()
print_tabulate(ol)

print("\n=== movimiento de mercado por liga ===")
ml = df.groupby("Div")[["odds_move_H", "odds_move_D", "odds_move_A"]].agg(["mean", "std"]).round(4).reset_index()
print_tabulate(ml)

print("\n=== underdogs visitante por liga (AvgA > 4) ===")
for liga in ligas:
    sub = df[df["Div"] == liga]
    ud = sub[sub["is_underdog_away"] == 1]
    print(f"{liga} — total: {len(ud)} | ganan: {ud['away_win'].sum()} ({round(ud['away_win'].mean()*100,2)}%) | avg odd: {round(ud['AvgA'].mean(),3)}")

print("\n=== underdogs local por liga (AvgH > 4) ===")
for liga in ligas:
    sub = df[df["Div"] == liga]
    ud = sub[sub["is_underdog_home"] == 1]
    if len(ud) > 0:
        print(f"{liga} — total: {len(ud)} | ganan: {ud['home_win'].sum()} ({round(ud['home_win'].mean()*100,2)}%) | avg odd: {round(ud['AvgH'].mean(),3)}")

print("\n=== diferencia de goles por liga ===")
gd = df.groupby("Div")["goal_diff"].agg(["mean", "std", "min", "max"]).round(3).reset_index()
print_tabulate(gd)


print("\n\n--- POR TEMPORADA ---")

print("\n=== goles por temporada ===")
gs = df.groupby("Season").agg(
    partidos=("total_goals", "count"),
    avg_local=("FTHG", "mean"),
    avg_visitante=("FTAG", "mean"),
    avg_total=("total_goals", "mean"),
    std_total=("total_goals", "std"),
).round(3).reset_index()
print_tabulate(gs)

print("\n=== btts / over / under por temporada ===")
gf_temp = df.groupby("Season")[goal_flags].mean().round(4) * 100
gf_temp = gf_temp.round(2).reset_index()
print_tabulate(gf_temp)

print("\n=== resultados por temporada ===")
ftr_s = df.groupby(["Season", "FTR"]).size().unstack(fill_value=0).reset_index()
ftr_s["total"] = ftr_s[["A", "D", "H"]].sum(axis=1)
ftr_s["pct_H"] = round(ftr_s["H"] / ftr_s["total"] * 100, 2)
ftr_s["pct_D"] = round(ftr_s["D"] / ftr_s["total"] * 100, 2)
ftr_s["pct_A"] = round(ftr_s["A"] / ftr_s["total"] * 100, 2)
print_tabulate(ftr_s)

print("\n=== overround por temporada ===")
ov_s = df.groupby("Season")["overround"].agg(["mean", "std", "min", "max"]).round(4).reset_index()
print_tabulate(ov_s)

print("\n=== movimiento de mercado por temporada ===")
ms = df.groupby("Season")[["odds_move_H", "odds_move_D", "odds_move_A"]].mean().round(4).reset_index()
print_tabulate(ms)


print("\n\n--- POR LIGA Y TEMPORADA ---")

print("\n=== goles por liga y temporada ===")
glt = df.groupby(["Div", "Season"]).agg(
    avg_total=("total_goals", "mean"),
    avg_local=("FTHG", "mean"),
    avg_visitante=("FTAG", "mean"),
).round(3).reset_index()
print_tabulate(glt)

print("\n=== btts / over / under por liga y temporada ===")
gf_lt = df.groupby(["Div", "Season"])[goal_flags].mean().round(4) * 100
gf_lt = gf_lt.round(2).reset_index()
print_tabulate(gf_lt)

print("\n=== resultados por liga y temporada ===")
ftr_lt = df.groupby(["Div", "Season", "FTR"]).size().unstack(fill_value=0).reset_index()
ftr_lt["total"] = ftr_lt[["A", "D", "H"]].sum(axis=1)
ftr_lt["pct_H"] = round(ftr_lt["H"] / ftr_lt["total"] * 100, 2)
ftr_lt["pct_D"] = round(ftr_lt["D"] / ftr_lt["total"] * 100, 2)
ftr_lt["pct_A"] = round(ftr_lt["A"] / ftr_lt["total"] * 100, 2)
print_tabulate(ftr_lt)

print("\n=== odds promedio por liga y temporada ===")
olt = df.groupby(["Div", "Season"])[["AvgH", "AvgD", "AvgA", "overround"]].mean().round(3).reset_index()
print_tabulate(olt)

print("\n=== underdogs visitante por liga y temporada ===")
ud_lt = df[df["is_underdog_away"] == 1].groupby(["Div", "Season"]).agg(
    total=("away_win", "count"),
    ganan=("away_win", "sum"),
    pct_win=("away_win", "mean"),
    avg_odd=("AvgA", "mean")
).round(3).reset_index()
ud_lt["pct_win"] = round(ud_lt["pct_win"] * 100, 2)
print_tabulate(ud_lt)

print("\n=== movimiento de mercado por liga y temporada ===")
mlt = df.groupby(["Div", "Season"])[["odds_move_H", "odds_move_D", "odds_move_A"]].mean().round(4).reset_index()
print_tabulate(mlt)


print("\n\n--- POR MES ---")

print("\n=== goles y resultados por mes ===")
gm = df.groupby("month").agg(
    partidos=("total_goals", "count"),
    avg_total=("total_goals", "mean"),
    pct_H=("home_win", "mean"),
    pct_D=("draw", "mean"),
    pct_A=("away_win", "mean"),
    pct_btts=("btts", "mean"),
    pct_over25=("over25", "mean"),
).round(3).reset_index()
print_tabulate(gm)


print("\n\n--- POR EQUIPO ---")

print("\n=== top 15 goles local ===")
hg = df.groupby("HomeTeam").agg(
    partidos=("FTHG", "count"),
    total=("FTHG", "sum"),
    avg=("FTHG", "mean"),
    max_partido=("FTHG", "max")
).round(3).sort_values("total", ascending=False).head(15).reset_index()
print_tabulate(hg)

print("\n=== top 15 goles visitante ===")
ag = df.groupby("AwayTeam").agg(
    partidos=("FTAG", "count"),
    total=("FTAG", "sum"),
    avg=("FTAG", "mean"),
    max_partido=("FTAG", "max")
).round(3).sort_values("total", ascending=False).head(15).reset_index()
print_tabulate(ag)

print("\n=== mayor % victorias local (min 50 partidos) ===")
hw = df.groupby("HomeTeam").agg(partidos=("home_win", "count"), victorias=("home_win", "sum"))
hw = hw[hw["partidos"] >= 50]
hw["pct_win"] = round(hw["victorias"] / hw["partidos"] * 100, 2)
print_tabulate(hw.sort_values("pct_win", ascending=False).head(15).reset_index())

print("\n=== mayor % victorias visitante (min 50 partidos) ===")
aw = df.groupby("AwayTeam").agg(partidos=("away_win", "count"), victorias=("away_win", "sum"))
aw = aw[aw["partidos"] >= 50]
aw["pct_win"] = round(aw["victorias"] / aw["partidos"] * 100, 2)
print_tabulate(aw.sort_values("pct_win", ascending=False).head(15).reset_index())

print("\n=== mayor % empates (min 50 partidos) ===")
dr = df.groupby("HomeTeam").agg(partidos=("draw", "count"), empates=("draw", "sum"))
dr = dr[dr["partidos"] >= 50]
dr["pct_draw"] = round(dr["empates"] / dr["partidos"] * 100, 2)
print_tabulate(dr.sort_values("pct_draw", ascending=False).head(15).reset_index())

print("\n=== equipos mas favoritos como local (avg odd mas baja) ===")
fav_h = df.groupby("HomeTeam")["AvgH"].agg(["mean", "count"]).reset_index()
fav_h = fav_h[fav_h["count"] >= 50].sort_values("mean").head(10)
print_tabulate(fav_h)

print("\n=== equipos mas underdogs como visitante (min 30 veces AvgA > 4) ===")
ud_team = df[df["is_underdog_away"] == 1].groupby("AwayTeam").agg(
    veces=("away_win", "count"),
    gana=("away_win", "sum"),
    avg_odd=("AvgA", "mean")
)
ud_team = ud_team[ud_team["veces"] >= 30]
ud_team["pct_win"] = round(ud_team["gana"] / ud_team["veces"] * 100, 2)
print_tabulate(ud_team.sort_values("pct_win", ascending=False).head(15).reset_index())


print("\n\n--- ANALISIS DE CUOTAS ---")

print("\n=== partidos con favorito local mas extremo ===")
print_tabulate(df[["Div", "Season", "Date", "HomeTeam", "AwayTeam", "AvgH", "AvgA", "FTR", "FTHG", "FTAG"]].sort_values("AvgH").head(15))

print("\n=== partidos con favorito visitante mas extremo ===")
print_tabulate(df[["Div", "Season", "Date", "HomeTeam", "AwayTeam", "AvgH", "AvgA", "FTR", "FTHG", "FTAG"]].sort_values("AvgA").head(15))

print("\n=== underdogs extremos visitante (AvgA > 10) que ganaron ===")
ude = df[(df["AvgA"] > 10) & (df["away_win"] == 1)]
print(f"total: {len(ude)}")
print_tabulate(ude[["Div", "Season", "Date", "HomeTeam", "AwayTeam", "AvgH", "AvgA", "FTHG", "FTAG"]].sort_values("AvgA", ascending=False))

print("\n=== smart money hacia local (odds bajan > 0.1) ===")
smart_h = df[df["odds_move_H"] < -0.1]
print(f"partidos: {len(smart_h)}")
print(f"gana local: {smart_h['home_win'].sum()} ({round(smart_h['home_win'].mean()*100,2)}%)")
print(f"empate:     {smart_h['draw'].sum()} ({round(smart_h['draw'].mean()*100,2)}%)")
print(f"gana visit: {smart_h['away_win'].sum()} ({round(smart_h['away_win'].mean()*100,2)}%)")

print("\n=== smart money hacia visitante (odds bajan > 0.1) ===")
smart_a = df[df["odds_move_A"] < -0.1]
print(f"partidos: {len(smart_a)}")
print(f"gana local: {smart_a['home_win'].sum()} ({round(smart_a['home_win'].mean()*100,2)}%)")
print(f"empate:     {smart_a['draw'].sum()} ({round(smart_a['draw'].mean()*100,2)}%)")
print(f"gana visit: {smart_a['away_win'].sum()} ({round(smart_a['away_win'].mean()*100,2)}%)")

print("\n=== smart money por liga ===")
for liga in ligas:
    sub = df[df["Div"] == liga]
    sh = sub[sub["odds_move_H"] < -0.1]
    sa = sub[sub["odds_move_A"] < -0.1]
    print(f"{liga} — smart money local: {len(sh)} partidos, gana local {round(sh['home_win'].mean()*100,2)}% | smart money visit: {len(sa)} partidos, gana visit {round(sa['away_win'].mean()*100,2)}%")

print("\n=== correlacion general ===")
corr_cols = ["total_goals", "FTHG", "FTAG", "AvgH", "AvgA", "AvgD",
             "overround", "odds_move_H", "odds_move_A", "imp_prob_H", "imp_prob_A"]
print(round(df[corr_cols].corr(), 3))


print("\n\n--- GRAFICAS ---")

draw_er_diagram("img/er_diagram.png")

scatter_group_by("img/odds_apertura_por_liga.png", df, "AvgH", "AvgA", "Div")
scatter_group_by("img/goles_local_vs_visitante.png", df, "FTHG", "FTAG", "Div")
scatter_group_by("img/odds_apertura_vs_cierre_H.png", df, "AvgH", "AvgCH", "Div")
scatter_group_by("img/odds_apertura_vs_cierre_A.png", df, "AvgA", "AvgCA", "Div")
scatter_group_by("img/implied_prob_H_vs_A.png", df, "imp_prob_H", "imp_prob_A", "Div")
scatter_group_by("img/movimiento_mercado_H_vs_A.png", df, "odds_move_H", "odds_move_A", "Div")

# overround por liga
fig, ax = plt.subplots(figsize=(10, 5))
for liga in ligas:
    ax.hist(df[df["Div"] == liga]["overround"], bins=40, alpha=0.5, label=liga)
ax.set_xlabel("overround")
ax.set_ylabel("frecuencia")
ax.legend()
plt.savefig("img/overround_por_liga.png")
plt.close()

# goles promedio por temporada y liga
fig, ax = plt.subplots(figsize=(12, 6))
cmap = get_cmap(len(ligas) + 1)
for i, liga in enumerate(ligas):
    sub = df[df["Div"] == liga].groupby("Season")["total_goals"].mean()
    ax.plot(sub.index.astype(str), sub.values, marker="o", label=liga, color=cmap(i))
ax.set_xlabel("temporada")
ax.set_ylabel("goles promedio")
ax.legend()
plt.savefig("img/goles_por_temporada_y_liga.png")
plt.close()

# resultados pie por liga
fig, axes = plt.subplots(1, 5, figsize=(20, 5))
for i, liga in enumerate(ligas):
    sub = df[df["Div"] == liga]["FTR"].value_counts()
    axes[i].pie(sub.values, labels=sub.index, autopct="%1.1f%%")
    axes[i].set_title(liga)
plt.savefig("img/resultados_pie_por_liga.png")
plt.close()

# btts y over por liga
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(ligas))
w = 0.15
metrics = [("btts", "btts"), ("over 1.5", "over15"), ("over 2.5", "over25"),
           ("over 3.5", "over35"), ("under 2.5", "under25")]
for j, (label, col) in enumerate(metrics):
    vals = [df[df["Div"] == l][col].mean() * 100 for l in ligas]
    ax.bar(x + j * w, vals, w, label=label)
ax.set_xticks(x + w * 2)
ax.set_xticklabels(ligas)
ax.set_ylabel("%")
ax.legend()
plt.savefig("img/btts_over_under_por_liga.png")
plt.close()

# distribucion goles totales
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df["total_goals"], bins=range(0, 15), edgecolor="black", align="left")
ax.set_xlabel("goles en el partido")
ax.set_ylabel("frecuencia")
plt.savefig("img/distribucion_goles_totales.png")
plt.close()

# distribucion goles por liga
fig, axes = plt.subplots(1, 5, figsize=(20, 5))
for i, liga in enumerate(ligas):
    sub = df[df["Div"] == liga]["total_goals"]
    axes[i].hist(sub, bins=range(0, 12), edgecolor="black", align="left")
    axes[i].set_title(liga)
    axes[i].set_xlabel("goles")
plt.savefig("img/distribucion_goles_por_liga.png")
plt.close()

# odds underdog gana vs pierde
fig, ax = plt.subplots(figsize=(10, 5))
ud_win = df[(df["is_underdog_away"] == 1) & (df["away_win"] == 1)]["AvgA"]
ud_lose = df[(df["is_underdog_away"] == 1) & (df["away_win"] == 0)]["AvgA"]
ax.hist(ud_win, bins=30, alpha=0.6, label="underdog gana")
ax.hist(ud_lose, bins=30, alpha=0.6, label="underdog pierde")
ax.set_xlabel("AvgA")
ax.legend()
plt.savefig("img/underdog_win_vs_lose.png")
plt.close()

# % over 2.5 por temporada y liga
fig, ax = plt.subplots(figsize=(12, 6))
cmap = get_cmap(len(ligas) + 1)
for i, liga in enumerate(ligas):
    sub = df[df["Div"] == liga].groupby("Season")["over25"].mean() * 100
    ax.plot(sub.index.astype(str), sub.values, marker="o", label=liga, color=cmap(i))
ax.set_xlabel("temporada")
ax.set_ylabel("% over 2.5")
ax.legend()
plt.savefig("img/over25_por_temporada_y_liga.png")
plt.close()

# btts por temporada y liga
fig, ax = plt.subplots(figsize=(12, 6))
for i, liga in enumerate(ligas):
    sub = df[df["Div"] == liga].groupby("Season")["btts"].mean() * 100
    ax.plot(sub.index.astype(str), sub.values, marker="o", label=liga, color=cmap(i))
ax.set_xlabel("temporada")
ax.set_ylabel("% btts")
ax.legend()
plt.savefig("img/btts_por_temporada_y_liga.png")
plt.close()

print("guardadas imagenes en img/")