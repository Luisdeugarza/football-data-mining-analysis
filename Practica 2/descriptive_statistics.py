import pandas as pd
from tabulate import tabulate
import matplotlib
import matplotlib.pyplot as plt
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

os.makedirs("img", exist_ok=True)

df = pd.read_csv("../Practica 1/data/clean/football_clean.csv", parse_dates=["Date"])

# columnas derivadas
df["total_goals"] = df["FTHG"] + df["FTAG"]
df["goal_diff"] = df["FTHG"] - df["FTAG"]
df["home_win"] = (df["FTR"] == "H").astype(int)
df["draw"] = (df["FTR"] == "D").astype(int)
df["away_win"] = (df["FTR"] == "A").astype(int)
df["btts"] = ((df["FTHG"] > 0) & (df["FTAG"] > 0)).astype(int)  # btts (ambos anotan)
df["over25"] = (df["total_goals"] > 2).astype(int)
df["over35"] = (df["total_goals"] > 3).astype(int)
df["imp_prob_H"] = round(1 / df["AvgH"], 4)
df["imp_prob_D"] = round(1 / df["AvgD"], 4)
df["imp_prob_A"] = round(1 / df["AvgA"], 4)
df["overround"] = round(df["imp_prob_H"] + df["imp_prob_D"] + df["imp_prob_A"], 4)
df["odds_move_H"] = round(df["AvgCH"] - df["AvgH"], 4)
df["odds_move_A"] = round(df["AvgCA"] - df["AvgA"], 4)
df["odds_move_D"] = round(df["AvgCD"] - df["AvgD"], 4)
df["is_underdog_away"] = (df["AvgA"] > 4).astype(int)
df["is_underdog_home"] = (df["AvgH"] > 4).astype(int)
df["value_home"] = round(df["imp_prob_H"] * df["AvgH"], 4)  # EV local 
df["value_away"] = round(df["imp_prob_A"] * df["AvgA"], 4)
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

print("\n=== btts y over 2.5 / 3.5 generales ===")
print(f"btts: {df['btts'].sum()} ({round(df['btts'].mean()*100,2)}%)")
print(f"over 2.5: {df['over25'].sum()} ({round(df['over25'].mean()*100,2)}%)")
print(f"over 3.5: {df['over35'].sum()} ({round(df['over35'].mean()*100,2)}%)")


print("\n\n--- POR LIGA ---")

print("\n=== goles por liga ===")
gl = df.groupby("Div").agg(
    partidos=("total_goals", "count"),
    avg_local=("FTHG", "mean"),
    avg_visitante=("FTAG", "mean"),
    avg_total=("total_goals", "mean"),
    max_goles=("total_goals", "max"),
    std_goles=("total_goals", "std"),
    pct_btts=("btts", "mean"),
    pct_over25=("over25", "mean"),
    pct_over35=("over35", "mean"),
).round(3).reset_index()
print_tabulate(gl)

print("\n=== resultados H/D/A por liga con porcentajes ===")
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

print("\n=== underdogs por liga (AvgA > 4) ===")
for liga in ligas:
    sub = df[df["Div"] == liga]
    ud = sub[sub["is_underdog_away"] == 1]
    wins = ud["away_win"].sum()
    print(f"{liga} — underdogs: {len(ud)} | ganan: {wins} ({round(ud['away_win'].mean()*100,2)}%) | odd promedio: {round(ud['AvgA'].mean(),3)}")

print("\n=== underdogs locales por liga (AvgH > 4) ===")
for liga in ligas:
    sub = df[df["Div"] == liga]
    ud = sub[sub["is_underdog_home"] == 1]
    if len(ud) > 0:
        wins = ud["home_win"].sum()
        print(f"{liga} — underdogs local: {len(ud)} | ganan: {wins} ({round(ud['home_win'].mean()*100,2)}%) | odd promedio: {round(ud['AvgH'].mean(),3)}")

print("\n=== diferencia de goles promedio por liga ===")
gd = df.groupby("Div")["goal_diff"].agg(["mean", "std", "min", "max"]).round(3).reset_index()
print_tabulate(gd)


print("\n\n--- POR TEMPORADA ---")

print("\n=== goles por temporada ===")
gs = df.groupby("Season").agg(
    partidos=("total_goals", "count"),
    avg_local=("FTHG", "mean"),
    avg_visitante=("FTAG", "mean"),
    avg_total=("total_goals", "mean"),
    pct_btts=("btts", "mean"),
    pct_over25=("over25", "mean"),
    pct_over35=("over35", "mean"),
).round(3).reset_index()
print_tabulate(gs)

print("\n=== resultados por temporada ===")
ftr_s = df.groupby(["Season", "FTR"]).size().unstack(fill_value=0).reset_index()
ftr_s["total"] = ftr_s[["A", "D", "H"]].sum(axis=1)
ftr_s["pct_H"] = round(ftr_s["H"] / ftr_s["total"] * 100, 2)
ftr_s["pct_D"] = round(ftr_s["D"] / ftr_s["total"] * 100, 2)
ftr_s["pct_A"] = round(ftr_s["A"] / ftr_s["total"] * 100, 2)
print_tabulate(ftr_s)

print("\n=== overround por temporada ===")
os_ = df.groupby("Season")["overround"].agg(["mean", "std", "min", "max"]).round(4).reset_index()
print_tabulate(os_)

print("\n=== movimiento de mercado por temporada ===")
ms = df.groupby("Season")[["odds_move_H", "odds_move_D", "odds_move_A"]].mean().round(4).reset_index()
print_tabulate(ms)


print("\n\n--- POR LIGA Y TEMPORADA ---")

print("\n=== goles por liga y temporada ===")
glt = df.groupby(["Div", "Season"]).agg(
    avg_total=("total_goals", "mean"),
    pct_btts=("btts", "mean"),
    pct_over25=("over25", "mean"),
    pct_H=("home_win", "mean"),
    pct_A=("away_win", "mean"),
).round(3).reset_index()
print_tabulate(glt)

print("\n=== odds promedio por liga y temporada ===")
olt = df.groupby(["Div", "Season"])[["AvgH", "AvgD", "AvgA", "overround"]].mean().round(3).reset_index()
print_tabulate(olt)

print("\n=== underdogs por liga y temporada ===")
ud_lt = df[df["is_underdog_away"] == 1].groupby(["Div", "Season"]).agg(
    total_underdogs=("away_win", "count"),
    ganan=("away_win", "sum"),
    pct_win=("away_win", "mean"),
    avg_odd=("AvgA", "mean")
).round(3).reset_index()
ud_lt["pct_win"] = round(ud_lt["pct_win"] * 100, 2)
print_tabulate(ud_lt)


print("\n\n--- POR EQUIPO ---")

print("\n=== top 15 equipos con mas goles de local ===")
hg = df.groupby("HomeTeam").agg(
    partidos=("FTHG", "count"),
    total=("FTHG", "sum"),
    avg=("FTHG", "mean"),
    max_en_un_partido=("FTHG", "max")
).round(3).sort_values("total", ascending=False).head(15).reset_index()
print_tabulate(hg)

print("\n=== top 15 equipos con mas goles de visitante ===")
ag = df.groupby("AwayTeam").agg(
    partidos=("FTAG", "count"),
    total=("FTAG", "sum"),
    avg=("FTAG", "mean"),
    max_en_un_partido=("FTAG", "max")
).round(3).sort_values("total", ascending=False).head(15).reset_index()
print_tabulate(ag)

print("\n=== equipos con mayor % victorias local (min 50 partidos) ===")
hw = df.groupby("HomeTeam").agg(partidos=("home_win", "count"), victorias=("home_win", "sum"))
hw = hw[hw["partidos"] >= 50]
hw["pct_win"] = round(hw["victorias"] / hw["partidos"] * 100, 2)
hw = hw.sort_values("pct_win", ascending=False).head(15).reset_index()
print_tabulate(hw)

print("\n=== equipos con mayor % victorias visitante (min 50 partidos) ===")
aw = df.groupby("AwayTeam").agg(partidos=("away_win", "count"), victorias=("away_win", "sum"))
aw = aw[aw["partidos"] >= 50]
aw["pct_win"] = round(aw["victorias"] / aw["partidos"] * 100, 2)
aw = aw.sort_values("pct_win", ascending=False).head(15).reset_index()
print_tabulate(aw)

print("\n=== equipos con mas empates (min 50 partidos) ===")
dr = df.groupby("HomeTeam").agg(partidos=("draw", "count"), empates=("draw", "sum"))
dr = dr[dr["partidos"] >= 50]
dr["pct_draw"] = round(dr["empates"] / dr["partidos"] * 100, 2)
dr = dr.sort_values("pct_draw", ascending=False).head(15).reset_index()
print_tabulate(dr)

print("\n=== equipos con odds promedio mas altas como local (favoritos) ===")
fav_h = df.groupby("HomeTeam")["AvgH"].agg(["mean", "count"]).reset_index()
fav_h = fav_h[fav_h["count"] >= 50].sort_values("mean").head(10)
print_tabulate(fav_h)

print("\n=== equipos mas underdogs como visitante (AvgA > 4, min 30 partidos) ===")
ud_team = df[df["is_underdog_away"] == 1].groupby("AwayTeam").agg(
    veces_underdog=("away_win", "count"),
    gana=("away_win", "sum"),
    avg_odd=("AvgA", "mean")
)
ud_team = ud_team[ud_team["veces_underdog"] >= 30]
ud_team["pct_win"] = round(ud_team["gana"] / ud_team["veces_underdog"] * 100, 2)
ud_team = ud_team.sort_values("pct_win", ascending=False).head(15).reset_index()
print_tabulate(ud_team)


print("\n\n--- ANALISIS DE CUOTAS ---")

print("\n=== partidos con odds mas extremas (favorito local) ===")
print_tabulate(df[["Div", "Season", "Date", "HomeTeam", "AwayTeam", "AvgH", "AvgA", "FTR", "FTHG", "FTAG"]].sort_values("AvgH").head(15))

print("\n=== partidos con odds mas extremas (favorito visitante) ===")
print_tabulate(df[["Div", "Season", "Date", "HomeTeam", "AwayTeam", "AvgH", "AvgA", "FTR", "FTHG", "FTAG"]].sort_values("AvgA").head(15))

print("\n=== underdogs extremos visitante (AvgA > 10) que ganaron ===")
ude = df[(df["AvgA"] > 10) & (df["away_win"] == 1)]
print(f"total: {len(ude)}")
print_tabulate(ude[["Div", "Season", "Date", "HomeTeam", "AwayTeam", "AvgH", "AvgA", "FTHG", "FTAG"]].sort_values("AvgA", ascending=False))

print("\n=== mayor movimiento de mercado hacia local (smart money local) ===")
print_tabulate(df[["Div", "Date", "HomeTeam", "AwayTeam", "AvgH", "AvgCH", "odds_move_H", "FTR"]].sort_values("odds_move_H").head(15))

print("\n=== mayor movimiento de mercado hacia visitante (smart money visitante) ===")
print_tabulate(df[["Div", "Date", "HomeTeam", "AwayTeam", "AvgA", "AvgCA", "odds_move_A", "FTR"]].sort_values("odds_move_A", ascending=False).head(15))

print("\n=== cuando el mercado mueve a favor del local, gana? ===")
smart_h = df[df["odds_move_H"] < -0.1]
print(f"partidos con movimiento significativo hacia local: {len(smart_h)}")
print(f"gana local: {smart_h['home_win'].sum()} ({round(smart_h['home_win'].mean()*100,2)}%)")
print(f"empate: {smart_h['draw'].sum()} ({round(smart_h['draw'].mean()*100,2)}%)")
print(f"gana visitante: {smart_h['away_win'].sum()} ({round(smart_h['away_win'].mean()*100,2)}%)")

print("\n=== cuando el mercado mueve a favor del visitante, gana? ===")
smart_a = df[df["odds_move_A"] > 0.1]
print(f"partidos con movimiento significativo hacia visitante: {len(smart_a)}")
print(f"gana local: {smart_a['home_win'].sum()} ({round(smart_a['home_win'].mean()*100,2)}%)")
print(f"empate: {smart_a['draw'].sum()} ({round(smart_a['draw'].mean()*100,2)}%)")
print(f"gana visitante: {smart_a['away_win'].sum()} ({round(smart_a['away_win'].mean()*100,2)}%)")

print("\n=== correlacion entre overround y total de goles ===")
print(round(df[["overround", "total_goals", "AvgH", "AvgA", "AvgD", "odds_move_H", "odds_move_A"]].corr(), 3))

print("\n\n--- GRAFICAS ---")

scatter_group_by("img/odds_apertura_por_liga.png", df, "AvgH", "AvgA", "Div")
scatter_group_by("img/goles_por_liga.png", df, "FTHG", "FTAG", "Div")
scatter_group_by("img/odds_apertura_vs_cierre_H.png", df, "AvgH", "AvgCH", "Div")
scatter_group_by("img/odds_apertura_vs_cierre_A.png", df, "AvgA", "AvgCA", "Div")
scatter_group_by("img/implied_prob_por_liga.png", df, "imp_prob_H", "imp_prob_A", "Div")
scatter_group_by("img/movimiento_mercado.png", df, "odds_move_H", "odds_move_A", "Div")

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
plt.savefig("img/goles_temporada_liga.png")
plt.close()

# resultados pie por liga
fig, axes = plt.subplots(1, 5, figsize=(20, 5))
for i, liga in enumerate(ligas):
    sub = df[df["Div"] == liga]["FTR"].value_counts()
    axes[i].pie(sub.values, labels=sub.index, autopct="%1.1f%%")
    axes[i].set_title(liga)
plt.savefig("img/resultados_pie_por_liga.png")
plt.close()

# btts y over25 por liga
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(ligas))
btts_vals = [df[df["Div"] == l]["btts"].mean() * 100 for l in ligas]
over25_vals = [df[df["Div"] == l]["over25"].mean() * 100 for l in ligas]
over35_vals = [df[df["Div"] == l]["over35"].mean() * 100 for l in ligas]
ax.bar(x - 0.25, btts_vals, 0.25, label="btts")
ax.bar(x, over25_vals, 0.25, label="over 2.5")
ax.bar(x + 0.25, over35_vals, 0.25, label="over 3.5")
ax.set_xticks(x)
ax.set_xticklabels(ligas)
ax.set_ylabel("%")
ax.legend()
plt.savefig("img/btts_over_por_liga.png")
plt.close()

# distribucion de goles totales
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df["total_goals"], bins=range(0, 15), edgecolor="black", align="left")
ax.set_xlabel("goles en el partido")
ax.set_ylabel("frecuencia")
plt.savefig("img/distribucion_goles_totales.png")
plt.close()

# odds de underdogs visitantes que ganaron vs que perdieron
fig, ax = plt.subplots(figsize=(10, 5))
ud_win = df[(df["is_underdog_away"] == 1) & (df["away_win"] == 1)]["AvgA"]
ud_lose = df[(df["is_underdog_away"] == 1) & (df["away_win"] == 0)]["AvgA"]
ax.hist(ud_win, bins=30, alpha=0.6, label="underdog gana")
ax.hist(ud_lose, bins=30, alpha=0.6, label="underdog pierde")
ax.set_xlabel("AvgA")
ax.legend()
plt.savefig("img/underdog_win_vs_lose.png")
plt.close()

print("guardadas imagenes en img/")