import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import os
from scipy.stats import gaussian_kde

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
DAY_NAMES   = {0:"Lunes",1:"Martes",2:"Miercoles",3:"Jueves",4:"Viernes",5:"Sabado",6:"Domingo"}
DAY_ORDER   = ["Lunes","Martes","Miercoles","Jueves","Viernes","Sabado","Domingo"]
SEASON_MAP  = {1920:"2019/20",2021:"2020/21",2122:"2021/22",
               2223:"2022/23",2324:"2023/24",2425:"2024/25",2526:"2025/26"}

# carga del csv y construccion de variables derivadas
df = pd.read_csv("../Practica 1/data/clean/football_clean.csv", parse_dates=["Date"])

df["total_goals"]       = df["FTHG"] + df["FTAG"]
df["ht_goals"]          = df["HTHG"] + df["HTAG"]
df["second_half_goals"] = df["total_goals"] - df["ht_goals"]
df["goal_diff"]         = df["FTHG"] - df["FTAG"]
df["home_win"]          = (df["FTR"] == "H").astype(int)
df["draw"]              = (df["FTR"] == "D").astype(int)
df["away_win"]          = (df["FTR"] == "A").astype(int)
df["btts"]              = ((df["FTHG"] > 0) & (df["FTAG"] > 0)).astype(int)
df["over15"]            = (df["total_goals"] > 1).astype(int)
df["over25"]            = (df["total_goals"] > 2).astype(int)
df["over35"]            = (df["total_goals"] > 3).astype(int)
df["goalless"]          = (df["total_goals"] == 0).astype(int)
df["high_scoring"]      = (df["total_goals"] >= 5).astype(int)
df["clean_sheet_h"]     = (df["FTAG"] == 0).astype(int)
df["clean_sheet_a"]     = (df["FTHG"] == 0).astype(int)
df["imp_prob_H"]        = round(1 / df["AvgH"], 4)
df["imp_prob_D"]        = round(1 / df["AvgD"], 4)
df["imp_prob_A"]        = round(1 / df["AvgA"], 4)
df["overround"]         = round(df["imp_prob_H"] + df["imp_prob_D"] + df["imp_prob_A"], 4)
df["odds_move_H"]       = round(df["AvgCH"] - df["AvgH"], 4)
df["odds_move_A"]       = round(df["AvgCA"] - df["AvgA"], 4)
df["odds_move_D"]       = round(df["AvgCD"] - df["AvgD"], 4)
df["month"]             = df["Date"].dt.month
df["dayofweek"]         = df["Date"].dt.dayofweek
df["day_name"]          = df["dayofweek"].map(DAY_NAMES)
df["Season_label"]      = df["Season"].map(SEASON_MAP)

ligas    = sorted(df["Div"].unique())
seasons  = sorted(df["Season"].unique())
slabels  = [SEASON_MAP[s] for s in seasons]

def cmap_n(n, name="tab10"):
    return matplotlib.colormaps.get_cmap(name).resampled(n)

print("cargado:", len(df), "partidos")

# distribuciones univariadas: histogramas, KDE y violins

# goles totales por liga con curva de densidad encima
fig, axes = plt.subplots(1, 5, figsize=(22, 5), sharey=False)
for i, liga in enumerate(ligas):
    sub = df[df["Div"] == liga]["total_goals"]
    bins = range(0, int(sub.max()) + 2)
    axes[i].hist(sub, bins=bins, color=LIGA_COLORS[liga], alpha=0.75,
                 edgecolor="white", density=True, align="left")
    kde_x = np.linspace(0, sub.max(), 200)
    kde   = gaussian_kde(sub, bw_method=0.4)
    axes[i].plot(kde_x, kde(kde_x), color="black", linewidth=1.5, linestyle="--")
    axes[i].set_title(LIGAS_NAME[liga], fontsize=9, fontweight="bold")
    axes[i].set_xlabel("goles")
    axes[i].set_ylabel("densidad" if i == 0 else "")
    axes[i].axvline(sub.mean(), color="#e74c3c", linewidth=1.2, linestyle=":")
    axes[i].text(sub.mean() + 0.1, axes[i].get_ylim()[1] * 0.9,
                 f"μ={sub.mean():.2f}", fontsize=7, color="#e74c3c")
fig.suptitle("Distribucion de goles totales por liga (densidad + KDE)", fontweight="bold")
plt.tight_layout()
plt.savefig("img/hist_goles_por_liga.png", dpi=130)
plt.close()
print("  1.1 hist_goles_por_liga")

# primer tiempo vs segundo tiempo superpuestos en histograma
fig, axes = plt.subplots(1, 5, figsize=(22, 5))
for i, liga in enumerate(ligas):
    sub = df[df["Div"] == liga]
    axes[i].hist(sub["ht_goals"],          bins=range(0, 8), alpha=0.6, color="#3498db",
                 edgecolor="white", align="left", label="HT", density=True)
    axes[i].hist(sub["second_half_goals"], bins=range(0, 8), alpha=0.6, color="#e74c3c",
                 edgecolor="white", align="left", label="ST", density=True)
    axes[i].set_title(LIGAS_NAME[liga], fontsize=9, fontweight="bold")
    axes[i].set_xlabel("goles"); axes[i].legend(fontsize=7)
fig.suptitle("Distribucion de goles HT vs ST por liga", fontweight="bold")
plt.tight_layout()
plt.savefig("img/hist_ht_vs_st_por_liga.png", dpi=130)
plt.close()
print("  1.2 hist_ht_vs_st_por_liga")

# distribucion de las tres cuotas usando KDE, una curva por liga
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, (col, titulo, color) in enumerate([
    ("AvgH", "Cuota local (AvgH)",    "#3498db"),
    ("AvgA", "Cuota visitante (AvgA)","#e74c3c"),
    ("AvgD", "Cuota empate (AvgD)",   "#9b59b6"),
]):
    for liga in ligas:
        sub = df[df["Div"] == liga][col].dropna()
        kde_x = np.linspace(sub.min(), min(sub.max(), 20), 300)
        kde   = gaussian_kde(sub, bw_method=0.3)
        axes[i].plot(kde_x, kde(kde_x), label=liga, color=LIGA_COLORS[liga], linewidth=1.8)
    axes[i].set_title(titulo, fontweight="bold")
    axes[i].set_xlabel("cuota"); axes[i].set_ylabel("densidad" if i == 0 else "")
    axes[i].legend(fontsize=8); axes[i].set_xlim(1, 15)
fig.suptitle("Distribucion de cuotas por liga (KDE)", fontweight="bold")
plt.tight_layout()
plt.savefig("img/hist_cuotas_kde.png", dpi=130)
plt.close()
print("  1.3 hist_cuotas_kde")

# violin del overround para ver donde se concentra el margen de cada casa
fig, ax = plt.subplots(figsize=(11, 6))
data_vio = [df[df["Div"] == l]["overround"].dropna().values for l in ligas]
parts = ax.violinplot(data_vio, positions=range(len(ligas)), showmedians=True,
                      showextrema=True)
for j, pc in enumerate(parts["bodies"]):
    pc.set_facecolor(list(LIGA_COLORS.values())[j])
    pc.set_alpha(0.7)
ax.set_xticks(range(len(ligas)))
ax.set_xticklabels([LIGAS_NAME[l] for l in ligas])
ax.set_ylabel("overround"); ax.set_title("Distribucion del overround por liga", fontweight="bold")
plt.tight_layout()
plt.savefig("img/violin_overround_por_liga.png", dpi=130)
plt.close()
print("  1.4 violin_overround_por_liga")

# diferencia de goles por partido: cuantos se ganan por 1, por 2, por 3...
fig, axes = plt.subplots(1, 5, figsize=(22, 5))
for i, liga in enumerate(ligas):
    sub = df[df["Div"] == liga]["goal_diff"]
    axes[i].hist(sub, bins=range(int(sub.min()) - 1, int(sub.max()) + 2),
                 color=LIGA_COLORS[liga], alpha=0.8, edgecolor="white", align="left")
    axes[i].axvline(0, color="black", linewidth=1, linestyle="--")
    axes[i].set_title(LIGAS_NAME[liga], fontsize=9, fontweight="bold")
    axes[i].set_xlabel("diferencia goles (local - visitante)")
fig.suptitle("Distribucion de diferencia de goles (resultado) por liga", fontweight="bold")
plt.tight_layout()
plt.savefig("img/hist_goal_diff_por_liga.png", dpi=130)
plt.close()
print("  1.5 hist_goal_diff_por_liga")

# boxplots para ver dispersion y outliers en goles, cuotas y movimiento de mercado

# goles totales: comparacion de dispersion entre ligas
fig, ax = plt.subplots(figsize=(11, 6))
data_bp = [df[df["Div"] == l]["total_goals"].values for l in ligas]
bp = ax.boxplot(data_bp, patch_artist=True, notch=False, vert=True)
for j, patch in enumerate(bp["boxes"]):
    patch.set_facecolor(list(LIGA_COLORS.values())[j])
    patch.set_alpha(0.7)
ax.set_xticklabels([LIGAS_NAME[l] for l in ligas])
ax.set_ylabel("goles"); ax.set_title("Goles totales por partido — boxplot por liga", fontweight="bold")
plt.tight_layout()
plt.savefig("img/boxplot_goles_por_liga.png", dpi=130)
plt.close()
print("  2.1 boxplot_goles_por_liga")

# las tres cuotas juntas para ver si una liga tiene mercados mas cerrados
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for i, (col, titulo) in enumerate([
    ("AvgH","Cuota local (apertura)"),
    ("AvgA","Cuota visitante (apertura)"),
    ("AvgD","Cuota empate (apertura)"),
]):
    data_bp2 = [df[df["Div"] == l][col].dropna().values for l in ligas]
    bp2 = axes[i].boxplot(data_bp2, patch_artist=True, vert=True)
    for j, patch in enumerate(bp2["boxes"]):
        patch.set_facecolor(list(LIGA_COLORS.values())[j])
        patch.set_alpha(0.7)
    axes[i].set_xticklabels([l for l in ligas], rotation=20)
    axes[i].set_title(titulo, fontweight="bold")
    axes[i].set_ylabel("cuota")
fig.suptitle("Boxplot de cuotas por liga", fontweight="bold")
plt.tight_layout()
plt.savefig("img/boxplot_cuotas_por_liga.png", dpi=130)
plt.close()
print("  2.2 boxplot_cuotas_por_liga")

# goles segun el dia de la semana en que se jugó
fig, ax = plt.subplots(figsize=(12, 6))
days_present = [d for d in DAY_ORDER if d in df["day_name"].values]
data_day = [df[df["day_name"] == d]["total_goals"].values for d in days_present]
bp3 = ax.boxplot(data_day, patch_artist=True)
cmap_d = cmap_n(len(days_present), "Set2")
for j, patch in enumerate(bp3["boxes"]):
    patch.set_facecolor(cmap_d(j)); patch.set_alpha(0.75)
ax.set_xticklabels(days_present)
ax.set_ylabel("goles"); ax.set_title("Goles por partido segun dia de la semana", fontweight="bold")
plt.tight_layout()
plt.savefig("img/boxplot_goles_dia_semana.png", dpi=130)
plt.close()
print("  2.3 boxplot_goles_dia_semana")

# overround por temporada para ver si el margen de las casas sube o baja con los años
fig, ax = plt.subplots(figsize=(13, 6))
data_or = [df[df["Season"] == s]["overround"].dropna().values for s in seasons]
bp4 = ax.boxplot(data_or, patch_artist=True)
cmap_s = cmap_n(len(seasons), "coolwarm")
for j, patch in enumerate(bp4["boxes"]):
    patch.set_facecolor(cmap_s(j)); patch.set_alpha(0.75)
ax.set_xticklabels(slabels, rotation=20)
ax.set_ylabel("overround"); ax.set_title("Evolucion del overround por temporada", fontweight="bold")
plt.tight_layout()
plt.savefig("img/boxplot_overround_temporada.png", dpi=130)
plt.close()
print("  2.4 boxplot_overround_temporada")

# movimiento de cuota local entre apertura y cierre, cero es la linea de referencia
fig, ax = plt.subplots(figsize=(11, 6))
data_mov = [df[df["Div"] == l]["odds_move_H"].dropna().values for l in ligas]
bp5 = ax.boxplot(data_mov, patch_artist=True)
for j, patch in enumerate(bp5["boxes"]):
    patch.set_facecolor(list(LIGA_COLORS.values())[j]); patch.set_alpha(0.7)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xticklabels([LIGAS_NAME[l] for l in ligas])
ax.set_ylabel("movimiento (cierre - apertura)")
ax.set_title("Movimiento de cuota local (apertura → cierre) por liga", fontweight="bold")
plt.tight_layout()
plt.savefig("img/boxplot_movimiento_cuota_H.png", dpi=130)
plt.close()
print("  2.5 boxplot_movimiento_cuota_H")

# barras, pies y marcadores: como se distribuyen resultados y goles

# pie de resultados finales, una rueda por liga
fig, axes = plt.subplots(1, 5, figsize=(22, 5))
for i, liga in enumerate(ligas):
    sub = df[df["Div"] == liga]["FTR"].value_counts()
    axes[i].pie(
        [sub.get("H", 0), sub.get("D", 0), sub.get("A", 0)],
        labels=["Local (H)", "Empate (D)", "Visitante (A)"],
        colors=["#3498db", "#95a5a6", "#e74c3c"],
        autopct="%1.1f%%", startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    axes[i].set_title(LIGAS_NAME[liga], fontsize=10, fontweight="bold")
fig.suptitle("Distribucion de resultados FT por liga", fontweight="bold", fontsize=13)
plt.tight_layout()
plt.savefig("img/pie_resultados_ft_por_liga.png", dpi=130)
plt.close()
print("  3.1 pie_resultados_ft_por_liga")

# lo mismo pero al descanso para comparar si el HT cambia mucho vs el FT
fig, axes = plt.subplots(1, 5, figsize=(22, 5))
for i, liga in enumerate(ligas):
    sub = df[df["Div"] == liga]["HTR"].value_counts()
    axes[i].pie(
        [sub.get("H", 0), sub.get("D", 0), sub.get("A", 0)],
        labels=["Local (H)", "Empate (D)", "Visitante (A)"],
        colors=["#3498db", "#95a5a6", "#e74c3c"],
        autopct="%1.1f%%", startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    axes[i].set_title(LIGAS_NAME[liga], fontsize=10, fontweight="bold")
fig.suptitle("Distribucion de resultados HT por liga", fontweight="bold", fontsize=13)
plt.tight_layout()
plt.savefig("img/pie_resultados_ht_por_liga.png", dpi=130)
plt.close()
print("  3.2 pie_resultados_ht_por_liga")

# barras agrupadas con todos los indicadores booleanos: btts, overs, clean sheets...
flags      = ["over15", "over25", "over35", "btts", "goalless", "high_scoring", "clean_sheet_h"]
flag_names = ["over1.5", "over2.5", "over3.5", "btts", "0-0", "5+ goles", "clean sheet local"]
flag_cols  = ["#1abc9c","#3498db","#9b59b6","#e67e22","#95a5a6","#e74c3c","#2ecc71"]

fig, ax = plt.subplots(figsize=(13, 6))
x = np.arange(len(ligas)); w = 0.11
for j, (flag, name, color) in enumerate(zip(flags, flag_names, flag_cols)):
    vals = [df[df["Div"] == l][flag].mean() * 100 for l in ligas]
    ax.bar(x + j * w, vals, w, label=name, color=color, alpha=0.85)
ax.set_xticks(x + w * 3)
ax.set_xticklabels([LIGAS_NAME[l] for l in ligas])
ax.set_ylabel("%"); ax.legend(loc="upper right", fontsize=8)
ax.set_title("Indicadores de goles y resultados por liga (%)", fontweight="bold")
plt.tight_layout()
plt.savefig("img/barras_flags_por_liga.png", dpi=130)
plt.close()
print("  3.3 barras_flags_por_liga")

# evolucion de los tres resultados posibles temporada a temporada
fig, ax = plt.subplots(figsize=(14, 6))
x  = np.arange(len(seasons)); w = 0.28
pH = [df[df["Season"] == s]["home_win"].mean() * 100 for s in seasons]
pD = [df[df["Season"] == s]["draw"].mean()     * 100 for s in seasons]
pA = [df[df["Season"] == s]["away_win"].mean() * 100 for s in seasons]
ax.bar(x - w, pH, w, label="% Local",    color="#3498db", alpha=0.85)
ax.bar(x,     pD, w, label="% Empate",   color="#95a5a6", alpha=0.85)
ax.bar(x + w, pA, w, label="% Visitante",color="#e74c3c", alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(slabels, rotation=20)
ax.set_ylabel("%"); ax.legend()
ax.set_title("Evolucion de resultados por temporada", fontweight="bold")
plt.tight_layout()
plt.savefig("img/barras_resultados_por_temporada.png", dpi=130)
plt.close()
print("  3.4 barras_resultados_por_temporada")

# marcadores exactos mas frecuentes: azul gana local, rojo gana visitante, gris empate
df["score_ft"] = df["FTHG"].astype(str) + "-" + df["FTAG"].astype(str)
top15 = df["score_ft"].value_counts().head(15)
fig, ax = plt.subplots(figsize=(10, 7))
colors_15 = ["#3498db" if int(s.split("-")[0]) > int(s.split("-")[1])
             else "#e74c3c" if int(s.split("-")[0]) < int(s.split("-")[1])
             else "#95a5a6" for s in top15.index]
ax.barh(top15.index[::-1], top15.values[::-1], color=colors_15[::-1], alpha=0.85)
ax.set_xlabel("frecuencia")
ax.set_title("Top 15 marcadores exactos FT (azul=local, rojo=visitante, gris=empate)",
             fontweight="bold")
for i, v in enumerate(top15.values[::-1]):
    ax.text(v + 5, i, f"{v} ({round(v/len(df)*100,1)}%)", va="center", fontsize=8)
plt.tight_layout()
plt.savefig("img/barras_marcadores_exactos_ft.png", dpi=130)
plt.close()
print("  3.5 barras_marcadores_exactos_ft")

# misma logica pero para los marcadores al descanso
df["score_ht"] = df["HTHG"].astype(str) + "-" + df["HTAG"].astype(str)
top10_ht = df["score_ht"].value_counts().head(10)
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(top10_ht.index[::-1], top10_ht.values[::-1], color="#9b59b6", alpha=0.85)
ax.set_xlabel("frecuencia")
ax.set_title("Top 10 marcadores exactos al descanso (HT)", fontweight="bold")
for i, v in enumerate(top10_ht.values[::-1]):
    ax.text(v + 5, i, f"{v} ({round(v/len(df)*100,1)}%)", va="center", fontsize=8)
plt.tight_layout()
plt.savefig("img/barras_marcadores_exactos_ht.png", dpi=130)
plt.close()
print("  3.6 barras_marcadores_exactos_ht")

# primer tiempo vs segundo tiempo por temporada dentro de cada liga
fig, axes = plt.subplots(1, 5, figsize=(22, 5), sharey=True)
for i, liga in enumerate(ligas):
    sub   = df[df["Div"] == liga]
    ht_by = sub.groupby("Season_label")["ht_goals"].mean()
    st_by = sub.groupby("Season_label")["second_half_goals"].mean()
    common = [s for s in slabels if s in ht_by.index]
    x2 = np.arange(len(common)); w2 = 0.35
    axes[i].bar(x2 - w2/2, [ht_by[s] for s in common], w2, label="HT", color="#3498db", alpha=0.8)
    axes[i].bar(x2 + w2/2, [st_by[s] for s in common], w2, label="ST", color="#e74c3c", alpha=0.8)
    axes[i].set_xticks(x2); axes[i].set_xticklabels(common, rotation=45, fontsize=6)
    axes[i].set_title(LIGAS_NAME[liga], fontsize=9, fontweight="bold")
    axes[i].legend(fontsize=7)
    if i == 0: axes[i].set_ylabel("avg goles")
fig.suptitle("Avg goles primer vs segundo tiempo por liga y temporada", fontweight="bold")
plt.tight_layout()
plt.savefig("img/barras_ht_vs_st_por_liga_temporada.png", dpi=130)
plt.close()
print("  3.7 barras_ht_vs_st_por_liga_temporada")

# como evolucionan over y btts temporada a temporada en todas las ligas juntas
fig, ax = plt.subplots(figsize=(14, 6))
x  = np.arange(len(seasons))
o15 = [df[df["Season"] == s]["over15"].mean() * 100 for s in seasons]
o25 = [df[df["Season"] == s]["over25"].mean() * 100 for s in seasons]
o35 = [df[df["Season"] == s]["over35"].mean() * 100 for s in seasons]
btt = [df[df["Season"] == s]["btts"].mean()   * 100 for s in seasons]
ax.plot(slabels, o15, marker="o", label="over1.5", color="#1abc9c", linewidth=2)
ax.plot(slabels, o25, marker="s", label="over2.5", color="#3498db", linewidth=2)
ax.plot(slabels, o35, marker="^", label="over3.5", color="#9b59b6", linewidth=2)
ax.plot(slabels, btt, marker="D", label="btts",    color="#e67e22", linewidth=2)
ax.set_ylabel("%"); ax.legend(); ax.set_ylim(0, 100)
ax.set_title("Evolucion de over/btts por temporada", fontweight="bold")
plt.xticks(rotation=20); plt.tight_layout()
plt.savefig("img/lineas_over_btts_temporada.png", dpi=130)
plt.close()
print("  3.8 lineas_over_btts_temporada")

# series temporales: tendencias a lo largo de las temporadas

# promedio de goles por temporada, una linea por liga
fig, ax = plt.subplots(figsize=(12, 6))
for liga in ligas:
    sub = df[df["Div"] == liga].groupby("Season_label")["total_goals"].mean()
    common = [s for s in slabels if s in sub.index]
    ax.plot(common, [sub[s] for s in common],
            marker="o", label=LIGAS_NAME[liga], color=LIGA_COLORS[liga], linewidth=2)
ax.set_ylabel("avg goles por partido"); ax.legend()
ax.set_title("Evolucion del promedio de goles por temporada y liga", fontweight="bold")
plt.xticks(rotation=20); plt.tight_layout()
plt.savefig("img/lineas_goles_por_temporada_liga.png", dpi=130)
plt.close()
print("  4.1 lineas_goles_por_temporada_liga")

# victorias locales por temporada: muestra si la ventaja de local se esta perdiendo
fig, ax = plt.subplots(figsize=(12, 6))
for liga in ligas:
    sub = df[df["Div"] == liga].groupby("Season_label")["home_win"].mean() * 100
    common = [s for s in slabels if s in sub.index]
    ax.plot(common, [sub[s] for s in common],
            marker="o", label=LIGAS_NAME[liga], color=LIGA_COLORS[liga], linewidth=2)
ax.set_ylabel("% victorias local"); ax.legend()
ax.set_title("Evolucion del % victorias local por temporada y liga", fontweight="bold")
plt.xticks(rotation=20); plt.tight_layout()
plt.savefig("img/lineas_pct_local_por_temporada.png", dpi=130)
plt.close()
print("  4.2 lineas_pct_local_por_temporada")

# overround: si sube con los años significa que las casas estan aumentando su margen
fig, ax = plt.subplots(figsize=(12, 6))
for liga in ligas:
    sub = df[df["Div"] == liga].groupby("Season_label")["overround"].mean()
    common = [s for s in slabels if s in sub.index]
    ax.plot(common, [sub[s] for s in common],
            marker="s", label=LIGAS_NAME[liga], color=LIGA_COLORS[liga], linewidth=2)
ax.set_ylabel("overround promedio"); ax.legend()
ax.set_title("Evolucion del overround (margen de casa) por temporada y liga", fontweight="bold")
plt.xticks(rotation=20); plt.tight_layout()
plt.savefig("img/lineas_overround_por_temporada.png", dpi=130)
plt.close()
print("  4.3 lineas_overround_por_temporada")

# btts por temporada: si sube indica partidos mas abiertos en general
fig, ax = plt.subplots(figsize=(12, 6))
for liga in ligas:
    sub = df[df["Div"] == liga].groupby("Season_label")["btts"].mean() * 100
    common = [s for s in slabels if s in sub.index]
    ax.plot(common, [sub[s] for s in common],
            marker="^", label=LIGAS_NAME[liga], color=LIGA_COLORS[liga], linewidth=2)
ax.set_ylabel("% btts"); ax.legend()
ax.set_title("Evolucion del % BTTS por temporada y liga", fontweight="bold")
plt.xticks(rotation=20); plt.tight_layout()
plt.savefig("img/lineas_btts_por_temporada.png", dpi=130)
plt.close()
print("  4.4 lineas_btts_por_temporada")

# goles por mes del año empezando en agosto que es cuando arranca la temporada
fig, ax = plt.subplots(figsize=(12, 5))
month_names = {8:"Ago",9:"Sep",10:"Oct",11:"Nov",12:"Dic",1:"Ene",2:"Feb",3:"Mar",4:"Abr",5:"May",6:"Jun",7:"Jul"}
for liga in ligas:
    sub = df[df["Div"] == liga].groupby("month")["total_goals"].mean()
    ordered_months = sorted(sub.index, key=lambda m: (m < 7, m))
    ax.plot([month_names.get(m, m) for m in ordered_months],
            [sub[m] for m in ordered_months],
            marker="o", label=LIGAS_NAME[liga], color=LIGA_COLORS[liga], linewidth=1.8)
ax.set_ylabel("avg goles"); ax.legend(fontsize=8)
ax.set_title("Avg goles por mes del año por liga", fontweight="bold")
plt.tight_layout()
plt.savefig("img/lineas_goles_por_mes.png", dpi=130)
plt.close()
print("  4.5 lineas_goles_por_mes")

# ratio local/visitante: si baja de 1 significa que el visitante ya marca igual o mas
fig, ax = plt.subplots(figsize=(12, 6))
for liga in ligas:
    sub = df[df["Div"] == liga].groupby("Season_label").apply(
        lambda x: x["FTHG"].sum() / x["FTAG"].sum() if x["FTAG"].sum() > 0 else np.nan
    )
    common = [s for s in slabels if s in sub.index]
    ax.plot(common, [sub[s] for s in common],
            marker="D", label=LIGAS_NAME[liga], color=LIGA_COLORS[liga], linewidth=2)
ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
ax.set_ylabel("ratio goles local / visitante"); ax.legend()
ax.set_title("Decaimiento de la ventaja local: ratio goles loc/vis por temporada", fontweight="bold")
plt.xticks(rotation=20); plt.tight_layout()
plt.savefig("img/lineas_ratio_local_visitante.png", dpi=130)
plt.close()
print("  4.6 lineas_ratio_local_visitante")

# scatter plots para relaciones entre variables y comportamiento del mercado

# cuota local apertura vs cierre: puntos sobre la diagonal = cuota subio al cerrar
fig, ax = plt.subplots(figsize=(9, 7))
for liga in ligas:
    sub = df[df["Div"] == liga].sample(min(600, len(df[df["Div"] == liga])), random_state=42)
    ax.scatter(sub["AvgH"], sub["AvgCH"], alpha=0.25, s=8,
               color=LIGA_COLORS[liga], label=liga)
lim_min = min(df["AvgH"].min(), df["AvgCH"].min())
lim_max = min(max(df["AvgH"].max(), df["AvgCH"].max()), 15)
ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", linewidth=0.8, alpha=0.5)
ax.set_xlabel("AvgH apertura"); ax.set_ylabel("AvgCH cierre")
ax.set_title("Cuota local: apertura vs cierre por liga", fontweight="bold")
ax.legend(fontsize=8); ax.set_xlim(1, 12); ax.set_ylim(1, 12)
patches = [mpatches.Patch(color=LIGA_COLORS[l], label=LIGAS_NAME[l]) for l in ligas]
ax.legend(handles=patches, fontsize=8)
plt.tight_layout()
plt.savefig("img/scatter_apertura_vs_cierre_H.png", dpi=130)
plt.close()
print("  5.1 scatter_apertura_vs_cierre_H")

# lo mismo para la cuota del visitante
fig, ax = plt.subplots(figsize=(9, 7))
for liga in ligas:
    sub = df[df["Div"] == liga].sample(min(600, len(df[df["Div"] == liga])), random_state=42)
    ax.scatter(sub["AvgA"], sub["AvgCA"], alpha=0.25, s=8, color=LIGA_COLORS[liga])
ax.plot([1, 20], [1, 20], "k--", linewidth=0.8, alpha=0.5)
ax.set_xlabel("AvgA apertura"); ax.set_ylabel("AvgCA cierre")
ax.set_title("Cuota visitante: apertura vs cierre por liga", fontweight="bold")
ax.set_xlim(1, 20); ax.set_ylim(1, 20)
patches = [mpatches.Patch(color=LIGA_COLORS[l], label=LIGAS_NAME[l]) for l in ligas]
ax.legend(handles=patches, fontsize=8)
plt.tight_layout()
plt.savefig("img/scatter_apertura_vs_cierre_A.png", dpi=130)
plt.close()
print("  5.2 scatter_apertura_vs_cierre_A")

# cada partido como un punto: goles local en x, visitante en y, color segun quien gano
fig, ax = plt.subplots(figsize=(9, 7))
colors_ftr = {"H": "#3498db", "D": "#95a5a6", "A": "#e74c3c"}
for ftr in ["H", "D", "A"]:
    sub = df[df["FTR"] == ftr]
    ax.scatter(sub["FTHG"], sub["FTAG"], alpha=0.15, s=10,
               color=colors_ftr[ftr], label={"H":"Local gana","D":"Empate","A":"Visitante gana"}[ftr])
ax.set_xlabel("goles local"); ax.set_ylabel("goles visitante")
ax.set_title("Goles local vs visitante coloreado por resultado", fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig("img/scatter_goles_local_vs_visitante.png", dpi=130)
plt.close()
print("  5.3 scatter_goles_local_vs_visitante")

# calibracion del mercado: cada burbuja es un rango de cuota, el tamaño es la cantidad de partidos
bins   = [1.0, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 6.0, 25.0]
labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)]
d2     = df.copy()
d2["rng"] = pd.cut(d2["AvgH"], bins=bins, labels=labels, right=False)
pts = []
for rng in labels:
    sub = d2[d2["rng"] == rng]
    if len(sub) < 5: continue
    pts.append({"rng": rng, "imp": 1 / sub["AvgH"].mean() * 100,
                "real": sub["home_win"].mean() * 100, "n": len(sub)})
pts_df = pd.DataFrame(pts)
fig, ax = plt.subplots(figsize=(10, 6))
sc = ax.scatter(pts_df["imp"], pts_df["real"],
                s=pts_df["n"] / 10, c=pts_df["real"] - pts_df["imp"],
                cmap="RdYlGn", alpha=0.9, edgecolors="gray", linewidth=0.5)
for _, row in pts_df.iterrows():
    ax.annotate(row["rng"], (row["imp"], row["real"]), fontsize=7, alpha=0.8)
ax.plot([0, 80], [0, 80], "k--", linewidth=0.8)
ax.set_xlabel("probabilidad implicita %"); ax.set_ylabel("% real de victoria local")
ax.set_title("Calibracion del mercado: implied prob vs victoria real (local)", fontweight="bold")
plt.colorbar(sc, label="edge (real - implied)")
plt.tight_layout()
plt.savefig("img/scatter_calibracion_mercado.png", dpi=130)
plt.close()
print("  5.4 scatter_calibracion_mercado")

# overround vs goles: ver si partidos con mas margen de casa tienden a tener menos goles
sample = df.sample(min(3000, len(df)), random_state=42)
fig, ax = plt.subplots(figsize=(9, 6))
for liga in ligas:
    sub = sample[sample["Div"] == liga]
    ax.scatter(sub["overround"], sub["total_goals"], alpha=0.2, s=8, color=LIGA_COLORS[liga])
ax.set_xlabel("overround"); ax.set_ylabel("goles totales")
ax.set_title("Overround vs goles totales", fontweight="bold")
patches = [mpatches.Patch(color=LIGA_COLORS[l], label=LIGAS_NAME[l]) for l in ligas]
ax.legend(handles=patches, fontsize=8)
plt.tight_layout()
plt.savefig("img/scatter_overround_vs_goles.png", dpi=130)
plt.close()
print("  5.5 scatter_overround_vs_goles")

# smart money: si la cuota bajo antes del partido, ¿gano el local mas seguido?
sample2 = df.sample(min(4000, len(df)), random_state=7)
fig, ax = plt.subplots(figsize=(9, 6))
for liga in ligas:
    sub = sample2[sample2["Div"] == liga]
    ax.scatter(sub["odds_move_H"], sub["home_win"] + np.random.uniform(-0.05, 0.05, len(sub)),
               alpha=0.15, s=7, color=LIGA_COLORS[liga])
ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xlabel("movimiento cuota local (cierre - apertura)")
ax.set_ylabel("resultado (1=gana local)")
ax.set_title("Movimiento de cuota local vs resultado: ¿el smart money acierta?", fontweight="bold")
patches = [mpatches.Patch(color=LIGA_COLORS[l], label=LIGAS_NAME[l]) for l in ligas]
ax.legend(handles=patches, fontsize=8)
plt.tight_layout()
plt.savefig("img/scatter_smartmoney_vs_resultado.png", dpi=130)
plt.close()
print("  5.6 scatter_smartmoney_vs_resultado")

# heatmaps para patrones que no se ven bien con barras ni scatter

# matriz de transicion HT a FT: que tan probable es cada resultado final dado el del descanso
fig, axes = plt.subplots(1, 5, figsize=(22, 4))
for i, liga in enumerate(ligas):
    sub    = df[df["Div"] == liga]
    mat    = np.zeros((3, 3))
    order  = ["H", "D", "A"]
    for r, htr in enumerate(order):
        total = len(sub[sub["HTR"] == htr])
        for c, ftr in enumerate(order):
            mat[r, c] = len(sub[(sub["HTR"] == htr) & (sub["FTR"] == ftr)]) / total * 100 if total > 0 else 0
    im = axes[i].imshow(mat, cmap="Blues", vmin=0, vmax=80)
    axes[i].set_xticks([0, 1, 2]); axes[i].set_xticklabels(["FT=H", "FT=D", "FT=A"])
    axes[i].set_yticks([0, 1, 2]); axes[i].set_yticklabels(["HT=H", "HT=D", "HT=A"])
    axes[i].set_title(LIGAS_NAME[liga], fontsize=9, fontweight="bold")
    for r in range(3):
        for c in range(3):
            axes[i].text(c, r, f"{mat[r,c]:.1f}%", ha="center", va="center", fontsize=8)
fig.suptitle("Matriz de transicion HTR → FTR por liga (%)", fontweight="bold")
plt.tight_layout()
plt.savefig("img/heatmap_htr_ftr_por_liga.png", dpi=130)
plt.close()
print("  6.1 heatmap_htr_ftr_por_liga")

# correlaciones entre todas las variables numericas clave del dataset
corr_cols = ["total_goals","FTHG","FTAG","ht_goals","second_half_goals",
             "AvgH","AvgA","AvgD","overround","odds_move_H","odds_move_A","odds_move_D",
             "imp_prob_H","imp_prob_A","btts","over25","clean_sheet_h","goal_diff"]
corr_mat  = df[corr_cols].corr()
fig, ax   = plt.subplots(figsize=(14, 12))
im2 = ax.imshow(corr_mat.values, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr_cols))); ax.set_xticklabels(corr_cols, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(corr_cols))); ax.set_yticklabels(corr_cols, fontsize=8)
for r in range(len(corr_cols)):
    for c in range(len(corr_cols)):
        val = corr_mat.values[r, c]
        ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                fontsize=6, color="white" if abs(val) > 0.5 else "black")
plt.colorbar(im2, ax=ax)
ax.set_title("Heatmap de correlaciones entre variables numericas", fontweight="bold", fontsize=12)
plt.tight_layout()
plt.savefig("img/heatmap_correlaciones.png", dpi=130)
plt.close()
print("  6.2 heatmap_correlaciones")

# btts por mes y liga: detectar si hay meses mas o menos abiertos segun la liga
months_order = [8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7]
month_lbls   = [month_names.get(m, str(m)) for m in months_order]
btts_mat     = np.zeros((len(ligas), len(months_order)))
for i, liga in enumerate(ligas):
    sub = df[df["Div"] == liga]
    for j, m in enumerate(months_order):
        s = sub[sub["month"] == m]
        btts_mat[i, j] = s["btts"].mean() * 100 if len(s) > 0 else np.nan
fig, ax = plt.subplots(figsize=(14, 5))
im3 = ax.imshow(btts_mat, cmap="YlOrRd", vmin=40, vmax=65, aspect="auto")
ax.set_xticks(range(len(months_order))); ax.set_xticklabels(month_lbls)
ax.set_yticks(range(len(ligas)));        ax.set_yticklabels([LIGAS_NAME[l] for l in ligas])
for i in range(len(ligas)):
    for j in range(len(months_order)):
        if not np.isnan(btts_mat[i, j]):
            ax.text(j, i, f"{btts_mat[i,j]:.1f}", ha="center", va="center", fontsize=8)
plt.colorbar(im3, ax=ax, label="% BTTS")
ax.set_title("% BTTS por mes y liga", fontweight="bold")
plt.tight_layout()
plt.savefig("img/heatmap_btts_mes_liga.png", dpi=130)
plt.close()
print("  6.3 heatmap_btts_mes_liga")

# goles por dia de semana cruzado con liga: algunos dias son mas goleadores en algunas ligas
goles_mat = np.zeros((len(ligas), len(DAY_ORDER)))
for i, liga in enumerate(ligas):
    sub = df[df["Div"] == liga]
    for j, day in enumerate(DAY_ORDER):
        s = sub[sub["day_name"] == day]
        goles_mat[i, j] = s["total_goals"].mean() if len(s) > 0 else np.nan
fig, ax = plt.subplots(figsize=(12, 5))
im4 = ax.imshow(goles_mat, cmap="Blues", aspect="auto")
ax.set_xticks(range(len(DAY_ORDER))); ax.set_xticklabels(DAY_ORDER)
ax.set_yticks(range(len(ligas)));    ax.set_yticklabels([LIGAS_NAME[l] for l in ligas])
for i in range(len(ligas)):
    for j in range(len(DAY_ORDER)):
        if not np.isnan(goles_mat[i, j]):
            ax.text(j, i, f"{goles_mat[i,j]:.2f}", ha="center", va="center", fontsize=8)
plt.colorbar(im4, ax=ax, label="avg goles")
ax.set_title("Avg goles por dia de semana y liga", fontweight="bold")
plt.tight_layout()
plt.savefig("img/heatmap_goles_dia_liga.png", dpi=130)
plt.close()
print("  6.4 heatmap_goles_dia_liga")

# resumen de goles por celda liga x temporada para ver de un vistazo donde subio o bajo
goles_temp = np.zeros((len(ligas), len(slabels)))
for i, liga in enumerate(ligas):
    for j, sl in enumerate(slabels):
        sub = df[(df["Div"] == liga) & (df["Season_label"] == sl)]
        goles_temp[i, j] = sub["total_goals"].mean() if len(sub) > 0 else np.nan
fig, ax = plt.subplots(figsize=(13, 5))
im5 = ax.imshow(goles_temp, cmap="YlGn", aspect="auto")
ax.set_xticks(range(len(slabels)));  ax.set_xticklabels(slabels, rotation=20)
ax.set_yticks(range(len(ligas)));    ax.set_yticklabels([LIGAS_NAME[l] for l in ligas])
for i in range(len(ligas)):
    for j in range(len(slabels)):
        if not np.isnan(goles_temp[i, j]):
            ax.text(j, i, f"{goles_temp[i,j]:.2f}", ha="center", va="center", fontsize=9)
plt.colorbar(im5, ax=ax, label="avg goles")
ax.set_title("Avg goles por partido: liga x temporada", fontweight="bold")
plt.tight_layout()
plt.savefig("img/heatmap_goles_liga_temporada.png", dpi=130)
plt.close()
print("  6.5 heatmap_goles_liga_temporada")

# dashboards compuestos con multiples subplots por figura

# un dashboard de 3x3 por cada liga con los indicadores mas importantes
for liga in ligas:
    sub = df[df["Div"] == liga]
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(f"Dashboard — {LIGAS_NAME[liga]}", fontsize=15, fontweight="bold", y=1.01)
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # histograma de goles con la media marcada
    ax00 = fig.add_subplot(gs[0, 0])
    ax00.hist(sub["total_goals"], bins=range(0, int(sub["total_goals"].max()) + 2),
              color=LIGA_COLORS[liga], alpha=0.8, edgecolor="white", align="left")
    ax00.axvline(sub["total_goals"].mean(), color="red", linestyle="--", linewidth=1.2)
    ax00.set_title("Distribucion goles totales", fontsize=9)
    ax00.set_xlabel("goles"); ax00.set_ylabel("freq")

    # pie de resultados finales
    ax01 = fig.add_subplot(gs[0, 1])
    ftr_v = sub["FTR"].value_counts()
    ax01.pie([ftr_v.get("H",0), ftr_v.get("D",0), ftr_v.get("A",0)],
             labels=["H","D","A"], colors=["#3498db","#95a5a6","#e74c3c"],
             autopct="%1.1f%%", startangle=90,
             wedgeprops={"edgecolor":"white"})
    ax01.set_title("Resultados FT", fontsize=9)

    # boxplot de las tres cuotas para ver su dispersion
    ax02 = fig.add_subplot(gs[0, 2])
    ax02.boxplot([sub["AvgH"].dropna(), sub["AvgD"].dropna(), sub["AvgA"].dropna()],
                 tick_labels=["AvgH","AvgD","AvgA"], patch_artist=True,
                 boxprops=dict(facecolor=LIGA_COLORS[liga], alpha=0.6))
    ax02.set_title("Distribucion cuotas", fontsize=9); ax02.set_ylabel("cuota")

    # evolucion de goles promedio temporada a temporada
    ax10 = fig.add_subplot(gs[1, 0])
    tg_temp = sub.groupby("Season_label")["total_goals"].mean()
    common  = [s for s in slabels if s in tg_temp.index]
    ax10.plot(common, [tg_temp[s] for s in common],
              marker="o", color=LIGA_COLORS[liga], linewidth=2)
    ax10.set_title("Avg goles por temporada", fontsize=9)
    ax10.set_xlabel(""); ax10.set_ylabel("avg goles")
    ax10.tick_params(axis="x", rotation=30, labelsize=7)

    # indicadores booleanos principales en barras
    ax11 = fig.add_subplot(gs[1, 1])
    indicadores = {"btts": sub["btts"].mean()*100, "over2.5": sub["over25"].mean()*100,
                   "over3.5": sub["over35"].mean()*100, "goalless": sub["goalless"].mean()*100,
                   "cs_h":    sub["clean_sheet_h"].mean()*100}
    ax11.bar(list(indicadores.keys()), list(indicadores.values()),
             color=["#e67e22","#3498db","#9b59b6","#95a5a6","#2ecc71"], alpha=0.85)
    ax11.set_title("Indicadores (%)", fontsize=9); ax11.set_ylabel("%")
    ax11.tick_params(axis="x", rotation=20, labelsize=8)

    # matriz de transicion del descanso al resultado final
    ax12 = fig.add_subplot(gs[1, 2])
    mat  = np.zeros((3, 3)); order = ["H","D","A"]
    for r, htr in enumerate(order):
        total = len(sub[sub["HTR"] == htr])
        for c, ftr in enumerate(order):
            mat[r,c] = len(sub[(sub["HTR"]==htr)&(sub["FTR"]==ftr)]) / total * 100 if total > 0 else 0
    im_d = ax12.imshow(mat, cmap="Blues", vmin=0, vmax=80)
    ax12.set_xticks([0,1,2]); ax12.set_xticklabels(["FT=H","FT=D","FT=A"], fontsize=7)
    ax12.set_yticks([0,1,2]); ax12.set_yticklabels(["HT=H","HT=D","HT=A"], fontsize=7)
    ax12.set_title("HTR → FTR", fontsize=9)
    for r in range(3):
        for c in range(3):
            ax12.text(c, r, f"{mat[r,c]:.0f}%", ha="center", va="center", fontsize=7)

    # cuota local apertura vs cierre para ver si el mercado se mueve mucho
    ax20 = fig.add_subplot(gs[2, 0])
    ax20.scatter(sub["AvgH"], sub["AvgCH"], alpha=0.2, s=6, color=LIGA_COLORS[liga])
    ax20.plot([1, 10], [1, 10], "k--", linewidth=0.8)
    ax20.set_xlabel("AvgH ap", fontsize=8); ax20.set_ylabel("AvgCH ci", fontsize=8)
    ax20.set_title("Cuota local: ap vs ci", fontsize=9)
    ax20.set_xlim(1, 10); ax20.set_ylim(1, 10)

    # los 8 marcadores exactos mas frecuentes en esta liga
    ax21 = fig.add_subplot(gs[2, 1])
    top8 = sub["score_ft"].value_counts().head(8)
    ax21.barh(top8.index[::-1], top8.values[::-1], color=LIGA_COLORS[liga], alpha=0.8)
    ax21.set_title("Top 8 marcadores exactos FT", fontsize=9)
    ax21.set_xlabel("freq"); ax21.tick_params(axis="y", labelsize=7)

    # porcentaje de victorias locales mes a mes
    ax22 = fig.add_subplot(gs[2, 2])
    hw_month = sub.groupby("month")["home_win"].mean() * 100
    ordered  = sorted(hw_month.index, key=lambda m: (m < 7, m))
    ax22.plot([month_names.get(m, m) for m in ordered],
              [hw_month[m] for m in ordered],
              marker="o", color=LIGA_COLORS[liga], linewidth=2)
    ax22.set_title("% victoria local por mes", fontsize=9)
    ax22.set_ylabel("%"); ax22.tick_params(axis="x", rotation=35, labelsize=7)

    plt.savefig(f"img/dashboard_{liga}.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  7.1 dashboard_{liga}")

# comparativa de 8 metricas entre las 5 ligas en una sola figura de 4x2
fig, axes = plt.subplots(4, 2, figsize=(16, 20))
fig.suptitle("Comparativa entre ligas — metricas clave", fontsize=14, fontweight="bold")

metricas = [
    ("avg goles totales",    [df[df["Div"]==l]["total_goals"].mean() for l in ligas]),
    ("% victoria local",     [df[df["Div"]==l]["home_win"].mean()*100 for l in ligas]),
    ("% btts",               [df[df["Div"]==l]["btts"].mean()*100 for l in ligas]),
    ("% over 2.5",           [df[df["Div"]==l]["over25"].mean()*100 for l in ligas]),
    ("overround promedio",   [df[df["Div"]==l]["overround"].mean() for l in ligas]),
    ("avg cuota local",      [df[df["Div"]==l]["AvgH"].mean() for l in ligas]),
    ("% clean sheet local",  [df[df["Div"]==l]["clean_sheet_h"].mean()*100 for l in ligas]),
    ("% 0-0 (goalless)",     [df[df["Div"]==l]["goalless"].mean()*100 for l in ligas]),
]

for idx, (titulo, valores) in enumerate(metricas):
    ax = axes[idx // 2][idx % 2]
    bars = ax.bar([LIGAS_NAME[l] for l in ligas], valores,
                  color=[LIGA_COLORS[l] for l in ligas], alpha=0.85)
    ax.set_title(titulo, fontweight="bold", fontsize=10)
    ax.set_ylabel(titulo)
    ax.tick_params(axis="x", rotation=15, labelsize=8)
    for bar, val in zip(bars, valores):
        ax.text(bar.get_x() + bar.get_width()/2, val + max(valores)*0.01,
                f"{val:.2f}", ha="center", fontsize=8)

plt.tight_layout()
plt.savefig("img/dashboard_comparativa_ligas.png", dpi=130)
plt.close()
print("  7.2 dashboard_comparativa_ligas")

# seis tendencias temporales en un solo dashboard de 3x2
fig, axes = plt.subplots(3, 2, figsize=(16, 14))
fig.suptitle("Evolucion temporal de metricas clave por liga", fontsize=13, fontweight="bold")

evoluciones = [
    ("avg goles", "total_goals", False),
    ("% victoria local", "home_win", True),
    ("% btts", "btts", True),
    ("% over 2.5", "over25", True),
    ("overround", "overround", False),
    ("% empate", "draw", True),
]

for idx, (titulo, col, pct) in enumerate(evoluciones):
    ax = axes[idx // 2][idx % 2]
    for liga in ligas:
        sub = df[df["Div"] == liga].groupby("Season_label")[col].mean()
        if pct: sub = sub * 100
        common = [s for s in slabels if s in sub.index]
        ax.plot(common, [sub[s] for s in common],
                marker="o", label=LIGAS_NAME[liga],
                color=LIGA_COLORS[liga], linewidth=1.8, markersize=5)
    ax.set_title(titulo, fontweight="bold", fontsize=10)
    ax.set_ylabel("%" if pct else titulo)
    ax.tick_params(axis="x", rotation=25, labelsize=7)
    ax.legend(fontsize=6)

plt.tight_layout()
plt.savefig("img/dashboard_evolucion_temporal.png", dpi=130)
plt.close()
print("  7.3 dashboard_evolucion_temporal")

# graficas adicionales: calibracion, composicion, radar y equipos

# resultados reales vs lo que implica la cuota segun el rango en que cae
bins_h  = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 25.0]
lbl_h   = [f"{bins_h[i]}-{bins_h[i+1]}" for i in range(len(bins_h)-1)]
d3      = df.copy()
d3["rng_H"] = pd.cut(d3["AvgH"], bins=bins_h, labels=lbl_h, right=False)
pH_rng  = d3.groupby("rng_H")["home_win"].mean() * 100
pD_rng  = d3.groupby("rng_H")["draw"].mean()     * 100
pA_rng  = d3.groupby("rng_H")["away_win"].mean() * 100
imp_rng = d3.groupby("rng_H")["AvgH"].mean().map(lambda x: 1/x*100)
fig, ax = plt.subplots(figsize=(13, 6))
x3 = np.arange(len(lbl_h)); w3 = 0.22
ax.bar(x3 - w3,     [pH_rng.get(l, 0)  for l in lbl_h], w3, label="% H real",     color="#3498db", alpha=0.85)
ax.bar(x3,          [imp_rng.get(l, 0) for l in lbl_h], w3, label="% H implied",  color="#3498db", alpha=0.35)
ax.bar(x3 + w3,     [pD_rng.get(l, 0)  for l in lbl_h], w3, label="% D",          color="#95a5a6", alpha=0.85)
ax.bar(x3 + 2*w3,   [pA_rng.get(l, 0)  for l in lbl_h], w3, label="% A",          color="#e74c3c", alpha=0.85)
ax.set_xticks(x3 + w3/2); ax.set_xticklabels(lbl_h, rotation=20)
ax.set_ylabel("%"); ax.legend()
ax.set_title("Resultados reales vs implied por rango de cuota local", fontweight="bold")
plt.tight_layout()
plt.savefig("img/barras_resultados_por_rango_cuota.png", dpi=130)
plt.close()
print("  8.1 barras_resultados_por_rango_cuota")

# cuota de empate por mes del año para ver si hay meses donde se paga mas
fig, ax = plt.subplots(figsize=(12, 5))
for liga in ligas:
    sub = df[df["Div"] == liga].groupby("month")["AvgD"].mean()
    ordered = sorted(sub.index, key=lambda m: (m < 7, m))
    ax.plot([month_names.get(m,m) for m in ordered],
            [sub[m] for m in ordered],
            marker="o", label=LIGAS_NAME[liga], color=LIGA_COLORS[liga], linewidth=1.8)
ax.set_ylabel("avg cuota empate"); ax.legend(fontsize=8)
ax.set_title("Evolucion de la cuota de empate por mes del año", fontweight="bold")
plt.tight_layout()
plt.savefig("img/lineas_cuota_empate_mes.png", dpi=130)
plt.close()
print("  8.2 lineas_cuota_empate_mes")

# cada punto es una liga en una temporada: que tan bien calibra el mercado el empate
fig, ax = plt.subplots(figsize=(10, 7))
for liga in ligas:
    pts_e = []
    for sl in slabels:
        sub = df[(df["Div"] == liga) & (df["Season_label"] == sl)]
        if len(sub) < 20: continue
        pts_e.append({
            "imp": 1 / sub["AvgD"].mean() * 100,
            "real": sub["draw"].mean() * 100,
        })
    if not pts_e: continue
    pts_e_df = pd.DataFrame(pts_e)
    ax.scatter(pts_e_df["imp"], pts_e_df["real"], s=60, alpha=0.75,
               color=LIGA_COLORS[liga], label=LIGAS_NAME[liga], edgecolors="gray", linewidth=0.5)
ax.plot([20, 35], [20, 35], "k--", linewidth=0.8)
ax.set_xlabel("probabilidad implicita empate %"); ax.set_ylabel("% real empate")
ax.set_title("Calibracion del mercado para el empate por liga y temporada", fontweight="bold")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("img/scatter_calibracion_empate.png", dpi=130)
plt.close()
print("  8.3 scatter_calibracion_empate")

# goles apilados HT abajo y ST arriba: muestra que mitad aporta mas en cada liga
fig, ax = plt.subplots(figsize=(11, 6))
avg_ht_l = [df[df["Div"]==l]["ht_goals"].mean() for l in ligas]
avg_st_l = [df[df["Div"]==l]["second_half_goals"].mean() for l in ligas]
ax.bar([LIGAS_NAME[l] for l in ligas], avg_ht_l, label="HT", color="#3498db", alpha=0.85)
ax.bar([LIGAS_NAME[l] for l in ligas], avg_st_l, bottom=avg_ht_l, label="ST", color="#e74c3c", alpha=0.85)
ax.set_ylabel("avg goles"); ax.legend()
ax.set_title("Composicion de goles: primer vs segundo tiempo por liga", fontweight="bold")
for i, (h, s) in enumerate(zip(avg_ht_l, avg_st_l)):
    ax.text(i, h/2,   f"{h:.2f}", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    ax.text(i, h+s/2, f"{s:.2f}", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
plt.tight_layout()
plt.savefig("img/barras_composicion_ht_st.png", dpi=130)
plt.close()
print("  8.4 barras_composicion_ht_st")

# radar chart con el perfil de cada liga en seis dimensiones
categorias  = ["% H", "% btts", "% over2.5", "avg goles*10", "overround*100", "% cs_h"]
N           = len(categorias)
angulos     = [n / float(N) * 2 * np.pi for n in range(N)]
angulos    += angulos[:1]
fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
for liga in ligas:
    sub = df[df["Div"] == liga]
    valores = [
        sub["home_win"].mean()   * 100,
        sub["btts"].mean()       * 100,
        sub["over25"].mean()     * 100,
        sub["total_goals"].mean()* 10,
        sub["overround"].mean()  * 100,
        sub["clean_sheet_h"].mean() * 100,
    ]
    valores += valores[:1]
    ax.plot(angulos, valores, linewidth=2, label=LIGAS_NAME[liga], color=LIGA_COLORS[liga])
    ax.fill(angulos, valores, alpha=0.08, color=LIGA_COLORS[liga])
ax.set_xticks(angulos[:-1]); ax.set_xticklabels(categorias, fontsize=9)
ax.set_title("Radar: perfil comparativo de ligas", fontweight="bold", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=8)
plt.tight_layout()
plt.savefig("img/radar_perfil_ligas.png", dpi=130)
plt.close()
print("  8.5 radar_perfil_ligas")

# los 15 equipos que mas goles marcaron sumando local y visitante en todo el periodo
hg = df.groupby("HomeTeam")["FTHG"].sum()
ag = df.groupby("AwayTeam")["FTAG"].sum()
tg = hg.add(ag, fill_value=0).sort_values(ascending=False).head(15)
fig, ax = plt.subplots(figsize=(12, 6))
cmap_eq = cmap_n(15, "tab20")
bars_eq = ax.bar(tg.index, tg.values, color=[cmap_eq(i) for i in range(15)], alpha=0.85)
ax.set_ylabel("goles totales"); ax.tick_params(axis="x", rotation=35)
ax.set_title("Top 15 equipos goleadores (todo el periodo)", fontweight="bold")
for bar, val in zip(bars_eq, tg.values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 5, str(int(val)), ha="center", fontsize=8)
plt.tight_layout()
plt.savefig("img/barras_top15_goleadores.png", dpi=130)
plt.close()
print("  8.6 barras_top15_goleadores")

# cuanto se mueve cada tipo de cuota en promedio temporada a temporada
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, (col_ap, col_ci, titulo) in enumerate([
    ("AvgH","AvgCH","Cuota local"),
    ("AvgA","AvgCA","Cuota visitante"),
    ("AvgD","AvgCD","Cuota empate"),
]):
    for liga in ligas:
        sub = df[df["Div"] == liga]
        mov = sub.groupby("Season_label").apply(
            lambda x: (x[col_ci] - x[col_ap]).mean()
        )
        common = [s for s in slabels if s in mov.index]
        axes[i].plot(common, [mov[s] for s in common],
                     marker="o", label=LIGAS_NAME[liga], color=LIGA_COLORS[liga], linewidth=1.8)
    axes[i].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[i].set_title(titulo, fontweight="bold")
    axes[i].set_ylabel("movimiento (cierre - apertura)")
    axes[i].legend(fontsize=7); axes[i].tick_params(axis="x", rotation=25, labelsize=7)
fig.suptitle("Movimiento promedio de cuotas por temporada y liga", fontweight="bold")
plt.tight_layout()
plt.savefig("img/lineas_movimiento_cuotas_temporada.png", dpi=130)
plt.close()
print("  8.7 lineas_movimiento_cuotas_temporada")

print("\nlisto — todas las graficas guardadas en img/")