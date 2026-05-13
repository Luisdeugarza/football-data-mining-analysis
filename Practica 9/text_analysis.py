import sys
sys.stdout.reconfigure(encoding='utf-8')

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import os
from tabulate import tabulate

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "Practica 1", "data", "clean", "football_clean.csv")
IMG_DIR   = os.path.join(BASE_DIR, "img")
os.makedirs(IMG_DIR, exist_ok=True)

def print_tabulate(df: pd.DataFrame):
    print(tabulate(df, headers=df.columns, tablefmt="orgtbl"))

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
FTR_MAP = {
    "H": "HomeWin",
    "D": "Draw",
    "A": "AwayWin",
}


# cargué el dataset y construí el corpus de texto a partir de las columnas categóricas
# usé equipos, ligas y resultados porque son los campos más ricos en vocabulario

df = pd.read_csv(DATA_PATH, parse_dates=["Date"])

df["FTR_text"] = df["FTR"].map(FTR_map := {
    "H": "HomeWin",
    "D": "Draw",
    "A": "AwayWin",
})
df["Season_label"] = df["Season"].map({
    1920: "Season1920", 2021: "Season2021", 2122: "Season2122",
    2223: "Season2223", 2324: "Season2324", 2425: "Season2425", 2526: "Season2526",
})
df["Liga_text"] = df["Div"].map(LIGAS_NAME).str.replace(" ", "")


# construí el corpus global concatenando todas las columnas de texto relevantes
# repetí los equipos proporcional a sus apariciones para que el tamaño en la nube refleje frecuencia real

tokens_global = []
for _, row in df.iterrows():
    tokens_global += [
        row["HomeTeam"].replace(" ", ""),
        row["AwayTeam"].replace(" ", ""),
        row["Liga_text"],
        row["FTR_text"],
        row["Season_label"],
    ]

corpus_global = " ".join(tokens_global)

print("análisis de texto — corpus global")
print(f"  total tokens     : {len(tokens_global)}")
print(f"  tokens únicos    : {len(set(tokens_global))}")
print(f"  caracteres total : {len(corpus_global)}")


# calculé las frecuencias con Counter para mostrar las palabras más comunes
# antes de generar la nube

contador = Counter(tokens_global)
top_20   = contador.most_common(20)

print("\ntop 20 tokens más frecuentes:")
df_top = pd.DataFrame(top_20, columns=["Token", "Frecuencia"])
print_tabulate(df_top)


# generé la wordcloud global con todos los partidos de las 5 ligas
# usé fondo negro para que los colores resalten más

wordcloud_global = WordCloud(
    background_color="black",
    width=1400,
    height=700,
    min_font_size=8,
    max_words=200,
    colormap="Set2",
    collocations=False,
).generate(corpus_global)

plt.figure(figsize=(14, 7), facecolor="black")
plt.imshow(wordcloud_global, interpolation="bilinear")
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig(os.path.join(IMG_DIR, "wordcloud_global.png"), dpi=150,
            bbox_inches="tight", facecolor="black")
plt.close()
print("\n  guardado: img/wordcloud_global.png")


# generé una wordcloud separada solo con nombres de equipos
# para ver cuáles dominan en presencia histórica en el dataset

tokens_equipos = [row["HomeTeam"].replace(" ", "") for _, row in df.iterrows()] + \
                 [row["AwayTeam"].replace(" ", "") for _, row in df.iterrows()]

wordcloud_equipos = WordCloud(
    background_color="white",
    width=1400,
    height=700,
    min_font_size=8,
    max_words=150,
    colormap="tab20",
    collocations=False,
).generate(" ".join(tokens_equipos))

plt.figure(figsize=(14, 7))
plt.imshow(wordcloud_equipos, interpolation="bilinear")
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig(os.path.join(IMG_DIR, "wordcloud_equipos.png"), dpi=150,
            bbox_inches="tight")
plt.close()
print("  guardado: img/wordcloud_equipos.png")


# grafiqué el top 20 en barras para complementar la nube con información cuantitativa

fig, ax = plt.subplots(figsize=(12, 6))
tokens_top = [t[0] for t in top_20]
freqs_top  = [t[1] for t in top_20]
colors_bar = ["#3498db" if f > 2000 else "#2ecc71" if f > 500 else "#e74c3c"
               for f in freqs_top]

ax.barh(tokens_top[::-1], freqs_top[::-1], color=colors_bar[::-1], alpha=0.85)
ax.set_xlabel("frecuencia")
ax.set_title("Top 20 tokens más frecuentes en el corpus de fútbol europeo",
             fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "wordcloud_top20_barras.png"), dpi=130)
plt.close()
print("  guardado: img/wordcloud_top20_barras.png")


# generé una wordcloud por liga en un loop para comparar el vocabulario de equipos
# que tiene cada competición — usé el colormap distintivo de cada liga

print("\nnubes de palabras por liga")

LIGA_COLORMAPS = {
    "E0":  "Blues",
    "SP1": "Reds",
    "D1":  "Oranges",
    "I1":  "Greens",
    "F1":  "Purples",
}

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes_flat = axes.flatten()

for i, (liga, nombre) in enumerate(LIGAS_NAME.items()):
    sub = df[df["Div"] == liga]

    tokens_liga = []
    for _, row in sub.iterrows():
        tokens_liga += [
            row["HomeTeam"].replace(" ", ""),
            row["AwayTeam"].replace(" ", ""),
            row["FTR_text"],
        ]

    wc = WordCloud(
        background_color="white",
        width=800,
        height=500,
        min_font_size=6,
        max_words=80,
        colormap=LIGA_COLORMAPS[liga],
        collocations=False,
    ).generate(" ".join(tokens_liga))

    axes_flat[i].imshow(wc, interpolation="bilinear")
    axes_flat[i].axis("off")
    axes_flat[i].set_title(nombre, fontweight="bold", fontsize=13,
                            color=LIGA_COLORS[liga])

    cnt_liga  = Counter(tokens_liga)
    top3_liga = [t for t, _ in cnt_liga.most_common(10)
                 if t not in ("HomeWin", "AwayWin", "Draw")][:3]
    axes_flat[i].set_xlabel(f"top equipos: {' · '.join(top3_liga)}",
                             fontsize=9, color="gray")
    print(f"  {nombre}: top3 = {top3_liga}")

axes_flat[-1].axis("off")
plt.suptitle("Wordcloud por liga — equipos y resultados más frecuentes",
             fontweight="bold", fontsize=15)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "wordcloud_por_liga.png"), dpi=130,
            bbox_inches="tight")
plt.close()
print("  guardado: img/wordcloud_por_liga.png")


# construí un corpus por temporada para ver cómo cambia el vocabulario con el tiempo
# en temporadas COVID hay equipos que desaparecen o aparecen por primera vez

print("\nnubes de palabras por temporada")

SEASON_LABELS = {
    1920: "2019/20", 2021: "2020/21", 2122: "2021/22",
    2223: "2022/23", 2324: "2023/24", 2425: "2024/25", 2526: "2025/26",
}

seasons_sorted = sorted(df["Season"].dropna().unique().astype(int))
fig, axes      = plt.subplots(2, 4, figsize=(20, 9))
axes_flat      = axes.flatten()

for i, season in enumerate(seasons_sorted):
    sub   = df[df["Season"] == season]
    label = SEASON_LABELS.get(season, str(season))

    tokens_season = []
    for _, row in sub.iterrows():
        tokens_season += [
            row["HomeTeam"].replace(" ", ""),
            row["AwayTeam"].replace(" ", ""),
            row["FTR_text"],
            row["Liga_text"],
        ]

    wc = WordCloud(
        background_color="black",
        width=700,
        height=400,
        min_font_size=6,
        max_words=60,
        colormap="coolwarm",
        collocations=False,
    ).generate(" ".join(tokens_season))

    axes_flat[i].imshow(wc, interpolation="bilinear")
    axes_flat[i].axis("off")
    axes_flat[i].set_title(label, fontweight="bold", fontsize=11, color="white")
    axes_flat[i].set_facecolor("black")
    print(f"  temporada {label}: {len(sub)} partidos")

for j in range(len(seasons_sorted), len(axes_flat)):
    axes_flat[j].axis("off")

fig.patch.set_facecolor("black")
plt.suptitle("Wordcloud por temporada — evolución del vocabulario 2019-2026",
             fontweight="bold", fontsize=14, color="white")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "wordcloud_por_temporada.png"), dpi=130,
            bbox_inches="tight", facecolor="black")
plt.close()
print("  guardado: img/wordcloud_por_temporada.png")


# calculé la diversidad de vocabulario por liga — equipos únicos y distribución
# de resultados para complementar lo visual con números concretos

print("\ndiversidad de vocabulario por liga")

resumen_vocab = []
for liga, nombre in LIGAS_NAME.items():
    sub            = df[df["Div"] == liga]
    equipos_unicos = set(sub["HomeTeam"].tolist() + sub["AwayTeam"].tolist())
    resultados     = sub["FTR_text"].value_counts().to_dict()
    resumen_vocab.append({
        "Liga":      nombre,
        "Equipos":   len(equipos_unicos),
        "Partidos":  len(sub),
        "HomeWin":   resultados.get("HomeWin", 0),
        "Draw":      resultados.get("Draw", 0),
        "AwayWin":   resultados.get("AwayWin", 0),
        "%HomeWin":  f"{resultados.get('HomeWin',0)/len(sub)*100:.1f}%",
        "%Draw":     f"{resultados.get('Draw',0)/len(sub)*100:.1f}%",
        "%AwayWin":  f"{resultados.get('AwayWin',0)/len(sub)*100:.1f}%",
    })

print_tabulate(pd.DataFrame(resumen_vocab))


# construí un corpus de descripciones de partidos en texto natural
# formé una frase por partido describiendo el resultado y los goles
# así el análisis de texto tiene contenido narrativo real y no solo nombres

print("\ncorpus de descripciones narrativas de partidos")

descripciones = []
for _, row in df.iterrows():
    home   = row["HomeTeam"].replace(" ", "")
    away   = row["AwayTeam"].replace(" ", "")
    ftr    = row["FTR_text"]
    goles  = int(row["FTHG"] + row["FTAG"])
    liga   = row["Liga_text"]

    if ftr == "HomeWin":
        desc = f"{home} victoria {home} gana gana gana {liga}"
    elif ftr == "AwayWin":
        desc = f"{away} victoria {away} gana gana gana {liga}"
    else:
        desc = f"empate Draw {home} {away} {liga}"

    if goles == 0:
        desc += " cerogoles sinGoles"
    elif goles >= 5:
        desc += " golazo golazo goleada muchosgoles muchosgoles"
    elif goles >= 3:
        desc += " goles goles buen partido"

    descripciones.append(desc)

corpus_narrativo = " ".join(descripciones)
contador_narrativo = Counter(corpus_narrativo.split())
top15_narrativo    = contador_narrativo.most_common(15)

print(f"  total palabras corpus narrativo: {len(corpus_narrativo.split())}")
print(f"  palabras únicas               : {len(contador_narrativo)}")

df_narr = pd.DataFrame(top15_narrativo, columns=["Palabra", "Frecuencia"])
print_tabulate(df_narr)

wc_narrativo = WordCloud(
    background_color="white",
    width=1400,
    height=600,
    min_font_size=8,
    max_words=120,
    colormap="RdYlGn",
    collocations=False,
).generate(corpus_narrativo)

plt.figure(figsize=(14, 6))
plt.imshow(wc_narrativo, interpolation="bilinear")
plt.axis("off")
plt.title("Wordcloud narrativo — victoria, empate, goleada, liga",
          fontweight="bold", fontsize=13)
plt.tight_layout(pad=0.5)
plt.savefig(os.path.join(IMG_DIR, "wordcloud_narrativo.png"), dpi=150,
            bbox_inches="tight")
plt.close()
print("  guardado: img/wordcloud_narrativo.png")


# calculé la frecuencia relativa de cada equipo respecto al total de su liga
# para ver qué equipo domina más el vocabulario dentro de su competición

print("\nfrecuencia relativa de equipos por liga")

resumen_freq = []
for liga, nombre in LIGAS_NAME.items():
    sub    = df[df["Div"] == liga]
    total  = len(sub) * 2
    cnt    = Counter(sub["HomeTeam"].tolist() + sub["AwayTeam"].tolist())
    top5   = cnt.most_common(5)
    for equipo, freq in top5:
        resumen_freq.append({
            "Liga":      nombre,
            "Equipo":    equipo,
            "Apariciones": freq,
            "% del total": f"{freq/total*100:.1f}%",
        })

print_tabulate(pd.DataFrame(resumen_freq))

# grafiqué la frecuencia relativa del top 5 por liga en barras agrupadas

fig, axes = plt.subplots(1, 5, figsize=(18, 5), sharey=False)
for i, (liga, nombre) in enumerate(LIGAS_NAME.items()):
    sub_freq = [r for r in resumen_freq if r["Liga"] == nombre]
    equipos_f = [r["Equipo"] for r in sub_freq]
    aparic_f  = [r["Apariciones"] for r in sub_freq]
    axes[i].barh(equipos_f[::-1], aparic_f[::-1],
                 color=LIGA_COLORS[liga], alpha=0.85)
    axes[i].set_title(nombre, fontweight="bold", fontsize=9,
                      color=LIGA_COLORS[liga])
    axes[i].set_xlabel("apariciones", fontsize=8)
    axes[i].tick_params(axis='y', labelsize=7)

plt.suptitle("Top 5 equipos por apariciones en cada liga",
             fontweight="bold", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "wordcloud_freq_equipos_liga.png"), dpi=130,
            bbox_inches="tight")
plt.close()
print("  guardado: img/wordcloud_freq_equipos_liga.png")


# generé la figura comparativa final — nube global de equipos coloreada
# por frecuencia usando un colormap continuo para mostrar quién aparece más

print("\nnube final con colormap de frecuencia continua")

freq_dict = Counter(df["HomeTeam"].tolist() + df["AwayTeam"].tolist())

wc_freq = WordCloud(
    background_color="black",
    width=1600,
    height=800,
    min_font_size=6,
    max_words=200,
    colormap="plasma",
    collocations=False,
).generate_from_frequencies(freq_dict)

plt.figure(figsize=(16, 8), facecolor="black")
plt.imshow(wc_freq, interpolation="bilinear")
plt.axis("off")
plt.title("Equipos del fútbol europeo 2019-2026 — tamaño proporcional a apariciones",
          fontweight="bold", fontsize=13, color="white", pad=10)
plt.tight_layout(pad=0.5)
plt.savefig(os.path.join(IMG_DIR, "wordcloud_equipos_freq.png"), dpi=150,
            bbox_inches="tight", facecolor="black")
plt.close()
print("  guardado: img/wordcloud_equipos_freq.png")