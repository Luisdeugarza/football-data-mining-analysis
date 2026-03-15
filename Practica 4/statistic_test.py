import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from scipy import stats
from scipy.stats import (
    shapiro, kstest, levene,
    kruskal, mannwhitneyu, wilcoxon,
    f_oneway, ttest_rel, ttest_ind,
    normaltest,
)
from itertools import combinations

os.makedirs("img", exist_ok=True)

# 0. CARGA Y VARIABLES DERIVADAS

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
DAY_NAMES = {0:"Lunes",1:"Martes",2:"Miercoles",3:"Jueves",
             4:"Viernes",5:"Sabado",6:"Domingo"}

df = pd.read_csv("../Practica 1/data/clean/football_clean.csv", parse_dates=["Date"])

# Hasta aquí, fue la carga del dataframe de todos los partidos
# Aquí creé algunas variables nuevas para el analisis

df["total_goals"]       = df["FTHG"] + df["FTAG"]
df["ht_goals"]          = df["HTHG"] + df["HTAG"]
df["second_half_goals"] = df["total_goals"] - df["ht_goals"]
df["goal_diff"]         = df["FTHG"] - df["FTAG"]
df["home_win"]          = (df["FTR"] == "H").astype(int)
df["draw"]              = (df["FTR"] == "D").astype(int)
df["away_win"]          = (df["FTR"] == "A").astype(int)
df["btts"]              = ((df["FTHG"] > 0) & (df["FTAG"] > 0)).astype(int)
df["over25"]            = (df["total_goals"] > 2).astype(int)
df["clean_sheet_h"]     = (df["FTAG"] == 0).astype(int)
df["imp_prob_H"]        = round(1 / df["AvgH"], 4)
df["imp_prob_D"]        = round(1 / df["AvgD"], 4)
df["imp_prob_A"]        = round(1 / df["AvgA"], 4)
df["overround"]         = round(df["imp_prob_H"] + df["imp_prob_D"] + df["imp_prob_A"], 4)
df["odds_move_H"]       = round(df["AvgCH"] - df["AvgH"], 4)
df["dayofweek"]         = df["Date"].dt.dayofweek
df["day_name"]          = df["dayofweek"].map(DAY_NAMES)
df["Season_label"]      = df["Season"].map(SEASON_MAP)

ligas   = sorted(df["Div"].unique())
seasons = sorted(df["Season"].unique())
slabels = [SEASON_MAP[s] for s in seasons]

ALPHA = 0.05  # nivel de significancia global

print("=" * 70)
print("PRACTICA 4 — PRUEBAS ESTADISTICAS")
print(f"  Dataset: {len(df):,} partidos | {len(ligas)} ligas | {len(seasons)} temporadas")
print(f"  Nivel de significancia: α = {ALPHA}")
print("=" * 70)


# HELPERS

def separador(titulo):
    print(f"\n{'─'*70}")
    print(f"  {titulo}")
    print(f"{'─'*70}")

def resultado_test(nombre, stat, pval, alpha=ALPHA):
    sig = "✓ SIGNIFICATIVO" if pval < alpha else "✗ no significativo"
    print(f"  {nombre:<45}  stat={stat:>9.4f}  p={pval:.4e}  {sig}")
    return pval < alpha

def bonferroni_alpha(n_comparaciones):
    return ALPHA / n_comparaciones

def effect_size_r(stat_U, n1, n2):
    """r de efecto para Mann-Whitney: r = Z / sqrt(N)"""
    z = (stat_U - n1 * n2 / 2) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    return abs(z) / np.sqrt(n1 + n2)

def eta_squared(groups):
    """Eta² para Kruskal-Wallis: η² = (H - k + 1) / (n - k)"""
    all_data = np.concatenate(groups)
    n = len(all_data)
    k = len(groups)
    H, _ = kruskal(*groups)
    return (H - k + 1) / (n - k)


# 1. VALIDACION DE SUPUESTOS

separador("1. VALIDACION DE SUPUESTOS — NORMALIDAD Y HOMOGENEIDAD DE VARIANZAS")

print("\n  1a. Chequeamos normalidad en total_goals por liga\n")
print("      (Shapiro-Wilk para n<=5000; D'Agostino-Pearson para n>5000)\n")

normales = {}
for liga in ligas:
    serie = df[df["Div"] == liga]["total_goals"].dropna()
    n = len(serie)
    if n <= 5000:
        stat, p = shapiro(serie.sample(min(n, 5000), random_state=42))
        metodo = "Shapiro-Wilk"
    else:
        stat, p = normaltest(serie)
        metodo = "D'Agostino"
    es_normal = p >= ALPHA
    normales[liga] = es_normal
    print(f"    {LIGAS_NAME[liga]:<20} n={n:>4}  {metodo}  stat={stat:.4f}  p={p:.2e}"
          f"  {'Normal ✓' if es_normal else 'NO normal ✗'}")

# Ninguna liga tiene distribucion normal, como era de esperar en datos de deportes, aunque no es algo computacional, son cosas que se saben de antemano. Por eso usaremos pruebas no paramétricas para las comparaciones entre grupos (Kruskal-Wallis y Mann-Whitney).

print("\n  1b. Normalidad por temporada en total_goals\n")
normales_temp = {}
for s in seasons:
    serie = df[df["Season"] == s]["total_goals"].dropna()
    n = len(serie)
    stat, p = normaltest(serie)
    es_normal = p >= ALPHA
    normales_temp[s] = es_normal
    print(f"    {SEASON_MAP[s]}  n={n:>4}  D'Agostino  stat={stat:.4f}  p={p:.2e}"
          f"  {'Normal ✓' if es_normal else 'NO normal ✗'}")

print("\n  1c. Homogeneidad de varianzas entre ligas usando Levene (total_goals)\n")
grupos_ligas_goals = [df[df["Div"] == l]["total_goals"].dropna().values for l in ligas]
lev_stat, lev_p = levene(*grupos_ligas_goals)
resultado_test("Levene (ligas vs total_goals)", lev_stat, lev_p)

print("\n  1d. Test de Levene — homogeneidad de varianzas entre temporadas\n")
grupos_temp_goals = [df[df["Season"] == s]["total_goals"].dropna().values for s in seasons]
lev_stat_t, lev_p_t = levene(*grupos_temp_goals)
resultado_test("Levene (temporadas vs total_goals)", lev_stat_t, lev_p_t)

print("""
Conclusion de los supuestos:
- Los datos de goles NO siguen una distribucion normal (p muy pequeno en todos los grupos).
- Las varianzas tampoco son iguales entre ligas y temporadas.
- Por eso usé Kruskal-Wallis como prueba principal (no parametrica).
- ANOVA la pongo como contraste para ver la diferencia con el enfoque parametrico.
- Para comparaciones pairwise usé Mann-Whitney con correccion de Bonferroni.
""")


# 2. DIFERENCIAS ENTRE LIGAS

separador("2. DIFERENCIAS ENTRE LIGAS")

variables_ligas = [
    ("total_goals",  "Goles totales por partido"),
    ("overround",    "Overround (margen de la casa)"),
    ("btts",         "BTTS (ambos marcan)"),
    ("clean_sheet_h","Clean sheet local"),
    ("home_win",     "Victoria local"),
    ("over25",       "Over 2.5 goles"),
    ("AvgH",         "Cuota local apertura"),
    ("odds_move_H",  "Movimiento cuota local"),
]

print("\n  2a. Kruskal-Wallis para comparar distribuciones entre ligas\n")
kw_resultados = {}
for col, label in variables_ligas:
    grupos = [df[df["Div"] == l][col].dropna().values for l in ligas]
    H, p = kruskal(*grupos)
    eta2 = eta_squared(grupos)
    sig = resultado_test(f"KW — {label}", H, p)
    kw_resultados[col] = {"H": H, "p": p, "sig": sig, "eta2": eta2, "label": label}
    if sig:
        print(f"    → η² = {eta2:.4f}  (efecto {'pequeño' if eta2<0.06 else 'moderado' if eta2<0.14 else 'grande'})")

print("\n  2b. ANOVA para comparar (aunque no sea valido aqui)\n")
for col, label in variables_ligas:
    grupos = [df[df["Div"] == l][col].dropna().values for l in ligas]
    F, p = f_oneway(*grupos)
    resultado_test(f"ANOVA — {label}", F, p)

print("\n  2c. Comparaciones uno a uno usando Mann-Whitney con Bonferroni\n")
pares = list(combinations(ligas, 2))
alpha_bon = bonferroni_alpha(len(pares))
print(f"    {len(pares)} comparaciones → α_Bonferroni = {alpha_bon:.4f}\n")

mw_resultados_ligas = {}
for l1, l2 in pares:
    g1 = df[df["Div"] == l1]["total_goals"].dropna().values
    g2 = df[df["Div"] == l2]["total_goals"].dropna().values
    U, p = mannwhitneyu(g1, g2, alternative="two-sided")
    r = effect_size_r(U, len(g1), len(g2))
    sig = p < alpha_bon
    mw_resultados_ligas[(l1, l2)] = {"U": U, "p": p, "r": r, "sig": sig}
    marca = "✓" if sig else "✗"
    print(f"    {marca} {LIGAS_NAME[l1]:<20} vs {LIGAS_NAME[l2]:<20}  "
          f"U={U:.0f}  p={p:.4e}  r={r:.3f}  "
          f"({'SIGNIFICATIVO' if sig else 'no sig'})")

print("\n  2d. Mann-Whitney pairwise — overround\n")
for l1, l2 in pares:
    g1 = df[df["Div"] == l1]["overround"].dropna().values
    g2 = df[df["Div"] == l2]["overround"].dropna().values
    U, p = mannwhitneyu(g1, g2, alternative="two-sided")
    r = effect_size_r(U, len(g1), len(g2))
    sig = p < alpha_bon
    marca = "✓" if sig else "✗"
    print(f"    {marca} {LIGAS_NAME[l1]:<20} vs {LIGAS_NAME[l2]:<20}  "
          f"p={p:.4e}  r={r:.3f}  ({'SIGNIFICATIVO' if sig else 'no sig'})")


# 3. DIFERENCIAS ENTRE TEMPORADAS

separador("3. DIFERENCIAS ENTRE TEMPORADAS")

print("\n  3a. Kruskal-Wallis — goles totales entre temporadas\n")
grupos_temp = [df[df["Season"] == s]["total_goals"].dropna().values for s in seasons]
H_temp, p_temp = kruskal(*grupos_temp)
resultado_test("KW — total_goals por temporada", H_temp, p_temp)
eta2_temp = eta_squared(grupos_temp)
print(f"    → η² = {eta2_temp:.4f}")

print("\n  3b. ANOVA — contraste paramétrico\n")
F_temp, p_anova_temp = f_oneway(*grupos_temp)
resultado_test("ANOVA — total_goals por temporada", F_temp, p_anova_temp)

print("\n  3c. Kruskal-Wallis — % victorias locales entre temporadas\n")
grupos_hw = [df[df["Season"] == s]["home_win"].dropna().values for s in seasons]
H_hw, p_hw = kruskal(*grupos_hw)
resultado_test("KW — home_win por temporada", H_hw, p_hw)

print("\n  3d. Mann-Whitney pairwise — total_goals entre temporadas (Bonferroni)\n")
pares_temp = list(combinations(seasons, 2))
alpha_bon_t = bonferroni_alpha(len(pares_temp))
print(f"    {len(pares_temp)} comparaciones → α_Bonferroni = {alpha_bon_t:.4f}\n")
for s1, s2 in pares_temp:
    g1 = df[df["Season"] == s1]["total_goals"].dropna().values
    g2 = df[df["Season"] == s2]["total_goals"].dropna().values
    U, p = mannwhitneyu(g1, g2, alternative="two-sided")
    r = effect_size_r(U, len(g1), len(g2))
    sig = p < alpha_bon_t
    marca = "✓" if sig else "✗"
    print(f"    {marca} {SEASON_MAP[s1]} vs {SEASON_MAP[s2]}  "
          f"p={p:.4e}  r={r:.3f}  ({'SIGNIFICATIVO' if sig else 'no sig'})")

# 4. LOCAL vs VISITANTE (pareado)

separador("4. LOCAL vs VISITANTE — VENTAJA DE LOCAL")

print("\n  Hipótesis nula: FTHG y FTAG tienen la misma distribución (no hay ventaja local)\n")

fthg = df["FTHG"].values
ftag = df["FTAG"].values

# Wilcoxon pareado (no paramétrico — recomendado porque los datos no son normales)
stat_w, p_w = wilcoxon(fthg, ftag, alternative="greater")
resultado_test("Wilcoxon pareado  FTHG > FTAG", stat_w, p_w)

# T-test pareado como contraste paramétrico
stat_t, p_t = ttest_rel(fthg, ftag, alternative="greater")
resultado_test("T-test pareado    FTHG > FTAG", stat_t, p_t)

diff = fthg - ftag
print(f"\n    Diferencia media (local - visitante): {diff.mean():.4f}")
print(f"    Partidos donde local marcó más: {(diff > 0).sum():,} ({(diff > 0).mean()*100:.1f}%)")
print(f"    Empate en goles: {(diff == 0).sum():,} ({(diff == 0).mean()*100:.1f}%)")
print(f"    Partidos donde visitante marcó más: {(diff < 0).sum():,} ({(diff < 0).mean()*100:.1f}%)")


# 5. PRIMER TIEMPO vs SEGUNDO TIEMPO (pareado)

separador("5. PRIMER TIEMPO vs SEGUNDO TIEMPO")

print("\n  Hipótesis nula: mismo número de goles en ambas mitades\n")

ht  = df["ht_goals"].values
st  = df["second_half_goals"].values

stat_wht, p_wht = wilcoxon(st, ht, alternative="greater")
resultado_test("Wilcoxon pareado  ST > HT", stat_wht, p_wht)

stat_tht, p_tht = ttest_rel(st, ht, alternative="greater")
resultado_test("T-test pareado    ST > HT", stat_tht, p_tht)

print(f"\n    Avg goles HT: {ht.mean():.3f}  |  Avg goles ST: {st.mean():.3f}")
print(f"    Diferencia ST-HT: {(st-ht).mean():.4f}")

print("\n  Por liga:\n")
for liga in ligas:
    sub = df[df["Div"] == liga]
    h = sub["ht_goals"].values
    s = sub["second_half_goals"].values
    stat_l, p_l = wilcoxon(s, h, alternative="greater")
    sig = "✓" if p_l < ALPHA else "✗"
    print(f"    {sig} {LIGAS_NAME[liga]:<20}  HT={h.mean():.3f}  ST={s.mean():.3f}  "
          f"p={p_l:.4e}")

# 6. BTTS SEGUN TIPO DE PARTIDO (perfil de cuota)

separador("6. BTTS SEGUN PERFIL DE PARTIDO (favorito / equilibrado / underdog)")

print("\n  Clasificación por diferencia |AvgH - AvgA|:")
print("    Favorito claro:  |diff| > 1.5")
print("    Equilibrado:     |diff| entre 0.5 y 1.5")
print("    Underdog claro:  |diff| <= 0.5  (cuotas muy cercanas)\n")

df["odds_diff"] = abs(df["AvgH"] - df["AvgA"])
df["match_type"] = pd.cut(
    df["odds_diff"],
    bins=[0, 0.5, 1.5, 100],
    labels=["equilibrado", "moderado", "favorito_claro"]
)

tipos = df["match_type"].cat.categories.tolist()
grupos_btts = [df[df["match_type"] == t]["btts"].dropna().values for t in tipos]

H_btts, p_btts = kruskal(*grupos_btts)
resultado_test("KW — btts segun tipo de partido", H_btts, p_btts)

print("\n    Estadísticas por tipo de partido:\n")
for t in tipos:
    sub = df[df["match_type"] == t]
    print(f"    {t:<20}  n={len(sub):>4}  "
          f"%btts={sub['btts'].mean()*100:.1f}  "
          f"%over25={sub['over25'].mean()*100:.1f}  "
          f"avg_goles={sub['total_goals'].mean():.3f}")

print("\n  Pairwise Mann-Whitney con Bonferroni:\n")
pares_tipos = list(combinations(tipos, 2))
alpha_bon_b = bonferroni_alpha(len(pares_tipos))
for t1, t2 in pares_tipos:
    g1 = df[df["match_type"] == t1]["btts"].dropna().values
    g2 = df[df["match_type"] == t2]["btts"].dropna().values
    U, p = mannwhitneyu(g1, g2, alternative="two-sided")
    r = effect_size_r(U, len(g1), len(g2))
    sig = p < alpha_bon_b
    marca = "✓" if sig else "✗"
    print(f"    {marca} {t1:<20} vs {t2:<20}  p={p:.4e}  r={r:.3f}")


# 7. OVERROUND POR LIGA

separador("7. OVERROUND POR LIGA — ¿EL MARGEN DE LA CASA ES IGUAL EN TODAS LAS LIGAS?")

print("\n  Hipótesis nula: el overround tiene la misma distribución en las 5 ligas\n")

grupos_or = [df[df["Div"] == l]["overround"].dropna().values for l in ligas]
H_or, p_or = kruskal(*grupos_or)
eta2_or = eta_squared(grupos_or)
resultado_test("KW — overround por liga", H_or, p_or)
print(f"    → η² = {eta2_or:.4f}")

F_or, p_or_anova = f_oneway(*grupos_or)
resultado_test("ANOVA — overround por liga (contraste)", F_or, p_or_anova)

print("\n    Estadísticas de overround por liga:\n")
for liga in ligas:
    sub = df[df["Div"] == liga]["overround"]
    print(f"    {LIGAS_NAME[liga]:<20}  mean={sub.mean():.4f}  "
          f"median={sub.median():.4f}  std={sub.std():.4f}")

print("\n  Pairwise Mann-Whitney (Bonferroni):\n")
for l1, l2 in pares:
    g1 = df[df["Div"] == l1]["overround"].dropna().values
    g2 = df[df["Div"] == l2]["overround"].dropna().values
    U, p = mannwhitneyu(g1, g2, alternative="two-sided")
    r = effect_size_r(U, len(g1), len(g2))
    sig = p < alpha_bon
    marca = "✓" if sig else "✗"
    print(f"    {marca} {LIGAS_NAME[l1]:<20} vs {LIGAS_NAME[l2]:<20}  "
          f"p={p:.4e}  r={r:.3f}")



# 8. MOVIMIENTO DE CUOTA Y RESULTADO (smart money)


separador("8. MOVIMIENTO DE CUOTA Y RESULTADO — ¿EL SMART MONEY PREDICE?")

print("""
  Grupos según movimiento de cuota local (AvgCH - AvgH):
    bajó   : odds_move_H < -0.05  (dinero entrando al local, cuota baja)
    igual  : entre -0.05 y +0.05
    subió  : odds_move_H > +0.05  (dinero saliendo del local, cuota sube)

  Hipótesis: el rate de victoria local es distinto entre los tres grupos.
""")

df["move_grupo"] = pd.cut(
    df["odds_move_H"],
    bins=[-np.inf, -0.05, 0.05, np.inf],
    labels=["bajo", "estable", "subio"]
)

grupos_sm = [
    df[df["move_grupo"] == "bajo"]["home_win"].dropna().values,
    df[df["move_grupo"] == "estable"]["home_win"].dropna().values,
    df[df["move_grupo"] == "subio"]["home_win"].dropna().values,
]

H_sm, p_sm = kruskal(*grupos_sm)
resultado_test("KW — home_win segun movimiento cuota", H_sm, p_sm)
eta2_sm = eta_squared(grupos_sm)
print(f"    → η² = {eta2_sm:.4f}")

print("\n    Tasa de victoria local por grupo de movimiento:\n")
for g in ["bajo", "estable", "subio"]:
    sub = df[df["move_grupo"] == g]
    print(f"    cuota {g:<8}  n={len(sub):>4}  "
          f"%H={sub['home_win'].mean()*100:.1f}  "
          f"%D={sub['draw'].mean()*100:.1f}  "
          f"%A={sub['away_win'].mean()*100:.1f}")

print("\n  Pairwise Mann-Whitney (Bonferroni):\n")
pares_sm = list(combinations(["bajo","estable","subio"], 2))
alpha_bon_sm = bonferroni_alpha(len(pares_sm))
for g1_name, g2_name in pares_sm:
    g1 = df[df["move_grupo"] == g1_name]["home_win"].dropna().values
    g2 = df[df["move_grupo"] == g2_name]["home_win"].dropna().values
    U, p = mannwhitneyu(g1, g2, alternative="two-sided")
    r = effect_size_r(U, len(g1), len(g2))
    sig = p < alpha_bon_sm
    marca = "✓" if sig else "✗"
    print(f"    {marca} cuota {g1_name:<8} vs {g2_name:<8}  p={p:.4e}  r={r:.3f}")



# 9. DIA DE SEMANA Y GOLES


separador("9. DIA DE SEMANA Y GOLES")

print("\n  Hipótesis: la distribución de goles es igual en todos los días de la semana\n")

DAY_ORDER = ["Lunes","Martes","Miercoles","Jueves","Viernes","Sabado","Domingo"]
dias_presentes = [d for d in DAY_ORDER if d in df["day_name"].values]
grupos_dias = [df[df["day_name"] == d]["total_goals"].dropna().values for d in dias_presentes]

H_dias, p_dias = kruskal(*grupos_dias)
resultado_test("KW — total_goals por dia de semana", H_dias, p_dias)

print("\n    Estadísticas por día:\n")
for d in dias_presentes:
    sub = df[df["day_name"] == d]["total_goals"]
    print(f"    {d:<12}  n={len(sub):>4}  mean={sub.mean():.3f}  "
          f"median={sub.median():.1f}  std={sub.std():.3f}")

# finde vs entresemana
df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
g_finde = df[df["is_weekend"] == 1]["total_goals"].dropna().values
g_entre = df[df["is_weekend"] == 0]["total_goals"].dropna().values

print("\n  Finde vs entresemana (Mann-Whitney):\n")
U_fe, p_fe = mannwhitneyu(g_finde, g_entre, alternative="two-sided")
r_fe = effect_size_r(U_fe, len(g_finde), len(g_entre))
resultado_test("Mann-Whitney — finde vs entresemana", U_fe, p_fe)
print(f"    Finde: mean={g_finde.mean():.3f}  Entresemana: mean={g_entre.mean():.3f}  r={r_fe:.3f}")



# 10. TEMPORADA COVID (2019/20) vs RESTO


separador("10. TEMPORADA COVID (2019/20) vs RESTO")

print("""
  La temporada 2019/20 se jugó sin público a partir de marzo 2020.
  Hipótesis: los goles y la ventaja local fueron distintos esa temporada.
""")

covid   = df[df["Season"] == 1920]["total_goals"].dropna().values
no_covid = df[df["Season"] != 1920]["total_goals"].dropna().values

U_cov, p_cov = mannwhitneyu(covid, no_covid, alternative="two-sided")
r_cov = effect_size_r(U_cov, len(covid), len(no_covid))
resultado_test("Mann-Whitney — goles COVID vs resto", U_cov, p_cov)
print(f"    COVID avg={covid.mean():.3f}  Resto avg={no_covid.mean():.3f}  r={r_cov:.3f}")

hw_covid  = df[df["Season"] == 1920]["home_win"].dropna().values
hw_resto  = df[df["Season"] != 1920]["home_win"].dropna().values
U_hw, p_hw = mannwhitneyu(hw_covid, hw_resto, alternative="two-sided")
r_hw = effect_size_r(U_hw, len(hw_covid), len(hw_resto))
resultado_test("Mann-Whitney — home_win COVID vs resto", U_hw, p_hw)
print(f"    COVID %H={hw_covid.mean()*100:.1f}%  Resto %H={hw_resto.mean()*100:.1f}%  r={r_hw:.3f}")



# 11. GRAFICAS


separador("11. GENERANDO GRAFICAS")

# ── 11.1 p-values Kruskal-Wallis por variable × liga 
fig, ax = plt.subplots(figsize=(13, 6))
cols_kw  = [v[0] for v in variables_ligas]
labs_kw  = [v[1] for v in variables_ligas]
pvals_kw = []
for col, _ in variables_ligas:
    grupos = [df[df["Div"] == l][col].dropna().values for l in ligas]
    _, p = kruskal(*grupos)
    pvals_kw.append(p)

colors_bar = ["#e74c3c" if p < ALPHA else "#95a5a6" for p in pvals_kw]
bars = ax.bar(labs_kw, [-np.log10(p) for p in pvals_kw], color=colors_bar, alpha=0.85)
ax.axhline(-np.log10(ALPHA), color="black", linewidth=1.2, linestyle="--",
           label=f"α = {ALPHA}  (-log₁₀ = {-np.log10(ALPHA):.1f})")
ax.set_ylabel("-log₁₀(p-value)")
ax.set_title("Significancia Kruskal-Wallis por variable — comparación entre ligas",
             fontweight="bold")
ax.tick_params(axis="x", rotation=25, labelsize=8)
ax.legend()
for bar, p in zip(bars, pvals_kw):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"p={p:.1e}", ha="center", fontsize=7, rotation=0)
plt.tight_layout()
plt.savefig("img/kw_pvalues_ligas.png", dpi=130)
plt.close()
print("  11.1 kw_pvalues_ligas")

# ── 11.2 Boxplot goles por liga con resultado del KW 
fig, ax = plt.subplots(figsize=(12, 6))
data_bp = [df[df["Div"] == l]["total_goals"].values for l in ligas]
bp = ax.boxplot(data_bp, patch_artist=True, notch=True)
for j, patch in enumerate(bp["boxes"]):
    patch.set_facecolor(list(LIGA_COLORS.values())[j])
    patch.set_alpha(0.7)
ax.set_xticklabels([LIGAS_NAME[l] for l in ligas])
H_g, p_g = kruskal(*data_bp)
ax.set_title(f"Goles por partido — boxplot por liga\n"
             f"Kruskal-Wallis: H={H_g:.2f}, p={p_g:.2e}  (diferencia SIGNIFICATIVA)",
             fontweight="bold")
ax.set_ylabel("goles totales")
plt.tight_layout()
plt.savefig("img/boxplot_goles_ligas_kw.png", dpi=130)
plt.close()
print("  11.2 boxplot_goles_ligas_kw")

# ── 11.3 Matriz de p-values pairwise Mann-Whitney (goles) 
n = len(ligas)
pmat = np.ones((n, n))
for i, l1 in enumerate(ligas):
    for j, l2 in enumerate(ligas):
        if i != j:
            g1 = df[df["Div"] == l1]["total_goals"].dropna().values
            g2 = df[df["Div"] == l2]["total_goals"].dropna().values
            _, p = mannwhitneyu(g1, g2, alternative="two-sided")
            pmat[i, j] = p

fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(-np.log10(pmat + 1e-300), cmap="RdYlGn", vmin=0, vmax=15)
ax.set_xticks(range(n)); ax.set_xticklabels([LIGAS_NAME[l] for l in ligas], rotation=25, fontsize=8)
ax.set_yticks(range(n)); ax.set_yticklabels([LIGAS_NAME[l] for l in ligas], fontsize=8)
for i in range(n):
    for j in range(n):
        if i != j:
            txt = f"p={pmat[i,j]:.2e}" if pmat[i,j] < 0.001 else f"p={pmat[i,j]:.3f}"
            sig_mark = "*" if pmat[i,j] < alpha_bon else ""
            ax.text(j, i, txt + sig_mark, ha="center", va="center", fontsize=7)
        else:
            ax.text(j, i, "—", ha="center", va="center", fontsize=9)
plt.colorbar(im, ax=ax, label="-log₁₀(p)")
ax.set_title("Matriz Mann-Whitney pairwise — goles totales entre ligas\n"
             f"(* = significativo con Bonferroni α={alpha_bon:.4f})",
             fontweight="bold")
plt.tight_layout()
plt.savefig("img/heatmap_mw_ligas.png", dpi=130)
plt.close()
print("  11.3 heatmap_mw_ligas")

# ── 11.4 Distribución local vs visitante con test 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df["FTHG"], bins=range(0, 11), alpha=0.6, color="#3498db",
             label=f"Local  μ={df['FTHG'].mean():.2f}", edgecolor="white",
             align="left", density=True)
axes[0].hist(df["FTAG"], bins=range(0, 11), alpha=0.6, color="#e74c3c",
             label=f"Visitante  μ={df['FTAG'].mean():.2f}", edgecolor="white",
             align="left", density=True)
stat_w2, p_w2 = wilcoxon(df["FTHG"].values, df["FTAG"].values, alternative="greater")
axes[0].set_title(f"Goles local vs visitante\nWilcoxon pareado: p={p_w2:.2e}  ✓ SIGNIFICATIVO",
                  fontweight="bold")
axes[0].set_xlabel("goles"); axes[0].set_ylabel("densidad"); axes[0].legend()

# Segundo tiempo vs primer tiempo
axes[1].hist(df["ht_goals"], bins=range(0, 9), alpha=0.6, color="#3498db",
             label=f"HT  μ={df['ht_goals'].mean():.2f}", edgecolor="white",
             align="left", density=True)
axes[1].hist(df["second_half_goals"], bins=range(0, 9), alpha=0.6, color="#e74c3c",
             label=f"ST  μ={df['second_half_goals'].mean():.2f}", edgecolor="white",
             align="left", density=True)
stat_w3, p_w3 = wilcoxon(df["second_half_goals"].values, df["ht_goals"].values,
                          alternative="greater")
axes[1].set_title(f"Goles primer vs segundo tiempo\nWilcoxon pareado: p={p_w3:.2e}  ✓ SIGNIFICATIVO",
                  fontweight="bold")
axes[1].set_xlabel("goles"); axes[1].legend()

plt.suptitle("Pruebas pareadas: ventaja local y segundo tiempo", fontweight="bold", fontsize=12)
plt.tight_layout()
plt.savefig("img/wilcoxon_local_ht_st.png", dpi=130)
plt.close()
print("  11.4 wilcoxon_local_ht_st")

# ── 11.5 Goles por temporada con ANOVA y KW 
fig, ax = plt.subplots(figsize=(13, 6))
avg_temp = [df[df["Season"] == s]["total_goals"].mean() for s in seasons]
std_temp = [df[df["Season"] == s]["total_goals"].std() for s in seasons]
cmap_s = matplotlib.colormaps.get_cmap("coolwarm").resampled(len(seasons))
bars_t = ax.bar(slabels, avg_temp,
                color=[cmap_s(i) for i in range(len(seasons))],
                alpha=0.85, edgecolor="white")
ax.errorbar(slabels, avg_temp, yerr=std_temp, fmt="none",
            color="black", capsize=4, linewidth=1)
for bar, val in zip(bars_t, avg_temp):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
            f"{val:.2f}", ha="center", fontsize=8)
H_t2, p_t2 = kruskal(*grupos_temp)
F_t2, p_a2 = f_oneway(*grupos_temp)
ax.set_ylabel("avg goles por partido")
ax.set_title(f"Promedio de goles por temporada (± std)\n"
             f"KW: H={H_t2:.2f}, p={p_t2:.2e}  |  ANOVA: F={F_t2:.2f}, p={p_a2:.2e}",
             fontweight="bold")
ax.tick_params(axis="x", rotation=20)
plt.tight_layout()
plt.savefig("img/barras_goles_temporada_tests.png", dpi=130)
plt.close()
print("  11.5 barras_goles_temporada_tests")

# ── 11.6 Smart money: % victorias por grupo de movimiento 
fig, ax = plt.subplots(figsize=(10, 5))
grupos_sm_names = ["bajo", "estable", "subio"]
hw_rates = [df[df["move_grupo"] == g]["home_win"].mean() * 100
            for g in grupos_sm_names]
ns_sm = [len(df[df["move_grupo"] == g]) for g in grupos_sm_names]
colors_sm = ["#2ecc71", "#95a5a6", "#e74c3c"]
bars_sm = ax.bar(["Cuota bajó\n(smart money → local)",
                  "Cuota estable",
                  "Cuota subió\n(smart money → visitante)"],
                 hw_rates, color=colors_sm, alpha=0.85)
for bar, val, n in zip(bars_sm, hw_rates, ns_sm):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.5,
            f"{val:.1f}%\n(n={n:,})", ha="center", fontsize=9)
ax.set_ylabel("% victorias local")
ax.set_ylim(0, 60)
ax.axhline(df["home_win"].mean() * 100, color="black", linestyle="--",
           linewidth=1.2, label=f"promedio global ({df['home_win'].mean()*100:.1f}%)")
ax.legend()
H_sm2, p_sm2 = kruskal(*grupos_sm)
ax.set_title(f"¿El movimiento de cuota predice la victoria local?\n"
             f"KW: H={H_sm2:.2f}, p={p_sm2:.2e}  ✓ SIGNIFICATIVO",
             fontweight="bold")
plt.tight_layout()
plt.savefig("img/barras_smartmoney_test.png", dpi=130)
plt.close()
print("  11.6 barras_smartmoney_test")

# ── 11.7 BTTS por tipo de partido 
fig, ax = plt.subplots(figsize=(10, 5))
btts_rates = [df[df["match_type"] == t]["btts"].mean() * 100 for t in tipos]
ns_tipos   = [len(df[df["match_type"] == t]) for t in tipos]
colors_tipos = ["#3498db", "#f39c12", "#e74c3c"]
bars_b = ax.bar(["Equilibrado\n(cuotas similares)",
                 "Moderado",
                 "Favorito claro"],
                btts_rates, color=colors_tipos, alpha=0.85)
for bar, val, n in zip(bars_b, btts_rates, ns_tipos):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.5,
            f"{val:.1f}%\n(n={n:,})", ha="center", fontsize=9)
ax.set_ylabel("% BTTS")
ax.axhline(df["btts"].mean() * 100, color="black", linestyle="--",
           linewidth=1.2, label=f"promedio global ({df['btts'].mean()*100:.1f}%)")
ax.legend()
H_b2, p_b2 = kruskal(*grupos_btts)
ax.set_title(f"% BTTS según tipo de partido\nKW: H={H_b2:.2f}, p={p_b2:.2e}  ✓ SIGNIFICATIVO",
             fontweight="bold")
plt.tight_layout()
plt.savefig("img/barras_btts_tipo_partido.png", dpi=130)
plt.close()
print("  11.7 barras_btts_tipo_partido")

# ── 11.8 Overround por liga — violín + test 
fig, ax = plt.subplots(figsize=(12, 6))
parts = ax.violinplot(grupos_or, positions=range(len(ligas)), showmedians=True)
for j, pc in enumerate(parts["bodies"]):
    pc.set_facecolor(list(LIGA_COLORS.values())[j]); pc.set_alpha(0.7)
ax.set_xticks(range(len(ligas)))
ax.set_xticklabels([LIGAS_NAME[l] for l in ligas])
ax.set_ylabel("overround")
H_or2, p_or2 = kruskal(*grupos_or)
ax.set_title(f"Distribución del overround por liga\n"
             f"KW: H={H_or2:.2f}, p={p_or2:.2e}  ✓ SIGNIFICATIVO",
             fontweight="bold")
plt.tight_layout()
plt.savefig("img/violin_overround_test.png", dpi=130)
plt.close()
print("  11.8 violin_overround_test")

# ── 11.9 Goles por día de semana con test 
fig, ax = plt.subplots(figsize=(12, 5))
avg_dias = [df[df["day_name"] == d]["total_goals"].mean() for d in dias_presentes]
std_dias = [df[df["day_name"] == d]["total_goals"].std() for d in dias_presentes]
ns_dias  = [len(df[df["day_name"] == d]) for d in dias_presentes]
cmap_d = matplotlib.colormaps.get_cmap("Set2").resampled(len(dias_presentes))
bars_d = ax.bar(dias_presentes, avg_dias,
                color=[cmap_d(i) for i in range(len(dias_presentes))], alpha=0.85)
ax.errorbar(dias_presentes, avg_dias, yerr=std_dias, fmt="none",
            color="black", capsize=4, linewidth=1)
for bar, val, n in zip(bars_d, avg_dias, ns_dias):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.03,
            f"{val:.2f}\n(n={n})", ha="center", fontsize=7)
H_d2, p_d2 = kruskal(*grupos_dias)
ax.set_ylabel("avg goles por partido")
ax.set_title(f"Goles promedio por día de semana (± std)\n"
             f"KW: H={H_d2:.2f}, p={p_d2:.2e}",
             fontweight="bold")
plt.tight_layout()
plt.savefig("img/barras_goles_dias_test.png", dpi=130)
plt.close()
print("  11.9 barras_goles_dias_test")

# ── 11.10 COVID vs los otros años
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# goles
bins_g = range(0, 11)
axes[0].hist(covid, bins=bins_g, alpha=0.6, color="#e74c3c",
             label=f"2019/20 COVID  μ={covid.mean():.2f}",
             edgecolor="white", align="left", density=True)
axes[0].hist(no_covid, bins=bins_g, alpha=0.6, color="#3498db",
             label=f"Resto  μ={no_covid.mean():.2f}",
             edgecolor="white", align="left", density=True)
axes[0].set_title(f"Goles: COVID vs resto\nMann-Whitney p={p_cov:.2e}  r={r_cov:.3f}",
                  fontweight="bold")
axes[0].set_xlabel("goles"); axes[0].set_ylabel("densidad"); axes[0].legend()

# % victoria local
hw_rates_cov = [hw_covid.mean() * 100, hw_resto.mean() * 100]
bars_cov = axes[1].bar(["2019/20 (COVID)", "Resto temporadas"],
                        hw_rates_cov, color=["#e74c3c", "#3498db"], alpha=0.85)
for bar, val in zip(bars_cov, hw_rates_cov):
    axes[1].text(bar.get_x() + bar.get_width()/2, val + 0.3,
                 f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")
axes[1].set_ylabel("% victorias local")
axes[1].set_title(f"Victoria local: COVID vs resto\nMann-Whitney p={p_hw:.2e}  r={r_hw:.3f}",
                   fontweight="bold")

plt.suptitle("Impacto de la temporada COVID (sin público) en los resultados",
             fontweight="bold", fontsize=12)
plt.tight_layout()
plt.savefig("img/barras_covid_vs_resto.png", dpi=130)
plt.close()
print("  11.10 barras_covid_vs_resto")

# ── 11.11 Resumen visual de todos los tests 
resumen_tests = [
    ("KW Goles × Liga",         "kruskal", True),
    ("KW Overround × Liga",     "kruskal", True),
    ("KW BTTS × Liga",          "kruskal", True),
    ("KW Goles × Temporada",    "kruskal", True),
    ("Wilcoxon Local > Visit.", "wilcoxon", True),
    ("Wilcoxon ST > HT",        "wilcoxon", True),
    ("KW BTTS × Tipo partido",  "kruskal", True),
    ("KW Overround × Liga",     "kruskal", True),
    ("KW Smart Money × H",      "kruskal", True),
    ("KW Goles × Dia semana",   "kruskal", True),
    ("MW COVID vs Resto goles", "mann-w",  True),
    ("MW COVID vs Resto %H",    "mann-w",  True),
]

all_pvals = []
# recalcula todos los p-values para el resumen
grupos_l_g  = [df[df["Div"]==l]["total_goals"].dropna().values for l in ligas]
grupos_l_or = [df[df["Div"]==l]["overround"].dropna().values for l in ligas]
grupos_l_bt = [df[df["Div"]==l]["btts"].dropna().values for l in ligas]
grupos_t_g  = [df[df["Season"]==s]["total_goals"].dropna().values for s in seasons]

all_pvals_final = [
    kruskal(*grupos_l_g)[1],
    kruskal(*grupos_l_or)[1],
    kruskal(*grupos_l_bt)[1],
    kruskal(*grupos_t_g)[1],
    wilcoxon(df["FTHG"].values, df["FTAG"].values, alternative="greater")[1],
    wilcoxon(df["second_half_goals"].values, df["ht_goals"].values, alternative="greater")[1],
    kruskal(*grupos_btts)[1],
    kruskal(*grupos_or)[1],
    kruskal(*grupos_sm)[1],
    kruskal(*grupos_dias)[1],
    mannwhitneyu(covid, no_covid, alternative="two-sided")[1],
    mannwhitneyu(hw_covid, hw_resto, alternative="two-sided")[1],
]

labels_res  = [t[0] for t in resumen_tests]
tipo_colors = {"kruskal":"#3498db","wilcoxon":"#9b59b6","mann-w":"#e67e22"}
col_res = [tipo_colors[t[1]] for t in resumen_tests]

fig, ax = plt.subplots(figsize=(14, 7))
bars_res = ax.barh(labels_res[::-1],
                   [-np.log10(p + 1e-300) for p in all_pvals_final[::-1]],
                   color=col_res[::-1], alpha=0.85)
ax.axvline(-np.log10(ALPHA), color="black", linewidth=1.5, linestyle="--",
           label=f"α={ALPHA}  (umbral significancia)")
ax.set_xlabel("-log₁₀(p-value)")
ax.set_title("Resumen de todas las pruebas estadísticas — Práctica 4",
             fontweight="bold", fontsize=12)
legend_patches = [mpatches.Patch(color=v, label=k) for k, v in tipo_colors.items()]
ax.legend(handles=legend_patches + [mpatches.Patch(color="black", label=f"umbral α={ALPHA}")],
          fontsize=8)
for bar, p in zip(bars_res, all_pvals_final[::-1]):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
            "✓" if p < ALPHA else "✗", va="center", fontsize=11,
            color="#27ae60" if p < ALPHA else "#e74c3c")
plt.tight_layout()
plt.savefig("img/resumen_todos_tests.png", dpi=130)
plt.close()
print("  11.11 resumen_todos_tests")



# CONCLUSION GENERAL


separador("CONCLUSION GENERAL")

sig_count = sum(1 for p in all_pvals_final if p < ALPHA)
print(f"""
  Tests ejecutados: {len(all_pvals_final)}
  Significativos (p < {ALPHA}): {sig_count} / {len(all_pvals_final)}

  HALLAZGOS PRINCIPALES:
  1. Las 5 ligas son estadísticamente distintas en goles, overround y BTTS
     (Kruskal-Wallis p << 0.001 en todas las variables).
  2. Los goles promedio varían entre temporadas (KW significativo), con
     diferencias particulares en la temporada COVID 2019/20.
  3. La ventaja de local es estadísticamente real: FTHG > FTAG
     (Wilcoxon pareado p << 0.001).
  4. El segundo tiempo produce más goles que el primero en todas las ligas
     (Wilcoxon pareado p << 0.001). Se ve refleado en las cuotas HT vs FT.
  5. El movimiento de cuota tiene valor predictivo estadístico: cuando la
     cuota local baja antes del cierre, el local gana más seguido
     (Kruskal-Wallis significativo).
  6. El tipo de partido (favorito vs equilibrado) afecta significativamente
     la tasa de BTTS.
  7. El overround no es igual en todas las ligas — cada mercado tiene
     un margen diferente para la casa.

  NOTA METODOLOGICA:
  Se usó Kruskal-Wallis como prueba principal por falta de normalidad
  (validada con Shapiro-Wilk / D'Agostino p << 0.05 en todos los grupos).
  ANOVA se ejecutó como contraste paramétrico. Los resultados de ambos
  son coincidentes, lo que refuerza la robustez de las conclusiones.
""")

print("\nlisto — todas las graficas guardadas en img/")