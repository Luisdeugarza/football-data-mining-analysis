import pandas as pd
from tabulate import tabulate
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
from typing import List, Tuple

def print_tabulate(df: pd.DataFrame):
    print(tabulate(df, headers=df.columns, tablefmt="orgtbl"))

def get_cmap(n, name="hsv"):
    return matplotlib.colormaps.get_cmap(name).resampled(n)

# distribuciones simuladas
def normalize_distribution(dist: np.ndarray, n: int) -> np.ndarray:
    b = dist - dist.min() + 1e-6
    c = (b / b.sum()) * n
    return np.round(c)

def create_distribution(mean: float, size: int) -> np.ndarray:
    return normalize_distribution(np.random.standard_normal(size), mean * size)

def simulate_goals_distribution(avg_local: float, avg_visit: float, n: int = 500) -> pd.DataFrame:
    """Genera distribución simulada de goles comparada con la real."""
    local_sim  = create_distribution(avg_local, n)
    visit_sim  = create_distribution(avg_visit, n)
    total_sim  = local_sim + visit_sim
    return pd.DataFrame({"local_sim": local_sim, "visit_sim": visit_sim, "total_sim": total_sim})

# funciones auxiliares
def describe_numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    stats = []
    for col in columns:
        if col in df.columns:
            s = df[col].dropna()
            stats.append({
                "column":  col,
                "mean":    round(s.mean(), 3),
                "median":  round(s.median(), 3),
                "std":     round(s.std(), 3),
                "min":     round(s.min(), 3),
                "max":     round(s.max(), 3),
                "q25":     round(s.quantile(0.25), 3),
                "q75":     round(s.quantile(0.75), 3),
                "skew":    round(s.skew(), 3),
                "kurt":    round(s.kurt(), 3),
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

def stats_goals(df):
    return {
        "partidos":      len(df),
        "avg_local":     round(df["FTHG"].mean(), 3),
        "avg_visitante": round(df["FTAG"].mean(), 3),
        "avg_total":     round(df["total_goals"].mean(), 3),
        "std_total":     round(df["total_goals"].std(), 3),
        "max_goles":     int(df["total_goals"].max()),
        "avg_ht_local":  round(df["HTHG"].mean(), 3),
        "avg_ht_visit":  round(df["HTAG"].mean(), 3),
    }

def stats_flags(df, goal_flags):
    return {f: round(df[f].mean() * 100, 2) for f in goal_flags}

def stats_results(df):
    ftr  = df["FTR"].value_counts()
    htr  = df["HTR"].value_counts() if "HTR" in df.columns else {}
    total = len(df)
    return {
        "H":    int(ftr.get("H", 0)), "pct_H": round(ftr.get("H", 0)/total*100, 2),
        "D":    int(ftr.get("D", 0)), "pct_D": round(ftr.get("D", 0)/total*100, 2),
        "A":    int(ftr.get("A", 0)), "pct_A": round(ftr.get("A", 0)/total*100, 2),
        "ht_H": int(htr.get("H", 0)),
        "ht_D": int(htr.get("D", 0)),
        "ht_A": int(htr.get("A", 0)),
    }

def stats_odds(df):
    return {
        "avg_AvgH":      round(df["AvgH"].mean(), 3),
        "avg_AvgD":      round(df["AvgD"].mean(), 3),
        "avg_AvgA":      round(df["AvgA"].mean(), 3),
        "avg_AvgCH":     round(df["AvgCH"].mean(), 3),
        "avg_AvgCA":     round(df["AvgCA"].mean(), 3),
        "avg_overround": round(df["overround"].mean(), 4),
        "med_AvgH":      round(df["AvgH"].median(), 3),
        "med_AvgA":      round(df["AvgA"].median(), 3),
    }

def stats_underdogs(df):
    ud_a     = df[df["is_underdog_away"] == 1]
    ud_h     = df[df["is_underdog_home"] == 1]
    ud_a_ext = df[df["AvgA"] > 8]
    return {
        "ud_away_total":    len(ud_a),
        "ud_away_wins":     int(ud_a["away_win"].sum()),
        "ud_away_pct":      round(ud_a["away_win"].mean()*100, 2) if len(ud_a) else 0,
        "ud_away_avg_odd":  round(ud_a["AvgA"].mean(), 3) if len(ud_a) else 0,
        "ud_home_total":    len(ud_h),
        "ud_home_wins":     int(ud_h["home_win"].sum()),
        "ud_home_pct":      round(ud_h["home_win"].mean()*100, 2) if len(ud_h) else 0,
        "ud_home_avg_odd":  round(ud_h["AvgH"].mean(), 3) if len(ud_h) else 0,
        "ud_extreme_total": len(ud_a_ext),
        "ud_extreme_wins":  int(ud_a_ext["away_win"].sum()),
        "ud_extreme_pct":   round(ud_a_ext["away_win"].mean()*100, 2) if len(ud_a_ext) else 0,
    }

def stats_smart_money(df):
    sh = df[df["odds_move_H"] < -0.1]
    sa = df[df["odds_move_A"] < -0.1]
    sd = df[df["odds_move_D"] < -0.1]
    return {
        "sm_local_partidos":  len(sh),
        "sm_local_gana_pct":  round(sh["home_win"].mean()*100, 2) if len(sh) else 0,
        "sm_visit_partidos":  len(sa),
        "sm_visit_gana_pct":  round(sa["away_win"].mean()*100, 2) if len(sa) else 0,
        "sm_draw_partidos":   len(sd),
        "sm_draw_gana_pct":   round(sd["draw"].mean()*100, 2) if len(sd) else 0,
    }

def stats_comeback(df):
    ht_lose_ft_win_h = df[(df["HTR"] == "A") & (df["FTR"] == "H")]
    ht_lose_ft_win_a = df[(df["HTR"] == "H") & (df["FTR"] == "A")]
    ht_win_ft_lose_h = df[(df["HTR"] == "H") & (df["FTR"] == "A")]
    ht_draw_ft_win_h = df[(df["HTR"] == "D") & (df["FTR"] == "H")]
    ht_draw_ft_win_a = df[(df["HTR"] == "D") & (df["FTR"] == "A")]
    total = len(df)
    return {
        "remontada_local":      len(ht_lose_ft_win_h),
        "pct_remontada_local":  round(len(ht_lose_ft_win_h)/total*100, 2),
        "remontada_visit":      len(ht_lose_ft_win_a),
        "pct_remontada_visit":  round(len(ht_lose_ft_win_a)/total*100, 2),
        "ht_win_ft_lose":       len(ht_win_ft_lose_h),
        "pct_ht_win_ft_lose":   round(len(ht_win_ft_lose_h)/total*100, 2),
        "ht_draw_ft_win_h":     len(ht_draw_ft_win_h),
        "ht_draw_ft_win_a":     len(ht_draw_ft_win_a),
    }

def top_scorers(df, col_team, col_goals, n=5):
    t = df.groupby(col_team)[col_goals].sum().sort_values(ascending=False).head(n)
    return {k: int(v) for k, v in t.items()}

def plot_simulated_vs_real(file_path: str, real_series: pd.Series,
                           avg: float, label: str, color_real: str, color_sim: str):
    """Compara histograma real con distribución simulada."""
    sim = create_distribution(avg, len(real_series))
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = range(0, int(real_series.max()) + 2)
    ax.hist(real_series, bins=bins, alpha=0.6, label=f"real {label}", color=color_real,
            density=True, align="left", edgecolor="white")
    ax.hist(sim, bins=bins, alpha=0.5, label=f"simulado (mean={avg})", color=color_sim,
            density=True, align="left", edgecolor="white")
    ax.set_xlabel("goles"); ax.set_ylabel("densidad"); ax.legend()
    ax.set_title(f"Real vs Simulado — {label}")
    plt.tight_layout()
    plt.savefig(file_path); plt.close()

def draw_er_diagram(file_path: str):
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14); ax.set_ylim(0, 9); ax.axis("off")

    def entity(ax, x, y, title, fields, w=2.2, h=2.6):
        ax.add_patch(mpatches.FancyBboxPatch((x-w/2, y-h/2), w, h,
            boxstyle="round,pad=0.1", linewidth=2,
            edgecolor="#2c3e50", facecolor="#d6eaf8"))
        ax.text(x, y+h/2-0.25, title, ha="center", va="center",
                fontsize=10, fontweight="bold", color="#2c3e50")
        ax.plot([x-w/2, x+w/2], [y+h/2-0.45, y+h/2-0.45], color="#2c3e50", linewidth=1)
        for i, f in enumerate(fields):
            ax.text(x, y+h/2-0.7-i*0.32, f, ha="center", va="center", fontsize=7.5, color="#1a252f")

    def relation(ax, x, y, label, w=1.3, h=0.55):
        ax.add_patch(plt.Polygon([[x,y+h],[x+w/2,y],[x,y-h],[x-w/2,y]],
            closed=True, linewidth=1.5, edgecolor="#2c3e50", facecolor="#fef9e7"))
        ax.text(x, y, label, ha="center", va="center", fontsize=8, color="#2c3e50")

    def arrow(ax, x1, y1, x2, y2, label=""):
        ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
                    arrowprops=dict(arrowstyle="-|>", color="#2c3e50", lw=1.5))
        if label:
            ax.text((x1+x2)/2+0.1, (y1+y2)/2+0.1, label, fontsize=7, color="#555")

    entity(ax, 7,  4.5, "Match",  ["Date","FTHG/FTAG","FTR","HTHG/HTAG","HTR"])
    entity(ax, 2,  7,   "Team",   ["HomeTeam / AwayTeam"])
    entity(ax, 12, 7,   "Odds",   ["B365H/D/A","MaxH/D/A","AvgH/D/A","B365CH/D/A","MaxCH/D/A","AvgCH/D/A"])
    entity(ax, 2,  2,   "League", ["Div (E0/SP1/D1/I1/F1)"])
    entity(ax, 12, 2,   "Season", ["Season (1920..2526)"])
    relation(ax, 4.3, 6.1, "plays");       relation(ax, 9.7, 6.1, "has odds")
    relation(ax, 4.3, 3.2, "belongs to");  relation(ax, 9.7, 3.2, "played in")
    arrow(ax, 2,6.7, 3.5,6.3);  arrow(ax, 5.1,6.1, 5.9,5.1)
    arrow(ax, 12,6.7, 10.5,6.3); arrow(ax, 8.8,5.5, 10.3,6.1, "1:1")
    arrow(ax, 2,2.7, 3.5,3.1);  arrow(ax, 5.1,3.2, 5.9,3.9, "N:1")
    arrow(ax, 12,2.7, 10.5,3.1); arrow(ax, 8.8,3.9, 10.3,3.2, "N:1")
    for x,y,t in [(3,6.6,"N"),(5.5,5.6,"1"),(11,6.6,"1"),(3,2.9,"N"),(11,2.9,"N")]:
        ax.text(x, y, t, fontsize=8, color="#e74c3c", fontweight="bold")
    ax.set_title("Diagrama Entidad-Relacion — European Football Dataset",
                 fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(file_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"diagrama ER guardado en {file_path}")

# setup
os.makedirs("img", exist_ok=True)
np.random.seed(42)

df = pd.read_csv("../Practica 1/data/clean/football_clean.csv", parse_dates=["Date"])
pd.set_option("display.float_format", lambda x: f"{x:.3f}")

df["total_goals"]      = df["FTHG"] + df["FTAG"]
df["ht_goals"]         = df["HTHG"] + df["HTAG"]
df["goal_diff"]        = df["FTHG"] - df["FTAG"]
df["home_win"]         = (df["FTR"] == "H").astype(int)
df["draw"]             = (df["FTR"] == "D").astype(int)
df["away_win"]         = (df["FTR"] == "A").astype(int)
df["btts"]             = ((df["FTHG"] > 0) & (df["FTAG"] > 0)).astype(int)
df["over15"]           = (df["total_goals"] > 1).astype(int)
df["over25"]           = (df["total_goals"] > 2).astype(int)
df["over35"]           = (df["total_goals"] > 3).astype(int)
df["over45"]           = (df["total_goals"] > 4).astype(int)
df["under15"]          = (df["total_goals"] <= 1).astype(int)
df["under25"]          = (df["total_goals"] <= 2).astype(int)
df["under35"]          = (df["total_goals"] <= 3).astype(int)
df["clean_sheet_h"]    = (df["FTAG"] == 0).astype(int)
df["clean_sheet_a"]    = (df["FTHG"] == 0).astype(int)
df["high_scoring"]     = (df["total_goals"] >= 5).astype(int)
df["goalless"]         = (df["total_goals"] == 0).astype(int)
df["imp_prob_H"]       = round(1 / df["AvgH"], 4)
df["imp_prob_D"]       = round(1 / df["AvgD"], 4)
df["imp_prob_A"]       = round(1 / df["AvgA"], 4)
df["overround"]        = round(df["imp_prob_H"] + df["imp_prob_D"] + df["imp_prob_A"], 4)
df["odds_move_H"]      = round(df["AvgCH"] - df["AvgH"], 4)
df["odds_move_A"]      = round(df["AvgCA"] - df["AvgA"], 4)
df["odds_move_D"]      = round(df["AvgCD"] - df["AvgD"], 4)
df["is_underdog_away"] = (df["AvgA"] > 4).astype(int)
df["is_underdog_home"] = (df["AvgH"] > 4).astype(int)
df["odds_margin_ha"]   = round(df["AvgH"] - df["AvgA"], 3)
df["month"]            = df["Date"].dt.month
df["Season_label"]     = df["Season"].map({
    1920:"2019/20", 2021:"2020/21", 2122:"2021/22",
    2223:"2022/23", 2324:"2023/24", 2425:"2024/25", 2526:"2025/26"
})

goal_flags = ["btts","over15","over25","over35","over45",
              "under15","under25","under35",
              "clean_sheet_h","clean_sheet_a","high_scoring","goalless"]

ligas      = sorted(df["Div"].unique())
temporadas = sorted(df["Season"].unique())

season_labels = {
    1920:"2019/20", 2021:"2020/21", 2122:"2021/22",
    2223:"2022/23", 2324:"2023/24", 2425:"2024/25", 2526:"2025/26"
}

resumen_temporadas      = []
resumen_ligas_temporada = []

#INICIO DEL CICLO
for temporada in temporadas:
    slabel = season_labels.get(temporada, str(temporada))
    df_t   = df[df["Season"] == temporada]
    g_t    = stats_goals(df_t)

    print(f"\n{'='*70}")
    print(f"  TEMPORADA {slabel}  ({len(df_t)} partidos)")
    print(f"{'='*70}")

    # estadísticas numéricas con skew y kurtosis
    print(f"\n--- goles generales {slabel} ---")
    print_tabulate(describe_numeric(df_t, ["FTHG","FTAG","HTHG","HTAG","total_goals","ht_goals","goal_diff"]))

    # distribución simulada vs real
    print(f"\n--- distribucion simulada vs real {slabel} ---")
    sim_df = simulate_goals_distribution(g_t["avg_local"], g_t["avg_visitante"], n=len(df_t))
    sim_stats = pd.DataFrame({
        "variable": ["total_real", "total_simulado"],
        "mean":   [round(df_t["total_goals"].mean(), 3), round(sim_df["total_sim"].mean(), 3)],
        "std":    [round(df_t["total_goals"].std(),  3), round(sim_df["total_sim"].std(),  3)],
        "min":    [int(df_t["total_goals"].min()),       int(sim_df["total_sim"].min())],
        "max":    [int(df_t["total_goals"].max()),       int(sim_df["total_sim"].max())],
    })
    print_tabulate(sim_stats)

    # resultados
    print(f"\n--- resultados FT {slabel} ---")
    print_tabulate(describe_categorical(df_t, "FTR"))
    print(f"\n--- resultados HT {slabel} ---")
    print_tabulate(describe_categorical(df_t, "HTR"))

    # flags
    print(f"\n--- btts / over / under / clean sheets {slabel} ---")
    for flag in goal_flags:
        print(f"  {flag}: {int(df_t[flag].sum())} ({round(df_t[flag].mean()*100,2)}%)")

    # remontadas
    print(f"\n--- remontadas {slabel} ---")
    cb = stats_comeback(df_t)
    print(f"  local remonta (pierde HT, gana FT): {cb['remontada_local']} ({cb['pct_remontada_local']}%)")
    print(f"  visit remonta (pierde HT, gana FT): {cb['remontada_visit']} ({cb['pct_remontada_visit']}%)")
    print(f"  gana HT pero pierde FT (local):     {cb['ht_win_ft_lose']} ({cb['pct_ht_win_ft_lose']}%)")
    print(f"  empate HT -> gana local FT:         {cb['ht_draw_ft_win_h']}")
    print(f"  empate HT -> gana visit FT:         {cb['ht_draw_ft_win_a']}")

    # odds
    print(f"\n--- odds generales {slabel} ---")
    print_tabulate(describe_numeric(df_t, ["AvgH","AvgD","AvgA","AvgCH","AvgCD","AvgCA","overround"]))
    print(f"\n--- movimiento de mercado {slabel} ---")
    print_tabulate(describe_numeric(df_t, ["odds_move_H","odds_move_D","odds_move_A"]))

    # underdogs
    ud_t = stats_underdogs(df_t)
    print(f"\n--- underdogs {slabel} ---")
    print(f"  visitante (AvgA>4):  {ud_t['ud_away_total']} | ganan {ud_t['ud_away_wins']} ({ud_t['ud_away_pct']}%) | avg odd {ud_t['ud_away_avg_odd']}")
    print(f"  local (AvgH>4):      {ud_t['ud_home_total']} | ganan {ud_t['ud_home_wins']} ({ud_t['ud_home_pct']}%) | avg odd {ud_t['ud_home_avg_odd']}")
    print(f"  extremos (AvgA>8):   {ud_t['ud_extreme_total']} | ganan {ud_t['ud_extreme_wins']} ({ud_t['ud_extreme_pct']}%)")

    ude_t = df_t[(df_t["AvgA"] > 8) & (df_t["away_win"] == 1)]
    if len(ude_t):
        print(f"\n--- underdogs extremos que ganaron en {slabel} ---")
        print_tabulate(ude_t[["Div","Date","HomeTeam","AwayTeam","AvgH","AvgA","FTHG","FTAG"]].sort_values("AvgA", ascending=False))

    # smart money
    sm_t = stats_smart_money(df_t)
    print(f"\n--- smart money {slabel} ---")
    print(f"  hacia local:     {sm_t['sm_local_partidos']} | gana {sm_t['sm_local_gana_pct']}%")
    print(f"  hacia visitante: {sm_t['sm_visit_partidos']} | gana {sm_t['sm_visit_gana_pct']}%")
    print(f"  hacia empate:    {sm_t['sm_draw_partidos']} | empata {sm_t['sm_draw_gana_pct']}%")

    # alta anotacion
    ha = df_t[df_t["high_scoring"] == 1]
    print(f"\n--- partidos alta anotacion (>=5 goles) {slabel}: {len(ha)} ({round(df_t['high_scoring'].mean()*100,2)}%) ---")
    if len(ha):
        print_tabulate(ha.sort_values("total_goals", ascending=False).head(10)[["Div","Date","HomeTeam","AwayTeam","FTHG","FTAG","FTR"]])

    # acumular
    row_t = {"season": slabel}
    row_t.update(g_t); row_t.update(stats_flags(df_t, goal_flags))
    row_t.update(stats_results(df_t)); row_t.update(stats_odds(df_t))
    row_t.update(ud_t); row_t.update(sm_t); row_t.update(cb)
    resumen_temporadas.append(row_t)

    # ciclo interior: liga dentro de la temporada
    for liga in ligas:
        df_tl = df_t[df_t["Div"] == liga]
        if df_tl.empty:
            continue

        print(f"\n  {'─'*60}")
        print(f"  {liga} | {slabel}  ({len(df_tl)} partidos)")
        print(f"  {'─'*60}")

        g  = stats_goals(df_tl)
        r  = stats_results(df_tl)
        f  = stats_flags(df_tl, goal_flags)
        o  = stats_odds(df_tl)
        ud = stats_underdogs(df_tl)
        sm = stats_smart_money(df_tl)
        cb_tl = stats_comeback(df_tl)

        # distribución simulada por liga/temporada
        sim_tl = simulate_goals_distribution(g["avg_local"], g["avg_visitante"], n=len(df_tl))

        print(f"  goles: local {g['avg_local']} | visit {g['avg_visitante']} | total {g['avg_total']} | std {g['std_total']} | max {g['max_goles']}")
        print(f"  sim total: mean {round(sim_tl['total_sim'].mean(),3)} | std {round(sim_tl['total_sim'].std(),3)}  (real mean {g['avg_total']})")
        print(f"  medio tiempo: local {g['avg_ht_local']} | visit {g['avg_ht_visit']}")
        print(f"  resultados FT: H {r['H']} ({r['pct_H']}%) | D {r['D']} ({r['pct_D']}%) | A {r['A']} ({r['pct_A']}%)")
        print(f"  resultados HT: H {r['ht_H']} | D {r['ht_D']} | A {r['ht_A']}")
        print(f"  btts {f['btts']}% | over1.5 {f['over15']}% | over2.5 {f['over25']}% | over3.5 {f['over35']}% | over4.5 {f['over45']}%")
        print(f"  under1.5 {f['under15']}% | under2.5 {f['under25']}% | under3.5 {f['under35']}%")
        print(f"  clean sheet local {f['clean_sheet_h']}% | visit {f['clean_sheet_a']}% | sin goles {f['goalless']}% | +5 goles {f['high_scoring']}%")
        print(f"  remontada local {cb_tl['remontada_local']} ({cb_tl['pct_remontada_local']}%) | visit {cb_tl['remontada_visit']} ({cb_tl['pct_remontada_visit']}%)")
        print(f"  odds avg: H {o['avg_AvgH']} | D {o['avg_AvgD']} | A {o['avg_AvgA']} | CH {o['avg_AvgCH']} | CA {o['avg_AvgCA']} | overround {o['avg_overround']}")
        print(f"  mov mercado: H {round(df_tl['odds_move_H'].mean(),4)} | D {round(df_tl['odds_move_D'].mean(),4)} | A {round(df_tl['odds_move_A'].mean(),4)}")
        print(f"  ud visit: {ud['ud_away_total']} ({ud['ud_away_pct']}% ganan, avg {ud['ud_away_avg_odd']}) | ud local: {ud['ud_home_total']} ({ud['ud_home_pct']}% ganan)")
        print(f"  ud extremos: {ud['ud_extreme_total']} | ganan {ud['ud_extreme_wins']} ({ud['ud_extreme_pct']}%)")
        print(f"  sm local: {sm['sm_local_partidos']} ({sm['sm_local_gana_pct']}%) | visit: {sm['sm_visit_partidos']} ({sm['sm_visit_gana_pct']}%) | empate: {sm['sm_draw_partidos']} ({sm['sm_draw_gana_pct']}%)")
        print(f"  top goles local:        {top_scorers(df_tl, 'HomeTeam', 'FTHG')}")
        print(f"  top goles visit:        {top_scorers(df_tl, 'AwayTeam', 'FTAG')}")
        print(f"  top clean sheets local: {top_scorers(df_tl, 'HomeTeam', 'clean_sheet_h')}")

        ude_tl = df_tl[(df_tl["AvgA"] > 8) & (df_tl["away_win"] == 1)]
        if len(ude_tl):
            print(f"  underdogs extremos que ganaron:")
            print_tabulate(ude_tl[["Date","HomeTeam","AwayTeam","AvgH","AvgA","FTHG","FTAG"]].sort_values("AvgA", ascending=False))

        row_tl = {"season": slabel, "liga": liga}
        row_tl.update(g); row_tl.update(f); row_tl.update(r)
        row_tl.update(o); row_tl.update(ud); row_tl.update(sm); row_tl.update(cb_tl)
        resumen_ligas_temporada.append(row_tl)

#entre temporadas
print(f"\n{'='*70}")
print("  COMPARACION ENTRE TEMPORADAS")
print(f"{'='*70}")

df_rt = pd.DataFrame(resumen_temporadas)

print("\n=== goles por temporada ===")
print_tabulate(df_rt[["season","partidos","avg_local","avg_visitante","avg_total","std_total","max_goles","avg_ht_local","avg_ht_visit"]])

print("\n=== btts / over / under por temporada ===")
print_tabulate(df_rt[["season","btts","over15","over25","over35","over45","under15","under25","under35"]])

print("\n=== clean sheets y partidos especiales por temporada ===")
print_tabulate(df_rt[["season","clean_sheet_h","clean_sheet_a","high_scoring","goalless"]])

print("\n=== resultados FT por temporada ===")
print_tabulate(df_rt[["season","H","pct_H","D","pct_D","A","pct_A"]])

print("\n=== resultados HT por temporada ===")
print_tabulate(df_rt[["season","ht_H","ht_D","ht_A"]])

print("\n=== remontadas por temporada ===")
print_tabulate(df_rt[["season","remontada_local","pct_remontada_local","remontada_visit","pct_remontada_visit","ht_win_ft_lose","pct_ht_win_ft_lose"]])

print("\n=== odds y overround por temporada ===")
print_tabulate(df_rt[["season","avg_AvgH","avg_AvgD","avg_AvgA","avg_AvgCH","avg_AvgCA","avg_overround"]])

print("\n=== underdogs por temporada ===")
print_tabulate(df_rt[["season","ud_away_total","ud_away_wins","ud_away_pct","ud_away_avg_odd","ud_home_total","ud_home_wins","ud_home_pct","ud_extreme_total","ud_extreme_pct"]])

print("\n=== smart money por temporada ===")
print_tabulate(df_rt[["season","sm_local_partidos","sm_local_gana_pct","sm_visit_partidos","sm_visit_gana_pct","sm_draw_partidos","sm_draw_gana_pct"]])

# distribución simulada global por temporada
print("\n=== distribucion simulada global vs real por temporada ===")
sim_rows = []
for row in resumen_temporadas:
    sim = simulate_goals_distribution(row["avg_local"], row["avg_visitante"], n=row["partidos"])
    sim_rows.append({
        "season":        row["season"],
        "real_mean":     row["avg_total"],
        "sim_mean":      round(sim["total_sim"].mean(), 3),
        "real_std":      row["std_total"],
        "sim_std":       round(sim["total_sim"].std(), 3),
        "diff_mean":     round(abs(row["avg_total"] - sim["total_sim"].mean()), 4),
    })
print_tabulate(pd.DataFrame(sim_rows))

#entre ligas
print(f"\n{'='*70}")
print("  COMPARACION ENTRE LIGAS (todo el periodo)")
print(f"{'='*70}")

print("\n=== goles por liga ===")
gl = df.groupby("Div").agg(
    partidos=("total_goals","count"),
    avg_local=("FTHG","mean"), avg_visitante=("FTAG","mean"),
    avg_total=("total_goals","mean"), std_total=("total_goals","std"),
    max_goles=("total_goals","max"),
    avg_ht_local=("HTHG","mean"), avg_ht_visit=("HTAG","mean"),
).round(3).reset_index()
print_tabulate(gl)

print("\n=== skewness y kurtosis de goles por liga ===")
sk_rows = []
for liga in ligas:
    s = df[df["Div"] == liga]["total_goals"]
    sk_rows.append({"liga": liga, "skew": round(s.skew(),3), "kurt": round(s.kurt(),3),
                    "mean": round(s.mean(),3), "std": round(s.std(),3)})
print_tabulate(pd.DataFrame(sk_rows))

print("\n=== distribucion simulada vs real por liga ===")
sim_liga_rows = []
for liga in ligas:
    sub = df[df["Div"] == liga]
    g_l = stats_goals(sub)
    sim = simulate_goals_distribution(g_l["avg_local"], g_l["avg_visitante"], n=g_l["partidos"])
    sim_liga_rows.append({
        "liga":      liga,
        "real_mean": g_l["avg_total"],
        "sim_mean":  round(sim["total_sim"].mean(), 3),
        "real_std":  g_l["std_total"],
        "sim_std":   round(sim["total_sim"].std(), 3),
        "diff_mean": round(abs(g_l["avg_total"] - sim["total_sim"].mean()), 4),
    })
print_tabulate(pd.DataFrame(sim_liga_rows))

print("\n=== btts / over / under por liga ===")
gf_liga = df.groupby("Div")[goal_flags].mean().round(4) * 100
print_tabulate(gf_liga.round(2).reset_index())

print("\n=== resultados por liga ===")
ftr_l = df.groupby(["Div","FTR"]).size().unstack(fill_value=0).reset_index()
ftr_l["total"] = ftr_l[["A","D","H"]].sum(axis=1)
ftr_l["pct_H"] = round(ftr_l["H"]/ftr_l["total"]*100, 2)
ftr_l["pct_D"] = round(ftr_l["D"]/ftr_l["total"]*100, 2)
ftr_l["pct_A"] = round(ftr_l["A"]/ftr_l["total"]*100, 2)
print_tabulate(ftr_l)

print("\n=== remontadas por liga ===")
cb_liga = []
for liga in ligas:
    cb = stats_comeback(df[df["Div"] == liga]); cb["liga"] = liga; cb_liga.append(cb)
print_tabulate(pd.DataFrame(cb_liga)[["liga","remontada_local","pct_remontada_local","remontada_visit","pct_remontada_visit","ht_win_ft_lose","pct_ht_win_ft_lose"]])

print("\n=== odds por liga ===")
ol = df.groupby("Div")[["AvgH","AvgD","AvgA","AvgCH","AvgCD","AvgCA","overround"]].mean().round(3).reset_index()
print_tabulate(ol)

print("\n=== movimiento de mercado por liga ===")
ml = df.groupby("Div")[["odds_move_H","odds_move_D","odds_move_A"]].agg(["mean","std"]).round(4).reset_index()
print_tabulate(ml)

print("\n=== underdogs por liga ===")
for liga in ligas:
    ud = stats_underdogs(df[df["Div"] == liga])
    print(f"  {liga} — ud visit: {ud['ud_away_total']} ({ud['ud_away_pct']}% ganan, avg {ud['ud_away_avg_odd']}) | ud local: {ud['ud_home_total']} ({ud['ud_home_pct']}% ganan) | extremos: {ud['ud_extreme_total']} ({ud['ud_extreme_pct']}%)")

print("\n=== smart money por liga ===")
for liga in ligas:
    sm = stats_smart_money(df[df["Div"] == liga])
    print(f"  {liga} — local: {sm['sm_local_partidos']} ({sm['sm_local_gana_pct']}%) | visit: {sm['sm_visit_partidos']} ({sm['sm_visit_gana_pct']}%) | empate: {sm['sm_draw_partidos']} ({sm['sm_draw_gana_pct']}%)")

print("\n=== top 15 goleadores totales ===")
hg = df.groupby("HomeTeam")["FTHG"].sum().reset_index().rename(columns={"HomeTeam":"equipo","FTHG":"goles_local"})
ag = df.groupby("AwayTeam")["FTAG"].sum().reset_index().rename(columns={"AwayTeam":"equipo","FTAG":"goles_visit"})
top_goles = hg.merge(ag, on="equipo", how="outer").fillna(0)
top_goles[["goles_local","goles_visit"]] = top_goles[["goles_local","goles_visit"]].astype(int)
top_goles["total"] = top_goles["goles_local"] + top_goles["goles_visit"]
print_tabulate(top_goles.sort_values("total", ascending=False).head(15))

print("\n=== mayor % victorias local (min 50 partidos) ===")
hw = df.groupby("HomeTeam").agg(partidos=("home_win","count"), victorias=("home_win","sum"))
hw = hw[hw["partidos"] >= 50]; hw["pct_win"] = round(hw["victorias"]/hw["partidos"]*100, 2)
print_tabulate(hw.sort_values("pct_win", ascending=False).head(15).reset_index())

print("\n=== mayor % victorias visitante (min 50 partidos) ===")
aw = df.groupby("AwayTeam").agg(partidos=("away_win","count"), victorias=("away_win","sum"))
aw = aw[aw["partidos"] >= 50]; aw["pct_win"] = round(aw["victorias"]/aw["partidos"]*100, 2)
print_tabulate(aw.sort_values("pct_win", ascending=False).head(15).reset_index())

print("\n=== mayor % empates (min 50 partidos) ===")
dr = df.groupby("HomeTeam").agg(partidos=("draw","count"), empates=("draw","sum"))
dr = dr[dr["partidos"] >= 50]; dr["pct_draw"] = round(dr["empates"]/dr["partidos"]*100, 2)
print_tabulate(dr.sort_values("pct_draw", ascending=False).head(15).reset_index())

print("\n=== clean sheets local (min 50 partidos) ===")
cs_h = df.groupby("HomeTeam").agg(partidos=("clean_sheet_h","count"), cs=("clean_sheet_h","sum"))
cs_h = cs_h[cs_h["partidos"] >= 50]; cs_h["pct_cs"] = round(cs_h["cs"]/cs_h["partidos"]*100, 2)
print_tabulate(cs_h.sort_values("pct_cs", ascending=False).head(10).reset_index())

print("\n=== clean sheets visitante (min 50 partidos) ===")
cs_a = df.groupby("AwayTeam").agg(partidos=("clean_sheet_a","count"), cs=("clean_sheet_a","sum"))
cs_a = cs_a[cs_a["partidos"] >= 50]; cs_a["pct_cs"] = round(cs_a["cs"]/cs_a["partidos"]*100, 2)
print_tabulate(cs_a.sort_values("pct_cs", ascending=False).head(10).reset_index())

print("\n=== partidos sin goles por liga ===")
gl0 = df.groupby("Div")["goalless"].agg(["sum","mean"]).reset_index()
gl0.columns = ["Div","total_goalless","pct_goalless"]
gl0["pct_goalless"] = round(gl0["pct_goalless"]*100, 2)
print_tabulate(gl0)

print("\n=== partidos alta anotacion (>=5 goles) por liga ===")
ha_l = df.groupby("Div")["high_scoring"].agg(["sum","mean"]).reset_index()
ha_l.columns = ["Div","total_high","pct_high"]
ha_l["pct_high"] = round(ha_l["pct_high"]*100, 2)
print_tabulate(ha_l)

print("\n=== correlacion general ===")
corr_cols = ["total_goals","FTHG","FTAG","AvgH","AvgA","AvgD",
             "overround","odds_move_H","odds_move_A","imp_prob_H","imp_prob_A",
             "btts","over25","clean_sheet_h","clean_sheet_a"]
print(round(df[corr_cols].corr(), 3))

print("\n=== por mes (goles y resultados) ===")
gm = df.groupby("month").agg(
    partidos=("total_goals","count"), avg_total=("total_goals","mean"),
    pct_H=("home_win","mean"), pct_D=("draw","mean"), pct_A=("away_win","mean"),
    pct_btts=("btts","mean"), pct_over25=("over25","mean"),
).round(3).reset_index()
print_tabulate(gm)

# ============================================================
# GRAFICAS
# ============================================================
print("\n--- generando graficas ---")
cmap = get_cmap(len(ligas) + 1)

draw_er_diagram("img/er_diagram.png")

scatter_group_by("img/odds_apertura_por_liga.png",    df, "AvgH",     "AvgA",     "Div")
scatter_group_by("img/goles_local_vs_visitante.png",  df, "FTHG",     "FTAG",     "Div")
scatter_group_by("img/odds_apertura_vs_cierre_H.png", df, "AvgH",     "AvgCH",    "Div")
scatter_group_by("img/odds_apertura_vs_cierre_A.png", df, "AvgA",     "AvgCA",    "Div")
scatter_group_by("img/implied_prob_H_vs_A.png",       df, "imp_prob_H","imp_prob_A","Div")
scatter_group_by("img/movimiento_mercado.png",         df, "odds_move_H","odds_move_A","Div")

# líneas temporada/liga
for col, ylabel, fname in [
    ("total_goals", "goles promedio",    "goles_por_temporada_y_liga"),
    ("over25",      "% over 2.5",        "over25_por_temporada_y_liga"),
    ("btts",        "% btts",            "btts_por_temporada_y_liga"),
    ("home_win",    "% victoria local",  "pct_home_win_por_temporada_liga"),
    ("clean_sheet_h","% clean sheet loc","clean_sheet_por_temporada_liga"),
]:
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, liga in enumerate(ligas):
        sub = df[df["Div"] == liga].groupby("Season")[col].mean()
        if col in ("over25","btts","home_win","clean_sheet_h"):
            sub = sub * 100
        ax.plot(sub.index.astype(str), sub.values, marker="o", label=liga, color=cmap(i))
    ax.set_xlabel("temporada"); ax.set_ylabel(ylabel); ax.legend()
    plt.savefig(f"img/{fname}.png"); plt.close()

# pies de resultados
fig, axes = plt.subplots(1, 5, figsize=(20, 5))
for i, liga in enumerate(ligas):
    sub = df[df["Div"] == liga]["FTR"].value_counts()
    axes[i].pie(sub.values, labels=sub.index, autopct="%1.1f%%"); axes[i].set_title(liga)
plt.savefig("img/resultados_pie_por_liga.png"); plt.close()

# barras btts/over/under/cs
fig, ax = plt.subplots(figsize=(13, 5))
x = np.arange(len(ligas)); w = 0.13
for j, (mlabel, col) in enumerate([("btts","btts"),("over1.5","over15"),("over2.5","over25"),
                                    ("over3.5","over35"),("under2.5","under25"),("cs_h","clean_sheet_h")]):
    ax.bar(x + j*w, [df[df["Div"]==l][col].mean()*100 for l in ligas], w, label=mlabel)
ax.set_xticks(x + w*2.5); ax.set_xticklabels(ligas); ax.set_ylabel("%"); ax.legend()
plt.savefig("img/btts_over_under_cs_por_liga.png"); plt.close()

# histogramas goles por liga
fig, axes = plt.subplots(1, 5, figsize=(20, 5))
for i, liga in enumerate(ligas):
    axes[i].hist(df[df["Div"]==liga]["total_goals"], bins=range(0,12), edgecolor="black", align="left")
    axes[i].set_title(liga); axes[i].set_xlabel("goles")
plt.savefig("img/distribucion_goles_por_liga.png"); plt.close()

# overround
fig, ax = plt.subplots(figsize=(10, 5))
for liga in ligas:
    ax.hist(df[df["Div"]==liga]["overround"], bins=40, alpha=0.5, label=liga)
ax.set_xlabel("overround"); ax.legend()
plt.savefig("img/overround_por_liga.png"); plt.close()

# underdog gana vs pierde
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df[(df["is_underdog_away"]==1)&(df["away_win"]==1)]["AvgA"], bins=30, alpha=0.6, label="underdog gana")
ax.hist(df[(df["is_underdog_away"]==1)&(df["away_win"]==0)]["AvgA"], bins=30, alpha=0.6, label="underdog pierde")
ax.set_xlabel("AvgA"); ax.legend()
plt.savefig("img/underdog_win_vs_lose.png"); plt.close()

# goles totales
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df["total_goals"], bins=range(0,15), edgecolor="black", align="left")
ax.set_xlabel("goles en el partido"); ax.set_ylabel("frecuencia")
plt.savefig("img/distribucion_goles_totales.png"); plt.close()

# boxplots 
fig, ax = plt.subplots(figsize=(10, 6))
df.boxplot(column="total_goals", by="Div", ax=ax)
ax.set_title("Distribucion de goles por liga"); plt.suptitle("")
plt.savefig("img/boxplot_goles_por_liga.png"); plt.close()

fig, ax = plt.subplots(figsize=(10, 6))
df.boxplot(column="AvgH", by="Div", ax=ax)
ax.set_title("Distribucion odds local por liga"); plt.suptitle("")
plt.savefig("img/boxplot_odds_H_por_liga.png"); plt.close()

fig, ax = plt.subplots(figsize=(10, 6))
df.boxplot(column="odds_move_H", by="Div", ax=ax)
ax.set_title("Movimiento mercado local por liga"); plt.suptitle("")
plt.savefig("img/boxplot_mov_H_por_liga.png"); plt.close()

fig, ax = plt.subplots(figsize=(12, 6))
df.boxplot(column="total_goals", by="Season_label", ax=ax)
ax.set_title("Distribucion de goles por temporada"); plt.suptitle("")
plt.xticks(rotation=45)
plt.savefig("img/boxplot_goles_por_temporada.png"); plt.close()

# real vs simulado por liga (una grafica por liga)
for liga in ligas:
    sub = df[df["Div"] == liga]
    g_l = stats_goals(sub)
    plot_simulated_vs_real(
        f"img/sim_vs_real_{liga}.png",
        sub["total_goals"], g_l["avg_total"],
        liga, "#2ecc71", "#e74c3c"
    )

print("guardadas imagenes en img/")