import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
from tabulate import tabulate
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
from typing import List

DAY_NAMES  = {0:"Lunes",1:"Martes",2:"Miercoles",3:"Jueves",4:"Viernes",5:"Sabado",6:"Domingo"}
DAY_ORDER  = ["Lunes","Martes","Miercoles","Jueves","Viernes","Sabado","Domingo"]
LIGAS_NAME = {"E0":"Premier League","SP1":"La Liga","D1":"Bundesliga","I1":"Serie A","F1":"Ligue 1"}

# funciones auxiliares para estadistica descriptiva


def print_tabulate(df: pd.DataFrame):
    print(tabulate(df, headers=df.columns, tablefmt="orgtbl"))

def get_cmap(n, name="hsv"):
    return matplotlib.colormaps.get_cmap(name).resampled(n)

def describe_numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    rows = []
    for col in columns:
        if col not in df.columns: continue
        s = df[col].dropna()
        q25, q75, mean, std = s.quantile(0.25), s.quantile(0.75), s.mean(), s.std()
        rows.append({
            "column": col,
            "mean":   round(mean,3),
            "mode":   round(float(s.mode()[0]),3) if not s.mode().empty else None,
            "median": round(s.median(),3),
            "var":    round(s.var(),3),
            "std":    round(std,3),
            "cv_%":   round(std/mean*100,2) if mean!=0 else None,
            "min":    round(s.min(),3),   "max":   round(s.max(),3),
            "range":  round(s.max()-s.min(),3),
            "q25":    round(q25,3),       "q75":   round(q75,3),
            "iqr":    round(q75-q25,3),
            "p10":    round(s.quantile(0.10),3), "p90": round(s.quantile(0.90),3),
            "skew":   round(s.skew(),3),  "kurt":  round(s.kurt(),3),
        })
    return pd.DataFrame(rows)

def describe_categorical(df: pd.DataFrame, col: str) -> pd.DataFrame:
    c = df[col].value_counts().reset_index()
    c.columns = [col,"count"]
    c["pct"] = round(c["count"]/len(df)*100,2)
    return c

def stats_distribution(df: pd.DataFrame, col: str, bins: list) -> pd.DataFrame:
    labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]
    cut  = pd.cut(df[col], bins=bins, labels=labels, right=False)
    freq = cut.value_counts().sort_index().reset_index()
    freq.columns = ["rango","freq"]
    freq["freq_%"]    = round(freq["freq"]/freq["freq"].sum()*100,2)
    freq["freq_acum"] = round(freq["freq_%"].cumsum(),2)
    return freq

# goles

def stats_goals(df) -> dict:
    return {
        "partidos":      len(df),
        "avg_local":     round(df["FTHG"].mean(),3),
        "avg_visitante": round(df["FTAG"].mean(),3),
        "avg_total":     round(df["total_goals"].mean(),3),
        "moda_total":    int(df["total_goals"].mode()[0]) if not df["total_goals"].mode().empty else None,
        "std_total":     round(df["total_goals"].std(),3),
        "max_goles":     int(df["total_goals"].max()),
        "avg_ht":        round(df["ht_goals"].mean(),3),
        "moda_ht":       int(df["ht_goals"].mode()[0]) if not df["ht_goals"].mode().empty else None,
        "avg_st":        round(df["second_half_goals"].mean(),3),
        "moda_st":       int(df["second_half_goals"].mode()[0]) if not df["second_half_goals"].mode().empty else None,
        "ratio_ht_st":   round(df["ht_goals"].mean()/df["second_half_goals"].mean(),3)
                         if df["second_half_goals"].mean()>0 else None,
        "corr_ht_ft":    round(df["ht_goals"].corr(df["total_goals"]),4),
        "corr_ht_st":    round(df["ht_goals"].corr(df["second_half_goals"]),4),
    }

def stats_goals_mode_table(df) -> pd.DataFrame:
    """Tabla comparativa moda vs media para todas las variables de goles."""
    cols = ["FTHG","FTAG","total_goals","HTHG","HTAG","ht_goals","second_half_goals"]
    rows = []
    for c in cols:
        if c not in df.columns: continue
        s = df[c].dropna()
        rows.append({
            "variable": c,
            "mean":     round(s.mean(),3),
            "moda":     int(s.mode()[0]) if not s.mode().empty else None,
            "median":   round(s.median(),3),
            "std":      round(s.std(),3),
            "cv_%":     round(s.std()/s.mean()*100,2) if s.mean()!=0 else None,
        })
    return pd.DataFrame(rows)

def stats_halftime_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla HT=0/1/2/3+: que pasa en el FT segun como va el primer tiempo."""
    total = len(df)
    rows = []
    for label, mask in [("HT=0", df["ht_goals"]==0),("HT=1", df["ht_goals"]==1),
                         ("HT=2", df["ht_goals"]==2),("HT=3+",df["ht_goals"]>=3)]:
        sub = df[mask]
        if len(sub)==0: continue
        rows.append({
            "ht_goles":     label,
            "partidos":     len(sub),
            "pct_total":    round(len(sub)/total*100,2),
            "avg_ft":       round(sub["total_goals"].mean(),3),
            "moda_ft":      int(sub["total_goals"].mode()[0]) if not sub["total_goals"].mode().empty else None,
            "avg_st":       round(sub["second_half_goals"].mean(),3),
            "pct_over25":   round(sub["over25"].mean()*100,2),
            "pct_btts":     round(sub["btts"].mean()*100,2),
            "pct_5plus":    round(sub["high_scoring"].mean()*100,2),
            "pct_goalless": round((sub["total_goals"]==0).mean()*100,2),
            "pct_H":        round(sub["home_win"].mean()*100,2),
            "pct_D":        round(sub["draw"].mean()*100,2),
            "pct_A":        round(sub["away_win"].mean()*100,2),
        })
    return pd.DataFrame(rows)

def stats_high_scoring_analysis(df: pd.DataFrame) -> dict:
    """En partidos con 5+ goles: como iban al descanso."""
    hs = df[df["high_scoring"]==1]
    if len(hs)==0: return {}
    return {
        "total_5plus":       len(hs),
        "pct_5plus":         round(len(hs)/len(df)*100,2),
        "avg_ht_en_5plus":   round(hs["ht_goals"].mean(),3),
        "pct_ht0_en_5plus":  round((hs["ht_goals"]==0).mean()*100,2),
        "pct_ht1_en_5plus":  round((hs["ht_goals"]==1).mean()*100,2),
        "pct_ht2_en_5plus":  round((hs["ht_goals"]==2).mean()*100,2),
        "pct_ht3p_en_5plus": round((hs["ht_goals"]>=3).mean()*100,2),
        "avg_st_en_5plus":   round(hs["second_half_goals"].mean(),3),
        "max_goles":         int(hs["total_goals"].max()),
        "pct_btts_en_5plus": round(hs["btts"].mean()*100,2),
    }

def stats_over_under_goals(df: pd.DataFrame) -> pd.DataFrame:
    """Media de goles y perfil en partidos OVER vs UNDER por cada umbral."""
    rows = []
    for thr, lbl in [(1.5,"1.5"),(2.5,"2.5"),(3.5,"3.5"),(4.5,"4.5")]:
        ov = df[df["total_goals"]>thr]; un = df[df["total_goals"]<=thr]
        rows.append({
            "umbral":          lbl,
            "n_over":          len(ov),
            "pct_over_%":      round(len(ov)/len(df)*100,2),
            "avg_goles_over":  round(ov["total_goals"].mean(),3) if len(ov) else 0,
            "moda_over":       int(ov["total_goals"].mode()[0])  if len(ov) and not ov["total_goals"].mode().empty else None,
            "avg_ht_over":     round(ov["ht_goals"].mean(),3)    if len(ov) else 0,
            "avg_st_over":     round(ov["second_half_goals"].mean(),3) if len(ov) else 0,
            "pct_btts_over":   round(ov["btts"].mean()*100,2)    if len(ov) else 0,
            "n_under":         len(un),
            "avg_goles_under": round(un["total_goals"].mean(),3) if len(un) else 0,
            "moda_under":      int(un["total_goals"].mode()[0])  if len(un) and not un["total_goals"].mode().empty else None,
            "avg_ht_under":    round(un["ht_goals"].mean(),3)    if len(un) else 0,
            "pct_btts_under":  round(un["btts"].mean()*100,2)    if len(un) else 0,
        })
    return pd.DataFrame(rows)

def stats_btts_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Perfil comparativo partidos BTTS vs NO BTTS."""
    rows = []
    for lbl, mask in [("btts", df["btts"]==1), ("no_btts", df["btts"]==0)]:
        sub = df[mask]
        if len(sub)==0: continue
        rows.append({
            "tipo":           lbl,
            "partidos":       len(sub),
            "avg_goles":      round(sub["total_goals"].mean(),3),
            "moda_goles":     int(sub["total_goals"].mode()[0]) if not sub["total_goals"].mode().empty else None,
            "avg_ht":         round(sub["ht_goals"].mean(),3),
            "avg_st":         round(sub["second_half_goals"].mean(),3),
            "pct_over25":     round(sub["over25"].mean()*100,2),
            "pct_over35":     round(sub["over35"].mean()*100,2),
            "pct_high":       round(sub["high_scoring"].mean()*100,2),
            "pct_goalless":   round(sub["goalless"].mean()*100,2),
            "pct_H":          round(sub["home_win"].mean()*100,2),
            "pct_D":          round(sub["draw"].mean()*100,2),
            "avg_AvgH":       round(sub["AvgH"].mean(),3),
            "avg_AvgA":       round(sub["AvgA"].mean(),3),
        })
    return pd.DataFrame(rows)

# cuotas

def stats_odds_mode(df: pd.DataFrame) -> dict:
    """Moda y mediana de cuotas ganadoras (apertura y cierre) redondeadas a 1 decimal."""
    def mval(s):
        if s.empty: return None, 0
        m = s.round(1).value_counts(); return float(m.index[0]), int(m.iloc[0])
    def med(s): return round(float(s.median()),3) if not s.empty else None
    hw=df[df["FTR"]=="H"]; aw=df[df["FTR"]=="A"]; dw=df[df["FTR"]=="D"]
    ua=df[(df["is_underdog_away"]==1)&(df["away_win"]==1)]
    uh=df[(df["is_underdog_home"]==1)&(df["home_win"]==1)]
    return {
        "moda_ap_H":    mval(hw["AvgH"])[0],  "cnt_ap_H":    mval(hw["AvgH"])[1],  "med_ap_H":    med(hw["AvgH"]),
        "moda_ci_H":    mval(hw["AvgCH"])[0], "cnt_ci_H":    mval(hw["AvgCH"])[1], "med_ci_H":    med(hw["AvgCH"]),
        "moda_ap_A":    mval(aw["AvgA"])[0],  "cnt_ap_A":    mval(aw["AvgA"])[1],  "med_ap_A":    med(aw["AvgA"]),
        "moda_ci_A":    mval(aw["AvgCA"])[0], "cnt_ci_A":    mval(aw["AvgCA"])[1], "med_ci_A":    med(aw["AvgCA"]),
        "moda_ap_D":    mval(dw["AvgD"])[0],  "med_ap_D":    med(dw["AvgD"]),
        "moda_ud_ap_A": mval(ua["AvgA"])[0],  "cnt_ud_ap_A": mval(ua["AvgA"])[1],  "med_ud_ap_A": med(ua["AvgA"]),
        "moda_ud_ap_H": mval(uh["AvgH"])[0],  "med_ud_ap_H": med(uh["AvgH"]),
    }

def stats_odds_gap(df: pd.DataFrame) -> pd.DataFrame:
    """Analisis segun diferencia absoluta de cuotas H vs A:
    identicas (partido abierto) → extremas (favorito claro)."""
    df2 = df.copy()
    df2["odds_gap"] = (df2["AvgH"]-df2["AvgA"]).abs()
    bins   = [0,0.3,0.75,1.5,3.0,5.0,100]
    labels = ["0-0.3 identicas","0.3-0.75 parejas","0.75-1.5 algo dispares",
              "1.5-3.0 dispares","3.0-5.0 muy dispares","5.0+ extremas"]
    df2["gap_cat"] = pd.cut(df2["odds_gap"],bins=bins,labels=labels,right=False)
    rows = []
    for cat in labels:
        sub = df2[df2["gap_cat"]==cat]
        if len(sub)<5: continue
        rows.append({
            "equilibrio":   cat,
            "partidos":     len(sub),
            "pct_total":    round(len(sub)/len(df)*100,2),
            "avg_goles":    round(sub["total_goals"].mean(),3),
            "moda_goles":   int(sub["total_goals"].mode()[0]) if not sub["total_goals"].mode().empty else None,
            "avg_ht":       round(sub["ht_goals"].mean(),3),
            "avg_st":       round(sub["second_half_goals"].mean(),3),
            "pct_btts":     round(sub["btts"].mean()*100,2),
            "pct_over25":   round(sub["over25"].mean()*100,2),
            "pct_over35":   round(sub["over35"].mean()*100,2),
            "pct_goalless": round(sub["goalless"].mean()*100,2),
            "pct_H":        round(sub["home_win"].mean()*100,2),
            "pct_D":        round(sub["draw"].mean()*100,2),
            "pct_A":        round(sub["away_win"].mean()*100,2),
            "avg_AvgH":     round(sub["AvgH"].mean(),3),
            "avg_AvgA":     round(sub["AvgA"].mean(),3),
        })
    return pd.DataFrame(rows)

def stats_implied_vs_real(df) -> dict:
    return {
        "imp_H":  round(df["imp_prob_H"].mean()*100,2),  "real_H": round(df["home_win"].mean()*100,2),
        "diff_H": round((df["imp_prob_H"].mean()-df["home_win"].mean())*100,2),
        "imp_A":  round(df["imp_prob_A"].mean()*100,2),  "real_A": round(df["away_win"].mean()*100,2),
        "diff_A": round((df["imp_prob_A"].mean()-df["away_win"].mean())*100,2),
        "imp_D":  round(df["imp_prob_D"].mean()*100,2),  "real_D": round(df["draw"].mean()*100,2),
        "diff_D": round((df["imp_prob_D"].mean()-df["draw"].mean())*100,2),
    }

def stats_win_rate_by_odd(df) -> pd.DataFrame:
    """Tasa de acierto real vs implied probability por rango de odd local."""
    bins   = [1.0,1.3,1.5,1.75,2.0,2.5,3.0,4.0,6.0,25.0]
    labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]
    d2 = df.copy(); d2["rng"] = pd.cut(d2["AvgH"],bins=bins,labels=labels,right=False)
    rows = []
    for rng in labels:
        sub = d2[d2["rng"]==rng]
        if len(sub)<5: continue
        rows.append({
            "rango_H":  rng, "partidos": len(sub),
            "imp_%":    round(1/sub["AvgH"].mean()*100,2),
            "real_%":   round(sub["home_win"].mean()*100,2),
            "diff":     round(sub["home_win"].mean()*100 - 1/sub["AvgH"].mean()*100,2),
        })
    return pd.DataFrame(rows)

# resultados, flags y remontadas

def stats_flags(df, goal_flags) -> dict:
    return {f: round(df[f].mean()*100,2) for f in goal_flags}

def stats_results(df) -> dict:
    ftr=df["FTR"].value_counts(); htr=df["HTR"].value_counts() if "HTR" in df.columns else {}
    total=len(df)
    return {
        "H":int(ftr.get("H",0)),   "pct_H": round(ftr.get("H",0)/total*100,2),
        "D":int(ftr.get("D",0)),   "pct_D": round(ftr.get("D",0)/total*100,2),
        "A":int(ftr.get("A",0)),   "pct_A": round(ftr.get("A",0)/total*100,2),
        "ht_H":int(htr.get("H",0)),"ht_D": int(htr.get("D",0)),"ht_A":int(htr.get("A",0)),
    }

def stats_comeback(df) -> dict:
    rl=df[(df["HTR"]=="A")&(df["FTR"]=="H")]; rv=df[(df["HTR"]=="H")&(df["FTR"]=="A")]
    wl=df[(df["HTR"]=="H")&(df["FTR"]=="A")]; dh=df[(df["HTR"]=="D")&(df["FTR"]=="H")]
    da=df[(df["HTR"]=="D")&(df["FTR"]=="A")]; total=len(df)
    return {
        "remontada_loc":    len(rl), "pct_remontada_loc": round(len(rl)/total*100,2),
        "remontada_vis":    len(rv), "pct_remontada_vis": round(len(rv)/total*100,2),
        "ht_win_ft_lose":   len(wl), "pct_ht_win_ft_lose":round(len(wl)/total*100,2),
        "ht_draw_ft_win_h": len(dh), "ht_draw_ft_win_a":  len(da),
    }

def stats_comeback_teams(df: pd.DataFrame) -> pd.DataFrame:
    """Equipos que mas remontan y equipos que mas dejan escapar la victoria."""
    cb_h=df[(df["HTR"]=="A")&(df["FTR"]=="H")].groupby("HomeTeam").size().reset_index(name="cb_loc")
    cb_a=df[(df["HTR"]=="H")&(df["FTR"]=="A")].groupby("AwayTeam").size().reset_index(name="cb_vis")
    cb_a=cb_a.rename(columns={"AwayTeam":"HomeTeam"})
    ck_h=df[(df["HTR"]=="H")&(df["FTR"]=="A")].groupby("HomeTeam").size().reset_index(name="choke_loc")
    ck_a=df[(df["HTR"]=="A")&(df["FTR"]=="H")].groupby("AwayTeam").size().reset_index(name="choke_vis")
    ck_a=ck_a.rename(columns={"AwayTeam":"HomeTeam"})
    r=cb_h.merge(cb_a,on="HomeTeam",how="outer").merge(
       ck_h,on="HomeTeam",how="outer").merge(
       ck_a,on="HomeTeam",how="outer").fillna(0).rename(columns={"HomeTeam":"equipo"})
    for c in ["cb_loc","cb_vis","choke_loc","choke_vis"]:
        r[c]=r[c].astype(int)
    r["total_cb"]    = r["cb_loc"]    + r["cb_vis"]
    r["total_choke"] = r["choke_loc"] + r["choke_vis"]
    return r[["equipo","total_cb","cb_loc","cb_vis","total_choke","choke_loc","choke_vis"]].sort_values("total_cb",ascending=False)

# underdogs y smart money

def stats_underdogs(df) -> dict:
    ua=df[df["is_underdog_away"]==1]; uh=df[df["is_underdog_home"]==1]; ux=df[df["AvgA"]>8]
    return {
        "ud_away_total":  len(ua), "ud_away_wins": int(ua["away_win"].sum()),
        "ud_away_pct":    round(ua["away_win"].mean()*100,2) if len(ua) else 0,
        "ud_away_avgOdd": round(ua["AvgA"].mean(),3)         if len(ua) else 0,
        "ud_home_total":  len(uh), "ud_home_wins": int(uh["home_win"].sum()),
        "ud_home_pct":    round(uh["home_win"].mean()*100,2) if len(uh) else 0,
        "ud_home_avgOdd": round(uh["AvgH"].mean(),3)         if len(uh) else 0,
        "ud_ext_total":   len(ux), "ud_ext_wins":  int(ux["away_win"].sum()),
        "ud_ext_pct":     round(ux["away_win"].mean()*100,2) if len(ux) else 0,
    }

def stats_smart_money(df) -> dict:
    sh=df[df["odds_move_H"]<-0.1]; sa=df[df["odds_move_A"]<-0.1]; sd=df[df["odds_move_D"]<-0.1]
    return {
        "sm_loc_n":    len(sh), "sm_loc_pct":  round(sh["home_win"].mean()*100,2) if len(sh) else 0,
        "sm_loc_goles":round(sh["total_goals"].mean(),3) if len(sh) else 0,
        "sm_vis_n":    len(sa), "sm_vis_pct":  round(sa["away_win"].mean()*100,2) if len(sa) else 0,
        "sm_vis_goles":round(sa["total_goals"].mean(),3) if len(sa) else 0,
        "sm_draw_n":   len(sd), "sm_draw_pct": round(sd["draw"].mean()*100,2)     if len(sd) else 0,
    }

# dias de la semana y jornada

def stats_day_of_week(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby("day_name").agg(
        partidos    =("total_goals","count"),
        avg_goles   =("total_goals","mean"),
        var_goles   =("total_goals","var"),
        avg_ht      =("ht_goals","mean"),
        avg_st      =("second_half_goals","mean"),
        pct_btts    =("btts","mean"),
        pct_over25  =("over25","mean"),
        pct_H       =("home_win","mean"),
        pct_D       =("draw","mean"),
        pct_A       =("away_win","mean"),
    ).round(3)
    for c in ["pct_btts","pct_over25","pct_H","pct_D","pct_A"]:
        agg[c]=round(agg[c]*100,2)
    return agg.reindex([d for d in DAY_ORDER if d in agg.index]).reset_index()

def stats_jornada_size(df: pd.DataFrame) -> pd.DataFrame:
    ppf = df.groupby("Date").size().reset_index(name="n_dia")
    d   = df.merge(ppf,on="Date")
    d["tipo"] = d["n_dia"].map(lambda n:"1-unico" if n==1 else "2-3" if n<=3 else "4-6" if n<=6 else "7+")
    agg = d.groupby("tipo").agg(
        partidos  =("btts","count"), pct_btts=("btts","mean"), pct_over25=("over25","mean"),
        avg_goles =("total_goals","mean"), avg_ht=("ht_goals","mean"), avg_st=("second_half_goals","mean"),
    ).round(3).reset_index()
    agg["pct_btts"]  =round(agg["pct_btts"]*100,2)
    agg["pct_over25"]=round(agg["pct_over25"]*100,2)
    return agg

def stats_day_x_jornada(df: pd.DataFrame) -> pd.DataFrame:
    ppf=df.groupby("Date").size().reset_index(name="n_dia"); d=df.merge(ppf,on="Date")
    d["tipo"]=d["n_dia"].map(lambda n:"1-unico" if n==1 else "2-3" if n<=3 else "4-6" if n<=6 else "7+")
    agg=d.groupby(["day_name","tipo"]).agg(
        partidos=("btts","count"), pct_btts=("btts","mean"),
        avg_goles=("total_goals","mean"), avg_ht=("ht_goals","mean"),
    ).round(3).reset_index()
    agg["pct_btts"]=round(agg["pct_btts"]*100,2)
    return agg

def stats_month_season(df: pd.DataFrame) -> pd.DataFrame:
    """Goles por mes del año (agosto inicio de liga vs mayo/junio final con presion)."""
    agg=df.groupby("month").agg(
        partidos    =("total_goals","count"),
        avg_goles   =("total_goals","mean"),
        var_goles   =("total_goals","var"),
        avg_ht      =("ht_goals","mean"),
        avg_st      =("second_half_goals","mean"),
        pct_btts    =("btts","mean"), pct_over25=("over25","mean"),
        pct_H       =("home_win","mean"), pct_D=("draw","mean"), pct_A=("away_win","mean"),
        pct_high    =("high_scoring","mean"), pct_goalless=("goalless","mean"),
    ).round(3).reset_index()
    for c in ["pct_btts","pct_over25","pct_H","pct_D","pct_A","pct_high","pct_goalless"]:
        agg[c]=round(agg[c]*100,2)
    return agg

# rachas

def stats_streak_effect(df: pd.DataFrame) -> pd.DataFrame:
    """Efecto de la racha del equipo local: cuantas victorias/derrotas consecutivas
    llevaba ANTES del partido, y como influye en el resultado."""
    df2 = df.sort_values("Date").copy()
    streak_map = {}; home_streak = []
    for _, row in df2.iterrows():
        ht=row["HomeTeam"]; streak_map.setdefault(ht,0)
        home_streak.append(streak_map[ht])
        # actualizar local
        if row["FTR"]=="H":   streak_map[ht]=max(streak_map[ht],0)+1
        elif row["FTR"]=="A": streak_map[ht]=min(streak_map[ht],0)-1
        else:                 streak_map[ht]=0
        # actualizar visitante
        at=row["AwayTeam"]; streak_map.setdefault(at,0)
        if row["FTR"]=="A":   streak_map[at]=max(streak_map[at],0)+1
        elif row["FTR"]=="H": streak_map[at]=min(streak_map[at],0)-1
        else:                 streak_map[at]=0
    df2["home_streak"]=home_streak
    df2["streak_cat"]=df2["home_streak"].map(
        lambda s:"3+vic" if s>=3 else "2vic" if s==2 else "1vic" if s==1
                 else "neutro" if s==0 else "1der" if s==-1 else "2der" if s==-2 else "3+der"
    )
    order=["3+vic","2vic","1vic","neutro","1der","2der","3+der"]
    agg=df2.groupby("streak_cat").agg(
        partidos  =("home_win","count"),
        pct_H     =("home_win","mean"), pct_D=("draw","mean"), pct_A=("away_win","mean"),
        avg_goles =("total_goals","mean"), avg_ht=("ht_goals","mean"),
        pct_over25=("over25","mean"), pct_btts=("btts","mean"),
        avg_AvgH  =("AvgH","mean"),
    ).round(3)
    for c in ["pct_H","pct_D","pct_A","pct_over25","pct_btts"]:
        agg[c]=round(agg[c]*100,2)
    return agg.reindex([o for o in order if o in agg.index]).reset_index()

# tabla de posiciones y segmentos por rendimiento

def calc_standings(df_sub) -> pd.DataFrame:
    home=df_sub.groupby("HomeTeam").apply(lambda x: pd.Series({
        "pts_h":int((x["FTR"]=="H").sum()*3+(x["FTR"]=="D").sum()),
        "pj_h":len(x),"gf_h":int(x["FTHG"].sum()),"gc_h":int(x["FTAG"].sum()),
    })).reset_index().rename(columns={"HomeTeam":"equipo"})
    away=df_sub.groupby("AwayTeam").apply(lambda x: pd.Series({
        "pts_a":int((x["FTR"]=="A").sum()*3+(x["FTR"]=="D").sum()),
        "pj_a":len(x),"gf_a":int(x["FTAG"].sum()),"gc_a":int(x["FTHG"].sum()),
    })).reset_index().rename(columns={"AwayTeam":"equipo"})
    t=home.merge(away,on="equipo",how="outer").fillna(0)
    t["pts"]=t["pts_h"]+t["pts_a"]; t["pj"]=t["pj_h"]+t["pj_a"]
    t["gf"]=t["gf_h"]+t["gf_a"];   t["gc"]=t["gc_h"]+t["gc_a"]; t["dg"]=t["gf"]-t["gc"]
    return t[["equipo","pts","pj","gf","gc","dg"]].sort_values("pts",ascending=False).reset_index(drop=True)

def stats_segment(df, label) -> dict:
    return {
        "segmento":   label,   "partidos":  len(df),
        "avg_goles":  round(df["total_goals"].mean(),3),
        "moda_goles": int(df["total_goals"].mode()[0]) if not df["total_goals"].mode().empty else None,
        "std_goles":  round(df["total_goals"].std(),3),
        "avg_ht":     round(df["ht_goals"].mean(),3),
        "avg_st":     round(df["second_half_goals"].mean(),3),
        "pct_H":      round(df["home_win"].mean()*100,2),
        "pct_D":      round(df["draw"].mean()*100,2),
        "pct_A":      round(df["away_win"].mean()*100,2),
        "pct_over25": round(df["over25"].mean()*100,2),
        "pct_btts":   round(df["btts"].mean()*100,2),
        "avg_AvgH":   round(df["AvgH"].mean(),3),
        "avg_AvgA":   round(df["AvgA"].mean(),3),
    }

def stats_odds(df) -> dict:
    return {
        "avg_AvgH":round(df["AvgH"].mean(),3),   "avg_AvgD":round(df["AvgD"].mean(),3),
        "avg_AvgA":round(df["AvgA"].mean(),3),   "avg_AvgCH":round(df["AvgCH"].mean(),3),
        "avg_AvgCA":round(df["AvgCA"].mean(),3), "avg_overround":round(df["overround"].mean(),4),
        "med_AvgH":round(df["AvgH"].median(),3), "med_AvgA":round(df["AvgA"].median(),3),
    }

def top_scorers(df, col_team, col_goals, n=5) -> dict:
    t=df.groupby(col_team)[col_goals].sum().sort_values(ascending=False).head(n)
    return {k:int(v) for k,v in t.items()}

# helpers para graficas


# marcadores exactos mas frecuentes

def stats_exact_scores(df: pd.DataFrame, top_n=15) -> pd.DataFrame:
    """Top marcadores exactos mas frecuentes."""
    df2 = df.copy()
    df2["score"] = df2["FTHG"].astype(str)+"-"+df2["FTAG"].astype(str)
    cnt = df2["score"].value_counts().head(top_n).reset_index()
    cnt.columns = ["marcador","freq"]
    cnt["pct"]  = round(cnt["freq"]/len(df)*100,2)
    cnt["acum"] = round(cnt["pct"].cumsum(),2)
    return cnt

def stats_exact_scores_ht(df: pd.DataFrame, top_n=10) -> pd.DataFrame:
    """Top marcadores exactos al descanso."""
    df2 = df.copy()
    df2["ht_score"] = df2["HTHG"].astype(str)+"-"+df2["HTAG"].astype(str)
    cnt = df2["ht_score"].value_counts().head(top_n).reset_index()
    cnt.columns = ["marcador_ht","freq"]
    cnt["pct"] = round(cnt["freq"]/len(df)*100,2)
    return cnt

def stats_ht_to_ft_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Matriz de transicion HTR -> FTR con porcentaje por fila."""
    rows = []
    for htr in ["H","D","A"]:
        sub = df[df["HTR"]==htr]; total = len(sub)
        if total==0: continue
        rows.append({
            "HTR":htr, "total":total,
            "->H":int((sub["FTR"]=="H").sum()), "pct_H":round((sub["FTR"]=="H").mean()*100,2),
            "->D":int((sub["FTR"]=="D").sum()), "pct_D":round((sub["FTR"]=="D").mean()*100,2),
            "->A":int((sub["FTR"]=="A").sum()), "pct_A":round((sub["FTR"]=="A").mean()*100,2),
        })
    return pd.DataFrame(rows)

def stats_ht_consistency(df: pd.DataFrame) -> dict:
    """Que tan bien predice el resultado HT al resultado FT."""
    total = len(df); same = (df["HTR"]==df["FTR"]).sum()
    fH=df[df["HTR"]=="H"]; fD=df[df["HTR"]=="D"]; fA=df[df["HTR"]=="A"]
    return {
        "total":total, "ht_igual_ft":int(same),
        "pct_mismo":round(same/total*100,2),
        "si_htH_pct_ftH":round((fH["FTR"]=="H").mean()*100,2) if len(fH) else 0,
        "si_htD_pct_ftD":round((fD["FTR"]=="D").mean()*100,2) if len(fD) else 0,
        "si_htA_pct_ftA":round((fA["FTR"]=="A").mean()*100,2) if len(fA) else 0,
        "si_htH_remonta_A":round((fH["FTR"]=="A").mean()*100,2) if len(fH) else 0,
        "si_htA_remonta_H":round((fA["FTR"]=="H").mean()*100,2) if len(fA) else 0,
    }

def stats_scoreline_1_0_ht(df: pd.DataFrame) -> dict:
    """Distribucion FT cuando se va 1-0 al descanso."""
    sub = df[(df["HTHG"]==1)&(df["HTAG"]==0)]
    if len(sub)==0: return {}
    df2=sub.copy(); df2["score"]=df2["FTHG"].astype(str)+"-"+df2["FTAG"].astype(str)
    top = df2["score"].value_counts().head(8)
    return {
        "total":int(len(sub)), "pct_del_total":round(len(sub)/len(df)*100,2),
        "pct_mantiene_H":round((sub["FTR"]=="H").mean()*100,2),
        "pct_empata":    round((sub["FTR"]=="D").mean()*100,2),
        "pct_remonta_A": round((sub["FTR"]=="A").mean()*100,2),
        "avg_goles_ft":  round(sub["total_goals"].mean(),3),
        "top_marcadores":dict(zip(top.index.tolist(),top.values.tolist())),
    }

def stats_goal_diff_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Distribucion de la diferencia de goles en el resultado final."""
    df2=df.copy(); df2["abs_diff"]=df2["goal_diff"].abs()
    cnt=df2["abs_diff"].value_counts().sort_index().reset_index()
    cnt.columns=["diff_goles","freq"]
    cnt["pct"] =round(cnt["freq"]/len(df)*100,2)
    cnt["acum"]=round(cnt["pct"].cumsum(),2)
    return cnt

def stats_home_away_scoring(df: pd.DataFrame) -> dict:
    """Quién marca: local solo, visitante solo, ambos o nadie."""
    return {
        "pct_local_no_marca": round((df["FTHG"]==0).mean()*100,2),
        "pct_visit_no_marca": round((df["FTAG"]==0).mean()*100,2),
        "pct_ambos_marcan":   round(df["btts"].mean()*100,2),
        "pct_solo_local":     round(((df["FTHG"]>0)&(df["FTAG"]==0)).mean()*100,2),
        "pct_solo_visit":     round(((df["FTAG"]>0)&(df["FTHG"]==0)).mean()*100,2),
        "pct_ninguno":        round(df["goalless"].mean()*100,2),
        "avg_goles_si_local_marca":round(df[df["FTHG"]>0]["total_goals"].mean(),3),
        "avg_goles_si_visit_marca":round(df[df["FTAG"]>0]["total_goals"].mean(),3),
    }

def stats_st_vs_ht_rhythm(df: pd.DataFrame) -> dict:
    """Partidos donde el segundo tiempo fue más goleador que el primero."""
    mas  = df[df["second_half_goals"]>df["ht_goals"]]
    igual= df[df["second_half_goals"]==df["ht_goals"]]
    menos= df[df["second_half_goals"]<df["ht_goals"]]
    total= len(df)
    return {
        "st_mayor_ht":len(mas),   "pct_st_mayor":round(len(mas)/total*100,2),
        "st_igual_ht":len(igual), "pct_st_igual":round(len(igual)/total*100,2),
        "st_menor_ht":len(menos), "pct_st_menor":round(len(menos)/total*100,2),
        "avg_goles_st_mayor":round(mas["total_goals"].mean(),3)  if len(mas)  else 0,
        "avg_goles_st_menor":round(menos["total_goals"].mean(),3) if len(menos) else 0,
    }

def stats_goles_by_fav_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Goles y resultados segun perfil de cuota local (gran fav → underdog)."""
    rows=[]
    for lbl, mask in [
        ("gran_fav <1.5",     df["AvgH"]<1.5),
        ("fav 1.5-2.0",       (df["AvgH"]>=1.5)&(df["AvgH"]<2.0)),
        ("abierto 2.0-2.5",   (df["AvgH"]>=2.0)&(df["AvgH"]<2.5)),
        ("underdog >2.5",     df["AvgH"]>=2.5),
    ]:
        sub=df[mask]
        if len(sub)<5: continue
        rows.append({
            "perfil":lbl, "partidos":len(sub),
            "avg_goles_loc":round(sub["FTHG"].mean(),3),
            "avg_goles_vis":round(sub["FTAG"].mean(),3),
            "avg_total":    round(sub["total_goals"].mean(),3),
            "moda_total":   int(sub["total_goals"].mode()[0]) if not sub["total_goals"].mode().empty else None,
            "pct_H":round(sub["home_win"].mean()*100,2),
            "pct_D":round(sub["draw"].mean()*100,2),
            "pct_A":round(sub["away_win"].mean()*100,2),
            "pct_btts":  round(sub["btts"].mean()*100,2),
            "pct_over25":round(sub["over25"].mean()*100,2),
            "pct_cs_h":  round(sub["clean_sheet_h"].mean()*100,2),
        })
    return pd.DataFrame(rows)

# analisis extra de cuotas y comportamiento del mercado

def stats_overround_vs_goals(df: pd.DataFrame) -> pd.DataFrame:
    """Overround bajo / medio / alto: correlacion con goles y resultados."""
    df2=df.copy(); q33=df2["overround"].quantile(0.33); q66=df2["overround"].quantile(0.66)
    rows=[]
    for lbl, mask in [
        ("bajo <p33",    df2["overround"]<q33),
        ("medio p33-66",(df2["overround"]>=q33)&(df2["overround"]<q66)),
        ("alto >p66",   df2["overround"]>=q66),
    ]:
        sub=df2[mask]
        rows.append({
            "overround_nivel":lbl, "partidos":len(sub),
            "avg_overround":round(sub["overround"].mean(),4),
            "avg_goles":    round(sub["total_goals"].mean(),3),
            "pct_btts":     round(sub["btts"].mean()*100,2),
            "pct_over25":   round(sub["over25"].mean()*100,2),
            "pct_H":round(sub["home_win"].mean()*100,2),
            "pct_D":round(sub["draw"].mean()*100,2),
            "pct_A":round(sub["away_win"].mean()*100,2),
        })
    return pd.DataFrame(rows)

def stats_market_volatility(df: pd.DataFrame) -> dict:
    """Partidos con alta vs baja volatilidad de mercado (movimiento total de cuota)."""
    df2=df.copy()
    df2["total_move"]=df2["odds_move_H"].abs()+df2["odds_move_A"].abs()+df2["odds_move_D"].abs()
    umb=df2["total_move"].quantile(0.80)
    alta=df2[df2["total_move"]>=umb]; baja=df2[df2["total_move"]<umb]
    return {
        "umbral_p80":       round(float(umb),4),
        "alta_vol_n":       len(alta),
        "alta_pct_H":       round(alta["home_win"].mean()*100,2) if len(alta) else 0,
        "alta_avg_goles":   round(alta["total_goals"].mean(),3)  if len(alta) else 0,
        "alta_pct_btts":    round(alta["btts"].mean()*100,2)     if len(alta) else 0,
        "baja_vol_n":       len(baja),
        "baja_pct_H":       round(baja["home_win"].mean()*100,2) if len(baja) else 0,
        "baja_avg_goles":   round(baja["total_goals"].mean(),3)  if len(baja) else 0,
        "baja_pct_btts":    round(baja["btts"].mean()*100,2)     if len(baja) else 0,
        "corr_move_goles":  round(df2["total_move"].corr(df2["total_goals"]),4),
        "corr_move_homewin":round(df2["total_move"].corr(df2["home_win"]),4),
    }

def stats_apertura_vs_cierre(df: pd.DataFrame) -> pd.DataFrame:
    """Apertura vs cierre: cuanto se mueve el mercado y en que direccion."""
    rows=[]
    for resultado, odd_ap, odd_ci in [("H","AvgH","AvgCH"),("A","AvgA","AvgCA"),("D","AvgD","AvgCD")]:
        rows.append({
            "resultado":resultado,
            "avg_apertura":round(df[odd_ap].mean(),3), "avg_cierre":round(df[odd_ci].mean(),3),
            "diff_media":  round((df[odd_ci]-df[odd_ap]).mean(),4),
            "diff_abs":    round((df[odd_ci]-df[odd_ap]).abs().mean(),4),
            "pct_sube":    round((df[odd_ci]>df[odd_ap]).mean()*100,2),
            "pct_baja":    round((df[odd_ci]<df[odd_ap]).mean()*100,2),
            "pct_igual":   round((df[odd_ci]==df[odd_ap]).mean()*100,2),
            "max_subida":  round((df[odd_ci]-df[odd_ap]).max(),3),
            "max_bajada":  round((df[odd_ci]-df[odd_ap]).min(),3),
        })
    return pd.DataFrame(rows)

def stats_superfav_analysis(df: pd.DataFrame) -> dict:
    """Partidos con local gran favorito (imp_prob_H > 0.70)."""
    sf=df[df["imp_prob_H"]>0.70]; rest=df[df["imp_prob_H"]<=0.70]
    return {
        "superfav_n":        len(sf),
        "pct_del_total":     round(len(sf)/len(df)*100,2),
        "pct_H":             round(sf["home_win"].mean()*100,2)       if len(sf) else 0,
        "pct_D":             round(sf["draw"].mean()*100,2)           if len(sf) else 0,
        "pct_A":             round(sf["away_win"].mean()*100,2)       if len(sf) else 0,
        "avg_goles":         round(sf["total_goals"].mean(),3)        if len(sf) else 0,
        "pct_cs_h":          round(sf["clean_sheet_h"].mean()*100,2)  if len(sf) else 0,
        "pct_btts":          round(sf["btts"].mean()*100,2)           if len(sf) else 0,
        "avg_AvgH":          round(sf["AvgH"].mean(),3)               if len(sf) else 0,
        "pct_H_resto":       round(rest["home_win"].mean()*100,2)     if len(rest) else 0,
    }

def stats_draw_by_odd_range(df: pd.DataFrame) -> pd.DataFrame:
    """% empate real vs implied segun rango de cuota del empate."""
    bins  =[2.5,2.8,3.0,3.2,3.5,3.8,4.2,5.0,15.0]
    labels=[f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]
    d2=df.copy(); d2["rng_D"]=pd.cut(d2["AvgD"],bins=bins,labels=labels,right=False)
    rows=[]
    for rng in labels:
        sub=d2[d2["rng_D"]==rng]
        if len(sub)<5: continue
        rows.append({
            "rango_AvgD":rng, "partidos":len(sub),
            "imp_%":  round(1/sub["AvgD"].mean()*100,2),
            "real_%": round(sub["draw"].mean()*100,2),
            "diff":   round(sub["draw"].mean()*100 - 1/sub["AvgD"].mean()*100,2),
            "avg_goles":round(sub["total_goals"].mean(),3),
        })
    return pd.DataFrame(rows)

def stats_upset_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Con que frecuencia gana el menos favorito segun diferencia de cuotas."""
    df2=df.copy()
    df2["fav_gana"]=(((df2["AvgH"]<df2["AvgA"])&(df2["FTR"]=="H"))|((df2["AvgA"]<df2["AvgH"])&(df2["FTR"]=="A"))).astype(int)
    df2["upset"]   =(((df2["AvgH"]>df2["AvgA"])&(df2["FTR"]=="H"))|((df2["AvgA"]>df2["AvgH"])&(df2["FTR"]=="A"))).astype(int)
    bins  =[0,0.3,0.75,1.5,3.0,5.0,100]
    labels=["0-0.3","0.3-0.75","0.75-1.5","1.5-3.0","3.0-5.0","5.0+"]
    df2["gap"]    =(df2["AvgH"]-df2["AvgA"]).abs()
    df2["gap_cat"]=pd.cut(df2["gap"],bins=bins,labels=labels,right=False)
    rows=[]
    for cat in labels:
        sub=df2[df2["gap_cat"]==cat]
        if len(sub)<5: continue
        rows.append({
            "dif_cuotas":cat, "partidos":len(sub),
            "pct_fav_gana":round(sub["fav_gana"].mean()*100,2),
            "pct_upset":   round(sub["upset"].mean()*100,2),
            "pct_empate":  round(sub["draw"].mean()*100,2),
        })
    return pd.DataFrame(rows)

def stats_surprise_index(df: pd.DataFrame, top_n=20) -> pd.DataFrame:
    """Top partidos mas sorpresivos: el ganador tenia probabilidad implicita muy baja."""
    df2=df.copy()
    df2["prob_ganador"]=df2.apply(
        lambda r: r["imp_prob_H"] if r["FTR"]=="H" else r["imp_prob_A"] if r["FTR"]=="A" else r["imp_prob_D"],axis=1)
    df2["sorpresa"]=round(1-df2["prob_ganador"],4)
    return df2.nlargest(top_n,"sorpresa")[
        ["Date","Div","HomeTeam","AwayTeam","FTHG","FTAG","FTR","AvgH","AvgA","prob_ganador","sorpresa"]
    ].reset_index(drop=True)

def stats_market_consensus(df: pd.DataFrame):
    """Discrepancia entre B365 y el promedio de mercado (Avg): donde hay menos consenso."""
    df2=df.copy()
    df2["disc_H"]=(df2["B365H"]-df2["AvgH"]).abs()
    df2["disc_A"]=(df2["B365A"]-df2["AvgA"]).abs()
    df2["disc_D"]=(df2["B365D"]-df2["AvgD"]).abs()
    df2["disc_total"]=df2["disc_H"]+df2["disc_A"]+df2["disc_D"]
    rows=[]
    for col,lbl in [("disc_H","H"),("disc_A","A"),("disc_D","D"),("disc_total","total")]:
        rows.append({
            "resultado":lbl,
            "avg_disc":round(df2[col].mean(),4), "max_disc":round(df2[col].max(),4),
            "pct_disc>0.1":round((df2[col]>0.1).mean()*100,2),
            "pct_disc>0.3":round((df2[col]>0.3).mean()*100,2),
        })
    top=df2.nlargest(10,"disc_total")[["Date","Div","HomeTeam","AwayTeam","B365H","AvgH","B365A","AvgA","disc_total"]].reset_index(drop=True)
    return pd.DataFrame(rows), top

def stats_entropy_results(df: pd.DataFrame) -> pd.DataFrame:
    """Entropia de Shannon de resultados por liga: indica nivel de impredecibilidad."""
    rows=[]
    for div in sorted(df["Div"].unique()):
        sub=df[df["Div"]==div]; p=sub["FTR"].value_counts(normalize=True)
        ent=-sum(p*np.log2(p+1e-10))
        rows.append({
            "liga":div, "partidos":len(sub),
            "pct_H":round((sub["FTR"]=="H").mean()*100,2),
            "pct_D":round((sub["FTR"]=="D").mean()*100,2),
            "pct_A":round((sub["FTR"]=="A").mean()*100,2),
            "entropia":round(ent,4), "max_posible":round(np.log2(3),4),
        })
    return pd.DataFrame(rows)

def stats_gini_goals(df: pd.DataFrame) -> dict:
    """Coeficiente de Gini de goles por equipo: concentracion ofensiva."""
    hg=df.groupby("HomeTeam")["FTHG"].sum(); ag=df.groupby("AwayTeam")["FTAG"].sum()
    total=hg.add(ag,fill_value=0).sort_values()
    n=len(total); vals=total.values
    if n==0: return {}
    idx=np.arange(1,n+1)
    gini=round(float((2*idx-n-1).dot(vals)/(n*vals.sum())),4)
    return {
        "gini_goles":   gini,
        "max_equipo":   total.index[-1], "max_goles":int(vals[-1]),
        "min_equipo":   total.index[0],  "min_goles":int(vals[0]),
        "pct_top3":     round(vals[-3:].sum()/vals.sum()*100,2) if n>=3 else None,
        "pct_bot3":     round(vals[:3].sum()/vals.sum()*100,2)  if n>=3 else None,
    }

def stats_overround_evolution(df: pd.DataFrame) -> pd.DataFrame:
    """Evolucion del overround por temporada: crece o decrece el margen de la casa."""
    rows=[]
    for slbl in df["Season_label"].dropna().unique():
        sub=df[df["Season_label"]==slbl]
        rows.append({
            "temporada":slbl, "partidos":len(sub),
            "avg_overround":round(sub["overround"].mean(),4),
            "std_overround":round(sub["overround"].std(),4),
            "min_overround":round(sub["overround"].min(),4),
            "max_overround":round(sub["overround"].max(),4),
            "pct_sobre_105":round((sub["overround"]>1.05).mean()*100,2),
        })
    return pd.DataFrame(rows).sort_values("temporada")

def stats_home_advantage_decay(df: pd.DataFrame) -> pd.DataFrame:
    """Decaimiento de ventaja local por temporada."""
    agg=df.groupby("Season_label").agg(
        partidos=("home_win","count"),
        pct_H=("home_win","mean"), pct_D=("draw","mean"), pct_A=("away_win","mean"),
        avg_AvgH=("AvgH","mean"),
    ).round(3).reset_index()
    for c in ["pct_H","pct_D","pct_A"]:
        agg[c]=round(agg[c]*100,2)
    return agg

def stats_consecutive_streaks(df: pd.DataFrame, n_min=3) -> pd.DataFrame:
    """Rachas ganadoras maximas por equipo en todo el periodo."""
    df2=df.sort_values("Date").copy()
    best={}; current={}
    for _,row in df2.iterrows():
        for team,res,side in [(row["HomeTeam"],row["FTR"],"H"),(row["AwayTeam"],row["FTR"],"A")]:
            current.setdefault(team,0); best.setdefault(team,0)
            if res==side:
                current[team]+=1
                if current[team]>best[team]: best[team]=current[team]
            else:
                current[team]=0
    df_out=pd.DataFrame(list(best.items()),columns=["equipo","max_racha_vic"])
    return df_out[df_out["max_racha_vic"]>=n_min].sort_values("max_racha_vic",ascending=False).head(20)

def stats_clean_sheet_by_month(df: pd.DataFrame) -> pd.DataFrame:
    """Clean sheets por mes: mas defensas hermeticas en invierno."""
    agg=df.groupby("month").agg(
        partidos    =("clean_sheet_h","count"),
        pct_cs_h    =("clean_sheet_h","mean"),
        pct_cs_a    =("clean_sheet_a","mean"),
        pct_goalless=("goalless","mean"),
        avg_goles   =("total_goals","mean"),
    ).round(3).reset_index()
    for c in ["pct_cs_h","pct_cs_a","pct_goalless"]:
        agg[c]=round(agg[c]*100,2)
    return agg

def stats_btts_smart_money(df: pd.DataFrame) -> dict:
    """BTTS y goles segun hacia donde apunta el smart money."""
    sm_d=df[df["odds_move_D"]<-0.1]; no_sm=df[df["odds_move_D"]>=-0.1]
    sm_h=df[df["odds_move_H"]<-0.1]; sm_a=df[df["odds_move_A"]<-0.1]
    return {
        "sm_draw_btts":     round(sm_d["btts"].mean()*100,2)    if len(sm_d)  else 0,
        "no_sm_draw_btts":  round(no_sm["btts"].mean()*100,2)   if len(no_sm) else 0,
        "sm_draw_goles":    round(sm_d["total_goals"].mean(),3)  if len(sm_d)  else 0,
        "sm_H_btts":        round(sm_h["btts"].mean()*100,2)    if len(sm_h)  else 0,
        "sm_H_goles":       round(sm_h["total_goals"].mean(),3)  if len(sm_h)  else 0,
        "sm_A_btts":        round(sm_a["btts"].mean()*100,2)    if len(sm_a)  else 0,
        "sm_A_goles":       round(sm_a["total_goals"].mean(),3)  if len(sm_a)  else 0,
        "corr_move_D_btts": round(df["odds_move_D"].corr(df["btts"]),4),
        "corr_move_D_goles":round(df["odds_move_D"].corr(df["total_goals"]),4),
    }

def stats_home_local_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Ratio goles local/visitante por liga y temporada."""
    agg=df.groupby(["Div","Season_label"]).agg(
        goles_loc=("FTHG","sum"), goles_vis=("FTAG","sum"), partidos=("FTHG","count"),
    ).reset_index()
    agg["ratio"]=round(agg["goles_loc"]/agg["goles_vis"],3)
    agg["avg_loc"]=round(agg["goles_loc"]/agg["partidos"],3)
    agg["avg_vis"]=round(agg["goles_vis"]/agg["partidos"],3)
    return agg

def stats_value_betting(df: pd.DataFrame) -> pd.DataFrame:
    """EV y edge por rango de cuota local: donde hay mas valor real."""
    bins  =[1.0,1.5,2.0,2.5,3.0,4.0,6.0,25.0]
    labels=[f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]
    d2=df.copy(); d2["rng"]=pd.cut(d2["AvgH"],bins=bins,labels=labels,right=False)
    rows=[]
    for rng in labels:
        sub=d2[d2["rng"]==rng]
        if len(sub)<5: continue
        imp=1/sub["AvgH"].mean(); real=sub["home_win"].mean()
        rows.append({
            "rango_odd":rng, "n":len(sub),
            "imp_%": round(imp*100,2),  "real_%":round(real*100,2),
            "edge":  round((real-imp)*100,2),
            "ev_1u": round(sub["AvgH"].mean()*real-1,4),
        })
    return pd.DataFrame(rows)

# rachas detalladas por equipo

def stats_max_streaks_all(df: pd.DataFrame) -> pd.DataFrame:
    """Racha maxima de victorias, derrotas y empates por equipo (todo el periodo)."""
    df2 = df.sort_values("Date").copy()
    best_v = {}; best_d = {}; best_e = {}
    cur_v  = {}; cur_d  = {}; cur_e  = {}
    for _, row in df2.iterrows():
        for team, side in [(row["HomeTeam"],"H"),(row["AwayTeam"],"A")]:
            for d in [best_v,best_d,best_e,cur_v,cur_d,cur_e]:
                d.setdefault(team,0)
            won  = row["FTR"]==side
            lost = (row["FTR"]=="H" and side=="A") or (row["FTR"]=="A" and side=="H")
            draw = row["FTR"]=="D"
            if won:
                cur_v[team]+=1; cur_d[team]=0; cur_e[team]=0
            elif lost:
                cur_d[team]+=1; cur_v[team]=0; cur_e[team]=0
            else:
                cur_e[team]+=1; cur_v[team]=0; cur_d[team]=0
            best_v[team]=max(best_v[team],cur_v[team])
            best_d[team]=max(best_d[team],cur_d[team])
            best_e[team]=max(best_e[team],cur_e[team])
    equipos = sorted(best_v.keys())
    out = pd.DataFrame({
        "equipo":       equipos,
        "max_racha_vic":[best_v[e] for e in equipos],
        "max_racha_der":[best_d[e] for e in equipos],
        "max_racha_emp":[best_e[e] for e in equipos],
    })
    return out.sort_values("max_racha_vic",ascending=False).reset_index(drop=True)


def stats_bad_streak_then_goleada(df: pd.DataFrame, min_bad=3, min_goals=4) -> pd.DataFrame:
    """Fenomeno: equipo en mala racha golea un partido y al siguiente pierde.
    Detecta la secuencia: >=min_bad derrotas/empates -> partido con >=min_goals goles a favor -> siguiente resultado.
    """
    df2 = df.sort_values("Date").copy()
    resultados = {}; racha_mala = {}
    events = []
    for _, row in df2.iterrows():
        for team, side, gf, gc in [
            (row["HomeTeam"],"H",row["FTHG"],row["FTAG"]),
            (row["AwayTeam"],"A",row["FTAG"],row["FTHG"])
        ]:
            resultados.setdefault(team,[])
            racha_mala.setdefault(team,0)
            prev_racha = racha_mala[team]
            won = row["FTR"]==side
            if won:
                if prev_racha>=min_bad and gf>=min_goals:
                    events.append({
                        "equipo":team,"fecha_goleada":row["Date"],
                        "rival_goleada":row["AwayTeam"] if side=="H" else row["HomeTeam"],
                        "marcador_goleada":f"{gf}-{gc}","racha_mala_previa":prev_racha,
                        "_team":team,"_side":side,"_fecha":row["Date"]
                    })
                racha_mala[team]=0
            else:
                racha_mala[team]+=1
    if not events:
        return pd.DataFrame()
    # buscar partido siguiente despues de cada goleada
    out_rows=[]
    for ev in events:
        team=ev["equipo"]; fd=ev["fecha_goleada"]
        next_games=df2[
            ((df2["HomeTeam"]==team)|(df2["AwayTeam"]==team)) & (df2["Date"]>fd)
        ].sort_values("Date")
        if next_games.empty:
            continue
        ng=next_games.iloc[0]
        side_next="H" if ng["HomeTeam"]==team else "A"
        result_next=ng["FTR"]
        won_next=result_next==side_next
        lost_next=(result_next=="H" and side_next=="A") or (result_next=="A" and side_next=="H")
        out_rows.append({
            "equipo":team,
            "racha_mala_previa":ev["racha_mala_previa"],
            "fecha_goleada":ev["fecha_goleada"],
            "rival_goleada":ev["rival_goleada"],
            "marcador_goleada":ev["marcador_goleada"],
            "fecha_siguiente":ng["Date"],
            "rival_siguiente":ng["AwayTeam"] if side_next=="H" else ng["HomeTeam"],
            "resultado_siguiente":result_next,
            "gano_siguiente":int(won_next),
            "perdio_siguiente":int(lost_next),
            "empato_siguiente":int(result_next=="D"),
        })
    return pd.DataFrame(out_rows).sort_values(["equipo","fecha_goleada"]).reset_index(drop=True)


# rachas extendidas: version ampliada con mas categorias y contexto

def _build_streak_sequence(df: pd.DataFrame) -> pd.DataFrame:
    """Núcleo interno: para cada partido, calcula la racha previa de cada equipo
    (victorias positivas, derrotas negativas, empate rompe racha a 0),
    goles a favor/contra en el partido anterior, y si el anterior fue una goleada.
    Devuelve df enriquecido con columnas prev_* para local y visitante."""
    df2 = df.sort_values("Date").copy()

    streak  = {}   # racha neta (vic>0, der<0)
    prev_gf = {}   # goles marcados en el ultimo partido
    prev_gc = {}   # goles recibidos en el ultimo partido
    prev_res= {}   # resultado del ultimo partido ('W','L','D')

    rows_h = []  # (partido_idx, racha_previa_local, prev_gf_h, prev_gc_h, prev_res_h)
    rows_a = []  # idem visitante

    for idx, row in df2.iterrows():
        ht = row["HomeTeam"]; at = row["AwayTeam"]
        for d in [streak, prev_gf, prev_gc, prev_res]:
            d.setdefault(ht, 0); d.setdefault(at, 0)

        rows_h.append({
            "idx":           idx,
            "prev_streak_h": streak[ht],
            "prev_gf_h":     prev_gf[ht],
            "prev_gc_h":     prev_gc[ht],
            "prev_res_h":    prev_res[ht],
        })
        rows_a.append({
            "idx":           idx,
            "prev_streak_a": streak[at],
            "prev_gf_a":     prev_gf[at],
            "prev_gc_a":     prev_gc[at],
            "prev_res_a":    prev_res[at],
        })

        # actualizar local
        if row["FTR"] == "H":
            streak[ht]  = max(streak[ht], 0) + 1
            prev_res[ht]= "W"
        elif row["FTR"] == "A":
            streak[ht]  = min(streak[ht], 0) - 1
            prev_res[ht]= "L"
        else:
            streak[ht]  = 0
            prev_res[ht]= "D"
        prev_gf[ht] = int(row["FTHG"]); prev_gc[ht] = int(row["FTAG"])

        # actualizar visitante
        if row["FTR"] == "A":
            streak[at]  = max(streak[at], 0) + 1
            prev_res[at]= "W"
        elif row["FTR"] == "H":
            streak[at]  = min(streak[at], 0) - 1
            prev_res[at]= "L"
        else:
            streak[at]  = 0
            prev_res[at]= "D"
        prev_gf[at] = int(row["FTAG"]); prev_gc[at] = int(row["FTHG"])

    ph = pd.DataFrame(rows_h).set_index("idx")
    pa = pd.DataFrame(rows_a).set_index("idx")
    df2 = df2.join(ph).join(pa)
    return df2


def stats_bad_streak_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Análisis estadístico del fenómeno 'racha mala → qué pasa'.
    Agrupa partidos por la racha previa del equipo local (o visitante)
    y muestra: avg goles a favor/contra, % victoria, % goleada (4+ GF), % clean sheet.
    Responde: ¿los equipos en mala racha atacan más? ¿defienden peor?"""
    df2 = _build_streak_sequence(df)

    # Análisis desde perspectiva del local
    cats = [
        ("3+der", df2["prev_streak_h"] <= -3),
        ("2der",  df2["prev_streak_h"] == -2),
        ("1der",  df2["prev_streak_h"] == -1),
        ("neutro",df2["prev_streak_h"] == 0),
        ("1vic",  df2["prev_streak_h"] == 1),
        ("2vic",  df2["prev_streak_h"] == 2),
        ("3+vic", df2["prev_streak_h"] >= 3),
    ]
    rows = []
    for lbl, mask in cats:
        sub = df2[mask]
        if len(sub) < 5: continue
        rows.append({
            "racha_previa_local": lbl,
            "partidos":           len(sub),
            "avg_gf_local":       round(sub["FTHG"].mean(), 3),
            "avg_gc_local":       round(sub["FTAG"].mean(), 3),
            "avg_goles_total":    round(sub["total_goals"].mean(), 3),
            "pct_vic_local":      round(sub["home_win"].mean() * 100, 2),
            "pct_der_local":      round(sub["away_win"].mean() * 100, 2),
            "pct_empate":         round(sub["draw"].mean() * 100, 2),
            "pct_goleada_local":  round((sub["FTHG"] >= 4).mean() * 100, 2),   # golea 4+
            "pct_cs_local":       round(sub["clean_sheet_h"].mean() * 100, 2), # no recibe
            "pct_btts":           round(sub["btts"].mean() * 100, 2),
            "pct_over25":         round(sub["over25"].mean() * 100, 2),
            "avg_AvgH":           round(sub["AvgH"].mean(), 3),  # ¿ajusta el mercado?
        })
    return pd.DataFrame(rows)


def stats_streak_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """'Momentum': ¿qué pasa en el partido inmediatamente después de
    una victoria goleadora (4+ goles) vs una derrota goleadora (recibe 4+)?
    Categorías del partido previo:
      - goleada_a_favor:  marcó 4+ y ganó
      - goleada_recibida: recibió 4+ y perdió
      - victoria_normal:  ganó marcando 1-3
      - derrota_normal:   perdió recibiendo 1-3
      - empate:           empató
    """
    df2 = _build_streak_sequence(df)

    def cat_prev(res, gf, gc):
        if res == "W":
            return "goleada_favor" if gf >= 4 else "victoria_normal"
        elif res == "L":
            return "goleada_recibida" if gc >= 4 else "derrota_normal"
        return "empate"

    # Perspectiva local
    df2["cat_prev_h"] = df2.apply(
        lambda r: cat_prev(r["prev_res_h"], r["prev_gf_h"], r["prev_gc_h"]), axis=1)

    rows = []
    order = ["goleada_favor","victoria_normal","empate","derrota_normal","goleada_recibida"]
    for cat in order:
        sub = df2[df2["cat_prev_h"] == cat]
        if len(sub) < 5: continue
        rows.append({
            "partido_anterior_local": cat,
            "partidos":         len(sub),
            "pct_vic_local":    round(sub["home_win"].mean() * 100, 2),
            "pct_empate":       round(sub["draw"].mean() * 100, 2),
            "pct_der_local":    round(sub["away_win"].mean() * 100, 2),
            "avg_gf_hoy":       round(sub["FTHG"].mean(), 3),
            "avg_gc_hoy":       round(sub["FTAG"].mean(), 3),
            "pct_btts":         round(sub["btts"].mean() * 100, 2),
            "pct_over25":       round(sub["over25"].mean() * 100, 2),
            "pct_cs_local":     round(sub["clean_sheet_h"].mean() * 100, 2),
            "avg_AvgH":         round(sub["AvgH"].mean(), 3),
        })
    return pd.DataFrame(rows)


def stats_bad_streak_explosion_deep(df: pd.DataFrame,
                                     min_bad: int = 2,
                                     min_goles_goleada: int = 3) -> dict:
    """Versión profunda del fenómeno que observaste:
    'racha mala → goleada → siguiente partido'.
    Calcula tasas agregadas y también las separa por si la goleada
    fue como LOCAL o como VISITANTE, y por intensidad de la racha.

    Retorna:
      - resumen global
      - tabla por intensidad de racha (2, 3, 4+ partidos malos)
      - tabla: resultado del partido SIGUIENTE a la goleada
      - tasa de pérdida en el siguiente vs la media normal del equipo
    """
    df2 = _build_streak_sequence(df)

    # Partidos donde el equipo (local o visitante) venía de racha mala
    # y en ESE partido golea (marca min_goles_goleada+)
    events = []

    for _, row in df2.iterrows():
        for side, streak_col, gf_col, gc_col, team_col in [
            ("H", "prev_streak_h", "FTHG", "FTAG", "HomeTeam"),
            ("A", "prev_streak_a", "FTAG", "FTHG", "AwayTeam"),
        ]:
            streak_prev = row[streak_col]
            gf = row[gf_col]
            gc = row[gc_col]
            won = row["FTR"] == side

            if streak_prev <= -min_bad and won and gf >= min_goles_goleada:
                events.append({
                    "equipo":        row[team_col],
                    "liga":          row["Div"],
                    "fecha":         row["Date"],
                    "lado":          side,
                    "racha_previa":  streak_prev,
                    "gf_goleada":    gf,
                    "gc_goleada":    gc,
                    "rival":         row["AwayTeam"] if side=="H" else row["HomeTeam"],
                })

    if not events:
        return {"sin_eventos": True}

    ev_df = pd.DataFrame(events)

    # Buscar partido siguiente para cada evento
    df_sorted = df2.sort_values("Date")
    next_results = []
    for _, ev in ev_df.iterrows():
        team = ev["equipo"]; fecha = ev["fecha"]
        next_g = df_sorted[
            ((df_sorted["HomeTeam"] == team) | (df_sorted["AwayTeam"] == team)) &
            (df_sorted["Date"] > fecha)
        ].head(1)
        if next_g.empty: continue
        ng = next_g.iloc[0]
        side_next = "H" if ng["HomeTeam"] == team else "A"
        ftr = ng["FTR"]
        won_n  = ftr == side_next
        lost_n = (ftr == "H" and side_next == "A") or (ftr == "A" and side_next == "H")
        gf_n   = int(ng["FTHG"] if side_next == "H" else ng["FTAG"])
        gc_n   = int(ng["FTAG"] if side_next == "H" else ng["FTHG"])
        next_results.append({
            "equipo":           team,
            "liga":             ev["liga"],
            "racha_previa":     ev["racha_previa"],
            "gf_goleada":       ev["gf_goleada"],
            "lado_goleada":     ev["lado"],
            "resultado_sig":    ftr,
            "lado_sig":         side_next,
            "gano_sig":         int(won_n),
            "perdio_sig":       int(lost_n),
            "empato_sig":       int(ftr == "D"),
            "gf_sig":           gf_n,
            "gc_sig":           gc_n,
            "goleada_sig":      int(gf_n >= min_goles_goleada),  # repite goleada?
        })

    if not next_results:
        return {"sin_siguiente": True}

    nr = pd.DataFrame(next_results)
    total = len(nr)

    # Tasa global de victorias en cualquier partido (baseline)
    baseline_h = df["home_win"].mean()
    baseline_vic = (df["home_win"].mean() + df["away_win"].mean()) / 2  # aprox 50/50 no sirve
    # mejor baseline: % victorias en el partido INMEDIATAMENTE después de cualquier victoria
    df2_any_win = df2[
        ((df2["prev_res_h"] == "W") | (df2["prev_res_a"] == "W"))
    ]
    baseline_next_win = df2_any_win["home_win"].mean() if len(df2_any_win) > 0 else None

    # Tabla por intensidad de racha previa
    rows_int = []
    for umb, lbl in [(-2, "racha_2der"), (-3, "racha_3der"), (-4, "racha_4+der")]:
        sub = nr[nr["racha_previa"] <= umb]
        if len(sub) < 3: continue
        rows_int.append({
            "grupo":        lbl,
            "eventos":      len(sub),
            "pct_gana_sig": round(sub["gano_sig"].mean() * 100, 2),
            "pct_pierde_sig":round(sub["perdio_sig"].mean() * 100, 2),
            "pct_empata_sig":round(sub["empato_sig"].mean() * 100, 2),
            "avg_gf_sig":   round(sub["gf_sig"].mean(), 3),
            "avg_gc_sig":   round(sub["gc_sig"].mean(), 3),
            "pct_repite_goleada": round(sub["goleada_sig"].mean() * 100, 2),
        })

    return {
        "total_eventos":       total,
        "min_racha_requerida": min_bad,
        "min_goles_goleada":   min_goles_goleada,
        "pct_gana_siguiente":  round(nr["gano_sig"].mean() * 100, 2),
        "pct_pierde_siguiente":round(nr["perdio_sig"].mean() * 100, 2),
        "pct_empata_siguiente":round(nr["empato_sig"].mean() * 100, 2),
        "avg_gf_siguiente":    round(nr["gf_sig"].mean(), 3),
        "avg_gc_siguiente":    round(nr["gc_sig"].mean(), 3),
        "pct_repite_goleada":  round(nr["goleada_sig"].mean() * 100, 2),
        "baseline_pct_vic_tras_cualquier_victoria": round(baseline_h * 100, 2),
        "tabla_por_intensidad": pd.DataFrame(rows_int),
        "detalle":             nr,
    }


def stats_streak_goals_pattern(df: pd.DataFrame) -> pd.DataFrame:
    """Para cada posible racha previa del equipo local (-5 a +5),
    muestra avg goles marcados, recibidos, total, y si hay patrón
    de 'hambre' (racha mala → más goles) o 'relajación' (racha buena → menos goles)."""
    df2 = _build_streak_sequence(df)
    rows = []
    for val in range(-5, 6):
        if val < 0:
            mask = df2["prev_streak_h"] == val
        elif val > 0:
            mask = df2["prev_streak_h"] == val
        else:
            mask = df2["prev_streak_h"] == 0
        sub = df2[mask]
        if len(sub) < 10: continue
        rows.append({
            "racha_prev": val,
            "lbl":        f"{val:+d}" if val != 0 else "0",
            "partidos":   len(sub),
            "avg_gf":     round(sub["FTHG"].mean(), 3),
            "avg_gc":     round(sub["FTAG"].mean(), 3),
            "avg_total":  round(sub["total_goals"].mean(), 3),
            "pct_H":      round(sub["home_win"].mean() * 100, 2),
            "pct_D":      round(sub["draw"].mean() * 100, 2),
            "pct_A":      round(sub["away_win"].mean() * 100, 2),
            "pct_btts":   round(sub["btts"].mean() * 100, 2),
            "pct_over25": round(sub["over25"].mean() * 100, 2),
            "pct_goleada_h": round((sub["FTHG"] >= 4).mean() * 100, 2),
        })
    return pd.DataFrame(rows)


def stats_rebound_after_loss(df: pd.DataFrame) -> pd.DataFrame:
    """Rebote después de una derrota: ¿el partido siguiente es mejor o peor?
    Clasifica según cuántos goles se recibieron en la derrota:
      - derrota_minima:  perdió 0-1 o 1-2 (gc-gf = 1)
      - derrota_clara:   gc-gf = 2
      - goleada_recibida: gc-gf >= 3
    Y para cada grupo muestra qué pasa en el siguiente partido."""
    df2 = _build_streak_sequence(df)

    def cat_derrota(res, gf, gc):
        if res != "L": return None
        diff = gc - gf
        if diff == 1: return "derrota_minima (1 gol)"
        if diff == 2: return "derrota_clara (2 goles)"
        return "goleada_recibida (3+)"

    df2["cat_derrota_prev"] = df2.apply(
        lambda r: cat_derrota(r["prev_res_h"], r["prev_gf_h"], r["prev_gc_h"]), axis=1)

    rows = []
    for cat in ["derrota_minima (1 gol)", "derrota_clara (2 goles)", "goleada_recibida (3+)"]:
        sub = df2[df2["cat_derrota_prev"] == cat]
        if len(sub) < 5: continue
        rows.append({
            "derrota_previa":  cat,
            "partidos":        len(sub),
            "pct_rebote_vic":  round(sub["home_win"].mean() * 100, 2),
            "pct_empate":      round(sub["draw"].mean() * 100, 2),
            "pct_sigue_perdiendo": round(sub["away_win"].mean() * 100, 2),
            "avg_gf":          round(sub["FTHG"].mean(), 3),
            "avg_gc":          round(sub["FTAG"].mean(), 3),
            "pct_btts":        round(sub["btts"].mean() * 100, 2),
            "pct_over25":      round(sub["over25"].mean() * 100, 2),
            "avg_AvgH":        round(sub["AvgH"].mean(), 3),
        })
    return pd.DataFrame(rows)


def stats_winning_streak_end(df: pd.DataFrame) -> pd.DataFrame:
    """¿Cómo se rompen las rachas buenas? Cuando un equipo lleva N victorias
    consecutivas, ¿cómo pierde o empata?
    Analiza el partido que ROMPE la racha ganadora por longitud de la racha."""
    df2 = _build_streak_sequence(df)

    rows = []
    for umb, lbl in [(2,"tras 2vic"), (3,"tras 3vic"), (4,"tras 4vic"), (5,"tras 5+vic")]:
        if umb == 5:
            mask = (df2["prev_streak_h"] >= 5) & (df2["home_win"] == 0)
        else:
            mask = (df2["prev_streak_h"] == umb) & (df2["home_win"] == 0)
        sub = df2[mask]
        if len(sub) < 5: continue
        rows.append({
            "racha_rota":          lbl,
            "partidos":            len(sub),
            "pct_empate":          round(sub["draw"].mean() * 100, 2),
            "pct_derrota":         round(sub["away_win"].mean() * 100, 2),
            "avg_gf_en_derrota":   round(sub["FTHG"].mean(), 3),
            "avg_gc_en_derrota":   round(sub["FTAG"].mean(), 3),
            "pct_cs_roto":         round(sub["clean_sheet_h"].mean() * 100, 2),
            "pct_btts":            round(sub["btts"].mean() * 100, 2),
            "avg_AvgH":            round(sub["AvgH"].mean(), 3),
        })
    return pd.DataFrame(rows)


def stats_streak_by_team(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Por equipo: % de victorias según racha previa buena vs mala.
    Detecta qué equipos reaccionan mejor tras una mala racha (alta tasa de rebote)
    y qué equipos se 'caen' más cuando van en racha buena."""
    df2 = _build_streak_sequence(df)
    equipos = set(df["HomeTeam"].unique()) | set(df["AwayTeam"].unique())
    rows = []
    for team in equipos:
        # partidos como local
        sub = df2[df2["HomeTeam"] == team]
        if len(sub) < 20: continue
        malo = sub[sub["prev_streak_h"] <= -2]
        bueno = sub[sub["prev_streak_h"] >= 2]
        neutro = sub[sub["prev_streak_h"].between(-1, 1)]
        rows.append({
            "equipo":             team,
            "pj_local":           len(sub),
            "pct_vic_tras_mala":  round(malo["home_win"].mean() * 100, 2) if len(malo) >= 3 else None,
            "n_tras_mala":        len(malo),
            "pct_vic_tras_buena": round(bueno["home_win"].mean() * 100, 2) if len(bueno) >= 3 else None,
            "n_tras_buena":       len(bueno),
            "pct_vic_neutro":     round(neutro["home_win"].mean() * 100, 2) if len(neutro) >= 3 else None,
            "avg_gf_tras_mala":   round(malo["FTHG"].mean(), 3) if len(malo) >= 3 else None,
            "avg_gf_tras_buena":  round(bueno["FTHG"].mean(), 3) if len(bueno) >= 3 else None,
        })
    df_out = pd.DataFrame(rows).dropna(subset=["pct_vic_tras_mala","pct_vic_tras_buena"])
    df_out["rebote"] = round(df_out["pct_vic_tras_mala"] - df_out["pct_vic_neutro"], 2)
    df_out["caida"]  = round(df_out["pct_vic_tras_buena"] - df_out["pct_vic_neutro"], 2)
    return df_out.sort_values("rebote", ascending=False).head(top_n).reset_index(drop=True)


def stats_draw_streak_effect(df: pd.DataFrame) -> pd.DataFrame:
    """¿Qué pasa después de N empates consecutivos?
    Equipos que encadenan empates tienden a 'romper' con una victoria o derrota contundente."""
    df2 = df.sort_values("Date").copy()
    emp_streak = {}
    prev_emp = []
    for _, row in df2.iterrows():
        for team, side in [(row["HomeTeam"],"H"), (row["AwayTeam"],"A")]:
            emp_streak.setdefault(team, 0)
        prev_emp.append(emp_streak[row["HomeTeam"]])
        # actualizar
        for team, side in [(row["HomeTeam"],"H"), (row["AwayTeam"],"A")]:
            if row["FTR"] == "D":
                emp_streak[team] += 1
            else:
                emp_streak[team] = 0

    df2["prev_emp_streak_h"] = prev_emp
    rows = []
    for n in range(0, 5):
        sub = df2[df2["prev_emp_streak_h"] == n]
        if len(sub) < 5: continue
        rows.append({
            "empates_previos_local": n,
            "partidos":         len(sub),
            "pct_H":            round(sub["home_win"].mean() * 100, 2),
            "pct_D":            round(sub["draw"].mean() * 100, 2),
            "pct_A":            round(sub["away_win"].mean() * 100, 2),
            "avg_goles":        round(sub["total_goals"].mean(), 3),
            "pct_btts":         round(sub["btts"].mean() * 100, 2),
            "pct_over25":       round(sub["over25"].mean() * 100, 2),
        })
    return pd.DataFrame(rows)



# motor generico de rachas binarias
# funciona para cualquier columna 0/1: over25, btts, under25, gol_ht, over35, goalless...

def _binary_streaks_per_match(df: pd.DataFrame, flag: str) -> pd.DataFrame:
    """Calcula, partido a partido (ordenados por fecha), la racha
    consecutiva previa de flag=1 y la racha consecutiva previa de flag=0.
    Devuelve df enriquecido con:
      streak_on  : cuántos partidos seguidos con flag=1 antes de este
      streak_off : cuántos partidos seguidos con flag=0 antes de este
    (Se calculan a nivel de partido, no de equipo — son rachas del dataset
    completo filtrado, útil para tendencias de mercado y de liga.)"""
    df2 = df.sort_values("Date").copy()
    on_cur = 0; off_cur = 0
    ons = []; offs = []
    for val in df2[flag]:
        ons.append(on_cur); offs.append(off_cur)
        if val == 1:
            on_cur += 1; off_cur = 0
        else:
            off_cur += 1; on_cur = 0
    df2["streak_on"]  = ons
    df2["streak_off"] = offs
    return df2


def _binary_streaks_per_team(df: pd.DataFrame, flag: str) -> pd.DataFrame:
    """Igual que _binary_streaks_per_match pero lleva racha independiente
    por equipo (local+visitante combinados en orden cronológico).
    Devuelve df con:
      team_streak_on_h  / team_streak_off_h  (perspectiva del local)
      team_streak_on_a  / team_streak_off_a  (perspectiva del visitante)
    """
    df2 = df.sort_values("Date").copy()

    # Para el flag hay que saber el valor del partido para cada equipo.
    # Usamos el flag tal cual (aplica al partido, no al equipo individual).
    on_h = {}; off_h = {}
    on_a = {}; off_a = {}

    ts_on_h = []; ts_off_h = []
    ts_on_a = []; ts_off_a = []

    for _, row in df2.iterrows():
        ht = row["HomeTeam"]; at = row["AwayTeam"]
        val = int(row[flag])

        for d in [on_h, off_h, on_a, off_a]:
            d.setdefault(ht, 0); d.setdefault(at, 0)

        ts_on_h.append(on_h[ht]);  ts_off_h.append(off_h[ht])
        ts_on_a.append(on_a[at]);  ts_off_a.append(off_a[at])

        # actualizar local
        if val == 1:
            on_h[ht] += 1; off_h[ht] = 0
            on_a[at] += 1; off_a[at] = 0
        else:
            off_h[ht] += 1; on_h[ht] = 0
            off_a[at] += 1; on_a[at] = 0

    df2["team_streak_on_h"]  = ts_on_h
    df2["team_streak_off_h"] = ts_off_h
    df2["team_streak_on_a"]  = ts_on_a
    df2["team_streak_off_a"] = ts_off_a
    return df2


# tabla 1: distribucion de longitud de rachas (cuanto duran en promedio)

def streak_length_distribution(df: pd.DataFrame, flag: str, label: str) -> pd.DataFrame:
    """¿Cuánto duran las rachas de flag=1 y de flag=0?
    Devuelve tabla con longitud 1,2,3,4,5,6+ y su frecuencia."""
    df2 = df.sort_values("Date").copy()
    vals = df2[flag].tolist()
    # calcular longitudes de rachas
    runs_on = []; runs_off = []
    cur_val = vals[0]; cur_len = 1
    for v in vals[1:]:
        if v == cur_val:
            cur_len += 1
        else:
            if cur_val == 1: runs_on.append(cur_len)
            else:            runs_off.append(cur_len)
            cur_val = v; cur_len = 1
    if cur_val == 1: runs_on.append(cur_len)
    else:            runs_off.append(cur_len)

    rows = []
    for tipo, runs in [("racha_ON (flag=1)", runs_on), ("racha_OFF (flag=0)", runs_off)]:
        total = len(runs)
        if total == 0: continue
        for n in range(1, 8):
            cnt = sum(1 for r in runs if (r == n if n < 7 else r >= 7))
            rows.append({
                "flag":          label,
                "tipo":          tipo,
                "longitud":      f"{n}" if n < 7 else "7+",
                "rachas":        cnt,
                "pct_rachas":    round(cnt / total * 100, 2),
                "avg_longitud":  round(sum(runs) / total, 2),
                "max_longitud":  max(runs),
                "median":        round(float(pd.Series(runs).median()), 1),
            })
    return pd.DataFrame(rows)


# tabla 2: que pasa despues de una racha larga

def streak_after_effect(df: pd.DataFrame, flag: str, label: str,
                         outcome_cols: list = None) -> pd.DataFrame:
    """Tras N partidos consecutivos con flag=1 (o flag=0),
    ¿qué probabilidad hay de que el SIGUIENTE partido también cumpla el flag?
    outcome_cols: columnas extra para calcular su media en cada grupo."""
    if outcome_cols is None:
        outcome_cols = ["total_goals", "btts", "over25", "home_win", "draw"]

    df2 = _binary_streaks_per_match(df, flag)
    rows = []

    for src, streak_col, lbl_src in [
        ("tras_racha_ON",  "streak_on",  f"tras N seguidos CON {label}"),
        ("tras_racha_OFF", "streak_off", f"tras N seguidos SIN {label}"),
    ]:
        for n in range(0, 8):
            sub = df2[df2[streak_col] == n] if n < 7 else df2[df2[streak_col] >= 7]
            if len(sub) < 5: continue
            row = {
                "flag":          label,
                "situacion":     lbl_src,
                "racha_previa":  n if n < 7 else "7+",
                "partidos":      len(sub),
                f"pct_{flag}":   round(sub[flag].mean() * 100, 2),
            }
            for col in outcome_cols:
                if col in sub.columns:
                    row[f"avg_{col}"] = round(sub[col].mean() * 100 if sub[col].max() <= 1
                                              else sub[col].mean(), 3)
            rows.append(row)
    return pd.DataFrame(rows)


# tabla 3: rachas maximas por equipo

def streak_max_by_team(df: pd.DataFrame, flag: str, label: str,
                        min_pj: int = 20, top_n: int = 15) -> pd.DataFrame:
    """Para cada equipo, calcula su racha máxima de flag=1 y de flag=0
    (en partidos donde participó, como local o visitante)."""
    equipos = set(df["HomeTeam"].unique()) | set(df["AwayTeam"].unique())
    rows = []
    df2 = df.sort_values("Date")
    for team in equipos:
        sub = df2[(df2["HomeTeam"] == team) | (df2["AwayTeam"] == team)]
        if len(sub) < min_pj: continue
        vals = sub[flag].tolist()
        max_on = 0; max_off = 0; cur_on = 0; cur_off = 0
        for v in vals:
            if v == 1:
                cur_on += 1; cur_off = 0
                max_on = max(max_on, cur_on)
            else:
                cur_off += 1; cur_on = 0
                max_off = max(max_off, cur_off)
        rows.append({
            "equipo":        team,
            "pj":            len(sub),
            f"max_racha_{label}_ON":  max_on,
            f"max_racha_{label}_OFF": max_off,
            f"pct_{label}":  round(sub[flag].mean() * 100, 2),
        })
    df_out = pd.DataFrame(rows)
    return df_out.sort_values(f"max_racha_{label}_ON", ascending=False).head(top_n).reset_index(drop=True)


# tabla 4: rachas maximas por liga

def streak_max_by_league(df: pd.DataFrame, flag: str, label: str) -> pd.DataFrame:
    """Racha máxima histórica de flag=1 y flag=0 por liga."""
    rows = []
    for liga in sorted(df["Div"].unique()):
        sub = df[df["Div"] == liga].sort_values("Date")
        vals = sub[flag].tolist()
        max_on = 0; max_off = 0; cur_on = 0; cur_off = 0
        for v in vals:
            if v == 1:
                cur_on += 1; cur_off = 0; max_on = max(max_on, cur_on)
            else:
                cur_off += 1; cur_on = 0; max_off = max(max_off, cur_off)
        rows.append({
            "liga":                    liga,
            f"max_racha_{label}_ON":  max_on,
            f"max_racha_{label}_OFF": max_off,
            f"pct_{label}":           round(sub[flag].mean() * 100, 2),
            "partidos":               len(sub),
        })
    return pd.DataFrame(rows)


# tabla 5: estado actual de la racha al cierre del dataset

def streak_current_state(df: pd.DataFrame, flag: str, label: str) -> pd.DataFrame:
    """¿En qué racha termina cada liga/equipo al final del dataset?
    Útil para saber si actualmente hay una racha larga activa."""
    df2 = df.sort_values("Date")
    rows = []
    for liga in sorted(df2["Div"].unique()):
        sub = df2[df2["Div"] == liga]
        vals = sub[flag].tolist()
        cur = 0; last = vals[-1] if vals else 0
        for v in reversed(vals):
            if v == last: cur += 1
            else: break
        rows.append({
            "liga":           liga,
            "estado_actual":  f"ON ({label})" if last == 1 else f"OFF (no {label})",
            "partidos_activos": cur,
            "ultimo_partido": sub["Date"].max(),
        })
    return pd.DataFrame(rows)


# tabla 6: cuotas altas que terminaron ganando, con su racha

def _add_high_odd_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega flags de cuota alta ganadora para distintos umbrales."""
    df2 = df.copy()
    for umb in [3.0, 4.0, 5.0, 6.0]:
        col = f"high_odd_{str(umb).replace('.','')}_win"
        # cuota alta ganadora: local >umb y gana local, O visitante >umb y gana visitante
        df2[col] = (
            ((df2["AvgH"] >= umb) & (df2["FTR"] == "H")) |
            ((df2["AvgA"] >= umb) & (df2["FTR"] == "A"))
        ).astype(int)
    # cuota de empate alta (>3.5) y sale empate
    df2["high_draw_win"] = ((df2["AvgD"] >= 3.5) & (df2["FTR"] == "D")).astype(int)
    # cuota local extrema (>5) y gana local
    df2["extreme_upset"] = ((df2["AvgH"] >= 5.0) & (df2["FTR"] == "H")).astype(int)
    # doble cuota alta: ambas >2.5
    df2["balanced_high"] = ((df2["AvgH"] >= 2.5) & (df2["AvgA"] >= 2.5)).astype(int)
    return df2


# tabla 7: flags de goles en primer y segundo tiempo

def _add_half_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Flags de over/under y btts para primer y segundo tiempo por separado."""
    df2 = df.copy()
    # primer tiempo
    df2["gol_ht"]       = (df2["ht_goals"] >= 1).astype(int)
    df2["over15_ht"]    = (df2["ht_goals"] >= 2).astype(int)
    df2["under05_ht"]   = (df2["ht_goals"] == 0).astype(int)
    df2["under15_ht"]   = (df2["ht_goals"] <= 1).astype(int)
    df2["btts_ht"]      = ((df2["HTHG"] > 0) & (df2["HTAG"] > 0)).astype(int)
    df2["no_btts_ht"]   = (1 - df2["btts_ht"])
    # segundo tiempo
    df2["gol_st"]       = (df2["second_half_goals"] >= 1).astype(int)
    df2["over15_st"]    = (df2["second_half_goals"] >= 2).astype(int)
    df2["over25_st"]    = (df2["second_half_goals"] >= 3).astype(int)
    df2["under05_st"]   = (df2["second_half_goals"] == 0).astype(int)
    df2["under15_st"]   = (df2["second_half_goals"] <= 1).astype(int)
    df2["btts_st"]      = ((df2["FTHG"] - df2["HTHG"] > 0) & (df2["FTAG"] - df2["HTAG"] > 0)).astype(int)
    df2["no_btts_st"]   = (1 - df2["btts_st"])
    # global extras
    df2["no_btts"]      = (1 - df2["btts"])
    df2["no_goalless"]  = (1 - df2["goalless"])
    df2["st_mas_ht"]    = (df2["second_half_goals"] > df2["ht_goals"]).astype(int)
    df2["st_igual_ht"]  = (df2["second_half_goals"] == df2["ht_goals"]).astype(int)
    return df2


# funcion principal que ejecuta todos los analisis de rachas

def run_all_binary_streaks(df: pd.DataFrame) -> dict:
    """Ejecuta el análisis completo de rachas para todos los flags definidos.
    Devuelve un dict con claves = nombre del flag, valor = dict de tablas."""

    df2 = _add_high_odd_flags(_add_half_flags(df))

    # Catálogo de flags a analizar
    FLAGS = [
        # resultados FT
        ("over25",       "over2.5_FT"),
        ("over35",       "over3.5_FT"),
        ("over15",       "over1.5_FT"),
        ("over45",       "over4.5_FT"),
        ("under25",      "under2.5_FT"),
        ("under15",      "under1.5_FT"),
        ("btts",         "btts_FT"),
        ("no_btts",      "no_btts_FT"),
        ("goalless",     "0-0_FT"),
        ("high_scoring", "5+goles_FT"),
        ("home_win",     "victoria_local"),
        ("away_win",     "victoria_visita"),
        ("draw",         "empate"),
        # primer tiempo
        ("gol_ht",       "gol_en_HT"),
        ("under05_ht",   "0goles_HT"),
        ("under15_ht",   "under1.5_HT"),
        ("over15_ht",    "over1.5_HT"),
        ("btts_ht",      "btts_HT"),
        ("no_btts_ht",   "no_btts_HT"),
        # segundo tiempo
        ("gol_st",       "gol_en_ST"),
        ("under05_st",   "0goles_ST"),
        ("under15_st",   "under1.5_ST"),
        ("over15_st",    "over1.5_ST"),
        ("over25_st",    "over2.5_ST"),
        ("btts_st",      "btts_ST"),
        ("no_btts_st",   "no_btts_ST"),
        ("st_mas_ht",    "ST_supera_HT"),
        # cuotas altas
        ("high_odd_30_win",  "cuota_alta_30+_gana"),
        ("high_odd_40_win",  "cuota_alta_40+_gana"),
        ("high_odd_50_win",  "cuota_alta_50+_gana"),
        ("high_odd_60_win",  "cuota_alta_60+_gana"),
        ("high_draw_win",    "empate_cuota_alta"),
        ("extreme_upset",    "sorpresa_extrema"),
        ("balanced_high",    "partido_abierto"),
        # clean sheets
        ("clean_sheet_h",    "cs_local"),
        ("clean_sheet_a",    "cs_visita"),
    ]

    outcome_cols = ["total_goals", "btts", "over25", "home_win", "draw", "goalless"]
    results = {}

    for flag, label in FLAGS:
        if flag not in df2.columns:
            continue
        results[flag] = {
            "label":        label,
            "distribucion": streak_length_distribution(df2, flag, label),
            "efecto":       streak_after_effect(df2, flag, label, outcome_cols),
            "max_equipo":   streak_max_by_team(df2, flag, label),
            "max_liga":     streak_max_by_league(df2, flag, label),
            "estado_vivo":  streak_current_state(df2, flag, label),
        }

    return results, df2   # devuelve df2 con todos los flags extra


def print_binary_streak_report(results: dict, flags_to_print: list = None) -> None:
    """Imprime el reporte completo de rachas binarias.
    flags_to_print: lista de claves a imprimir (None = todas)."""
    if flags_to_print is None:
        flags_to_print = list(results.keys())

    for flag in flags_to_print:
        if flag not in results: continue
        r = results[flag]
        lbl = r["label"]
        sep = "─" * 60

        print(f"\n{sep}")
        print(f"  RACHA: {lbl.upper()}")
        print(f"{sep}")

        print(f"\n  [distribución de longitud de rachas ON y OFF]")
        # mostrar resumen compacto
        dist = r["distribucion"]
        on  = dist[dist["tipo"].str.startswith("racha_ON")]
        off = dist[dist["tipo"].str.startswith("racha_OFF")]
        if not on.empty:
            print(f"  ON  → max {on['max_longitud'].max()} | "
                  f"median {on['median'].iloc[0]} | "
                  f"avg {on['avg_longitud'].iloc[0]}")
        if not off.empty:
            print(f"  OFF → max {off['max_longitud'].max()} | "
                  f"median {off['median'].iloc[0]} | "
                  f"avg {off['avg_longitud'].iloc[0]}")
        print_tabulate(dist[["tipo","longitud","rachas","pct_rachas"]].drop_duplicates(
            subset=["tipo","longitud"]))

        print(f"\n  [efecto de la racha previa → ¿cambia la prob del siguiente?]")
        ef = r["efecto"]
        # solo mostrar col del flag y goles para no saturar
        cols_show = ["situacion","racha_previa","partidos",
                     f"pct_{flag}", "avg_total_goals","avg_btts","avg_over25"]
        cols_show = [c for c in cols_show if c in ef.columns]
        print_tabulate(ef[cols_show])

        print(f"\n  [racha máxima por liga]")
        print_tabulate(r["max_liga"])

        print(f"\n  [top 10 equipos con racha más larga de {lbl}]")
        print_tabulate(r["max_equipo"].head(10))

        print(f"\n  [estado actual al final del dataset]")
        print_tabulate(r["estado_vivo"])


def streak_cross_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Cuando hay racha de over2.5, ¿cuánto dura la racha de btts en el mismo periodo?
    Correlación entre rachas simultáneas de distintos flags."""
    df2 = _add_half_flags(df).sort_values("Date").copy()

    flag_pairs = [
        ("over25",  "btts",       "over2.5 y btts"),
        ("over25",  "over35",     "over2.5 y over3.5"),
        ("btts",    "gol_ht",     "btts y gol_en_HT"),
        ("btts",    "gol_st",     "btts y gol_en_ST"),
        ("over25",  "gol_ht",     "over2.5 y gol_HT"),
        ("goalless","under05_ht", "0-0_FT y 0-0_HT"),
        ("home_win","gol_ht",     "local gana y hay gol HT"),
        ("draw",    "btts",       "empate y btts"),
        ("over35",  "btts_ht",    "over3.5 y btts_HT"),
    ]

    rows = []
    for f1, f2, lbl in flag_pairs:
        if f1 not in df2.columns or f2 not in df2.columns: continue
        corr = round(df2[f1].corr(df2[f2]), 4)
        # cuando f1=1: ¿cuánto % f2=1?
        sub1 = df2[df2[f1] == 1]
        sub0 = df2[df2[f1] == 0]
        rows.append({
            "combinacion":          lbl,
            "corr":                 corr,
            f"pct_{f2}_cuando_{f1}=1": round(sub1[f2].mean() * 100, 2) if len(sub1) else 0,
            f"pct_{f2}_cuando_{f1}=0": round(sub0[f2].mean() * 100, 2) if len(sub0) else 0,
            "diferencia":           round((sub1[f2].mean() - sub0[f2].mean()) * 100, 2)
                                    if len(sub1) and len(sub0) else 0,
        })
    return pd.DataFrame(rows)


def streak_odds_high_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Análisis profundo de cuotas altas ganadoras:
    ¿En qué momento de una racha de 'favoritos ganando' aparece la sorpresa?
    Agrupa los partidos según cuántos favoritos consecutivos ganaron antes."""
    df2 = _add_high_odd_flags(df).sort_values("Date").copy()

    # flag: favorito ganó = cuota local < 2.0 y ganó, o cuota visitante < 2.0 y ganó
    df2["fav_win"] = (
        ((df2["AvgH"] < 2.0) & (df2["FTR"] == "H")) |
        ((df2["AvgA"] < 2.0) & (df2["FTR"] == "A"))
    ).astype(int)
    df2["upset_win"] = (1 - df2["fav_win"])  # sorpresa = no ganó el favorito (<2.0)

    df3 = _binary_streaks_per_match(df2.assign(**{
        "high_odd_30_win": df2["high_odd_30_win"]
    }), "high_odd_30_win")

    rows = []
    for n in range(0, 7):
        sub = df3[df3["streak_on"] == n] if n < 6 else df3[df3["streak_on"] >= 6]
        if len(sub) < 5: continue
        rows.append({
            "sorpresas_previas":    n if n < 6 else "6+",
            "partidos":             len(sub),
            "pct_otra_sorpresa":    round(sub["high_odd_30_win"].mean() * 100, 2),
            "avg_cuota_H":          round(sub["AvgH"].mean(), 3),
            "avg_cuota_A":          round(sub["AvgA"].mean(), 3),
            "pct_fav_gana_hoy":     round(sub["fav_win"].mean() * 100, 2),
            "avg_goles":            round(sub["total_goals"].mean(), 3),
            "pct_btts":             round(sub["btts"].mean() * 100, 2),
        })
    return pd.DataFrame(rows)


def streak_team_flag_profile(df: pd.DataFrame, flag: str, label: str,
                              min_pj: int = 30, top_n: int = 15) -> pd.DataFrame:
    """Por equipo: desglose de su racha actual, pct histórico y si está
    'caliente' (racha actual > media histórica) o 'fría'."""
    df2 = df.sort_values("Date")
    equipos = set(df["HomeTeam"].unique()) | set(df["AwayTeam"].unique())
    rows = []
    for team in equipos:
        sub = df2[(df2["HomeTeam"] == team) | (df2["AwayTeam"] == team)]
        if len(sub) < min_pj: continue
        vals = sub[flag].tolist()
        pct_hist = round(sum(vals) / len(vals) * 100, 2)
        # racha actual
        cur_val = vals[-1]; cur_len = 0
        for v in reversed(vals):
            if v == cur_val: cur_len += 1
            else: break
        max_on = 0; c = 0
        for v in vals:
            if v == 1: c += 1; max_on = max(max_on, c)
            else: c = 0
        rows.append({
            "equipo":         team,
            "pj":             len(sub),
            f"pct_{label}":   pct_hist,
            "racha_actual_tipo": f"ON ({label})" if cur_val == 1 else f"OFF",
            "racha_actual_n": cur_len,
            f"max_racha_ON":  max_on,
            "caliente":       "SI" if cur_val == 1 and cur_len >= 3 else
                              "FRIA" if cur_val == 0 and cur_len >= 3 else "-",
        })
    return pd.DataFrame(rows).sort_values("racha_actual_n", ascending=False).head(top_n).reset_index(drop=True)


def streak_summary_table(results: dict) -> pd.DataFrame:
    """Tabla resumen de todos los flags: max racha ON, max racha OFF, pct global."""
    rows = []
    for flag, r in results.items():
        dist = r["distribucion"]
        on  = dist[dist["tipo"].str.startswith("racha_ON")]
        off = dist[dist["tipo"].str.startswith("racha_OFF")]
        ligs = r["max_liga"]
        rows.append({
            "flag":          flag,
            "label":         r["label"],
            "max_racha_ON_global":  int(on["max_longitud"].max()) if not on.empty else 0,
            "max_racha_OFF_global": int(off["max_longitud"].max()) if not off.empty else 0,
            "avg_racha_ON":         float(on["avg_longitud"].iloc[0]) if not on.empty else 0,
            "avg_racha_OFF":        float(off["avg_longitud"].iloc[0]) if not off.empty else 0,
            "pct_flag_global":      round(ligs[f"pct_{flag}"].mean(), 2) if f"pct_{flag}" in ligs.columns else None,
        })
    return pd.DataFrame(rows).sort_values("max_racha_ON_global", ascending=False).reset_index(drop=True)


def stats_btts_by_ht_state(df: pd.DataFrame) -> dict:
    """% BTTS segun si el primer tiempo termino sin goles o con goles."""
    ht0 = df[df["ht_goals"]==0]; ht1p = df[df["ht_goals"]>=1]
    return {
        "ht_goalless_n":    len(ht0),
        "btts_si_ht0":      round(ht0["btts"].mean()*100,2)  if len(ht0)  else 0,
        "over25_si_ht0":    round(ht0["over25"].mean()*100,2) if len(ht0)  else 0,
        "avg_goles_si_ht0": round(ht0["total_goals"].mean(),3)if len(ht0)  else 0,
        "ht_con_gol_n":     len(ht1p),
        "btts_si_ht1p":     round(ht1p["btts"].mean()*100,2) if len(ht1p) else 0,
        "over25_si_ht1p":   round(ht1p["over25"].mean()*100,2)if len(ht1p) else 0,
        "avg_goles_si_ht1p":round(ht1p["total_goals"].mean(),3)if len(ht1p) else 0,
        # separar ht=1 vs ht=2+
        "btts_si_ht1":      round(df[df["ht_goals"]==1]["btts"].mean()*100,2) if len(df[df["ht_goals"]==1]) else 0,
        "btts_si_ht2p":     round(df[df["ht_goals"]>=2]["btts"].mean()*100,2) if len(df[df["ht_goals"]>=2]) else 0,
    }


def stats_btts_ht_by_team(df: pd.DataFrame, top_n=15) -> pd.DataFrame:
    """Equipos donde mas se da BTTS viniendo de un 0-0 al descanso."""
    ht0 = df[df["ht_goals"]==0].copy()
    rows=[]
    equipos = set(ht0["HomeTeam"].unique()) | set(ht0["AwayTeam"].unique())
    for team in equipos:
        sub = ht0[(ht0["HomeTeam"]==team)|(ht0["AwayTeam"]==team)]
        if len(sub)<10: continue
        rows.append({
            "equipo":team, "partidos_ht0":len(sub),
            "btts_desde_ht0":round(sub["btts"].mean()*100,2),
            "over25_desde_ht0":round(sub["over25"].mean()*100,2),
            "avg_goles_ft":round(sub["total_goals"].mean(),3),
        })
    return pd.DataFrame(rows).sort_values("btts_desde_ht0",ascending=False).head(top_n).reset_index(drop=True)


# partidos que van 0-0 al descanso: cuanto explotan en el segundo tiempo

def stats_teams_explosion_after_00ht(df: pd.DataFrame, min_goals_ft=3, min_pj=15) -> pd.DataFrame:
    """Equipos cuyos partidos 0-0 al HT terminan con mas goles en FT."""
    ht0 = df[df["ht_goals"]==0].copy()
    rows=[]
    equipos = set(ht0["HomeTeam"].unique()) | set(ht0["AwayTeam"].unique())
    for team in equipos:
        sub = ht0[(ht0["HomeTeam"]==team)|(ht0["AwayTeam"]==team)]
        if len(sub)<min_pj: continue
        rows.append({
            "equipo":team, "pj_ht0":len(sub),
            "avg_goles_ft":round(sub["total_goals"].mean(),3),
            "pct_3mas_goles":round((sub["total_goals"]>=min_goals_ft).mean()*100,2),
            "pct_btts":round(sub["btts"].mean()*100,2),
            "pct_over25":round(sub["over25"].mean()*100,2),
            "pct_goalless":round(sub["goalless"].mean()*100,2),
        })
    return pd.DataFrame(rows).sort_values("avg_goles_ft",ascending=False).head(20).reset_index(drop=True)


# equipos que suelen ir ganando al descanso

def stats_teams_winning_at_ht(df: pd.DataFrame, min_pj=30) -> pd.DataFrame:
    """Equipos que con mas frecuencia van ganando al descanso (HTHG > HTAG como local / HTAG > HTHG como visitante)."""
    rows=[]
    equipos = set(df["HomeTeam"].unique()) | set(df["AwayTeam"].unique())
    for team in equipos:
        loc = df[df["HomeTeam"]==team]
        vis = df[df["AwayTeam"]==team]
        pj = len(loc)+len(vis)
        if pj<min_pj: continue
        ganando_ht_loc = (loc["HTHG"]>loc["HTAG"]).sum()
        ganando_ht_vis = (vis["HTAG"]>vis["HTHG"]).sum()
        total_ganando_ht = int(ganando_ht_loc+ganando_ht_vis)
        rows.append({
            "equipo":team, "pj_total":pj,
            "ganando_ht":total_ganando_ht,
            "pct_ganando_ht":round(total_ganando_ht/pj*100,2),
            "ganando_ht_loc":int(ganando_ht_loc),
            "ganando_ht_vis":int(ganando_ht_vis),
        })
    return pd.DataFrame(rows).sort_values("pct_ganando_ht",ascending=False).head(20).reset_index(drop=True)


# remontadas del favorito, con enfoque especial en Barcelona

def stats_comeback_by_fav(df: pd.DataFrame) -> pd.DataFrame:
    """Remontadas segun si el equipo que remonto era favorito o no."""
    rows=[]
    for lbl, mask_remon, odd_col in [
        ("local_remonta",  (df["HTR"]=="A")&(df["FTR"]=="H"), "AvgH"),
        ("visita_remonta", (df["HTR"]=="H")&(df["FTR"]=="A"), "AvgA"),
    ]:
        sub = df[mask_remon].copy()
        if len(sub)==0: continue
        fav  = sub[sub[odd_col]<2.0]
        nofav= sub[sub[odd_col]>=2.0]
        rows.append({
            "tipo":lbl, "total_remontadas":len(sub),
            "pct_del_total":round(len(sub)/len(df)*100,2),
            "avg_odd_ganador":round(sub[odd_col].mean(),3),
            "era_fav_n":len(fav),       "pct_era_fav":round(len(fav)/len(sub)*100,2) if len(sub) else 0,
            "era_nofav_n":len(nofav),   "pct_era_nofav":round(len(nofav)/len(sub)*100,2) if len(sub) else 0,
            "avg_goles_remon":round(sub["total_goals"].mean(),3),
            "pct_btts_remon": round(sub["btts"].mean()*100,2),
        })
    return pd.DataFrame(rows)


def stats_barcelona_remontadas(df: pd.DataFrame) -> dict:
    """Analisis especifico de remontadas del Barcelona (local y visitante)."""
    barca = df[(df["HomeTeam"]=="Barcelona")|(df["AwayTeam"]=="Barcelona")].copy()
    if len(barca)==0:
        return {"error":"Barcelona no encontrado en el dataset"}
    barca_loc = barca[barca["HomeTeam"]=="Barcelona"]
    barca_vis = barca[barca["AwayTeam"]=="Barcelona"]
    # iba perdiendo al HT
    perdiendo_ht_loc = barca_loc[barca_loc["HTHG"]<barca_loc["HTAG"]]
    perdiendo_ht_vis = barca_vis[barca_vis["HTAG"]<barca_vis["HTHG"]]
    perdiendo_ht_all = pd.concat([perdiendo_ht_loc, perdiendo_ht_vis])
    # remonto: FTR==H siendo local o FTR==A siendo visitante
    remon_loc = perdiendo_ht_loc[perdiendo_ht_loc["FTR"]=="H"]
    remon_vis = perdiendo_ht_vis[perdiendo_ht_vis["FTR"]=="A"]
    # iba ganando al HT
    ganando_ht_loc = barca_loc[barca_loc["HTHG"]>barca_loc["HTAG"]]
    ganando_ht_vis = barca_vis[barca_vis["HTAG"]>barca_vis["HTHG"]]
    # perdio ventaja HT
    choke_loc = ganando_ht_loc[ganando_ht_loc["FTR"]=="A"]
    choke_vis = ganando_ht_vis[ganando_ht_vis["FTR"]=="H"]
    return {
        "total_partidos":len(barca),
        "pj_local":len(barca_loc), "pj_visitante":len(barca_vis),
        "pct_H_local":round((barca_loc["FTR"]=="H").mean()*100,2) if len(barca_loc) else 0,
        "pct_H_visitante":round((barca_vis["FTR"]=="A").mean()*100,2) if len(barca_vis) else 0,
        # iba perdiendo HT
        "veces_perdiendo_ht_total":len(perdiendo_ht_all),
        "veces_perdiendo_ht_loc":len(perdiendo_ht_loc),
        "veces_perdiendo_ht_vis":len(perdiendo_ht_vis),
        "remontadas_loc":len(remon_loc),
        "remontadas_vis":len(remon_vis),
        "remontadas_total":len(remon_loc)+len(remon_vis),
        "pct_remonta_si_pierde_ht":round((len(remon_loc)+len(remon_vis))/len(perdiendo_ht_all)*100,2) if len(perdiendo_ht_all) else 0,
        # iba ganando HT
        "veces_ganando_ht_loc":len(ganando_ht_loc),
        "veces_ganando_ht_vis":len(ganando_ht_vis),
        "chokes_loc":len(choke_loc),
        "chokes_vis":len(choke_vis),
        "pct_pierde_ventaja_ht":round((len(choke_loc)+len(choke_vis))/(len(ganando_ht_loc)+len(ganando_ht_vis))*100,2)
            if (len(ganando_ht_loc)+len(ganando_ht_vis))>0 else 0,
        "avg_goles_cuando_remonta":round(pd.concat([remon_loc,remon_vis])["total_goals"].mean(),3) if (len(remon_loc)+len(remon_vis))>0 else 0,
    }


def stats_barcelona_remontadas_detail(df: pd.DataFrame) -> pd.DataFrame:
    """Listado de cada partido donde Barcelona iba perdiendo al HT."""
    barca_loc = df[(df["HomeTeam"]=="Barcelona")&(df["HTHG"]<df["HTAG"])].copy()
    barca_vis = df[(df["AwayTeam"]=="Barcelona")&(df["HTAG"]<df["HTHG"])].copy()
    barca_loc["rol"]="local"; barca_vis["rol"]="visitante"
    barca_loc["remonto"]=(barca_loc["FTR"]=="H").astype(int)
    barca_vis["remonto"]=(barca_vis["FTR"]=="A").astype(int)
    cols=["Season_label","Div","Date","HomeTeam","AwayTeam","HTHG","HTAG","FTHG","FTAG","FTR","rol","remonto"]
    return pd.concat([barca_loc,barca_vis])[cols].sort_values("Date").reset_index(drop=True)


# comparativa entre el inicio y el final de temporada

def stats_season_thirds(df: pd.DataFrame) -> pd.DataFrame:
    """Compara inicio (primer tercio), mitad y final (ultimo tercio) de cada temporada por liga."""
    rows=[]
    for div in sorted(df["Div"].unique()):
        for slbl in df["Season_label"].dropna().unique():
            sub = df[(df["Div"]==div)&(df["Season_label"]==slbl)].sort_values("Date")
            if len(sub)<30: continue
            n = len(sub); t1 = sub.iloc[:n//3]; t3 = sub.iloc[2*n//3:]
            for seg, s in [("inicio",t1),("final",t3)]:
                rows.append({
                    "liga":div,"temporada":slbl,"segmento":seg,"partidos":len(s),
                    "pct_H":round(s["home_win"].mean()*100,2),
                    "pct_D":round(s["draw"].mean()*100,2),
                    "pct_A":round(s["away_win"].mean()*100,2),
                    "avg_goles":round(s["total_goals"].mean(),3),
                    "pct_btts":round(s["btts"].mean()*100,2),
                    "pct_over25":round(s["over25"].mean()*100,2),
                    "avg_AvgH":round(s["AvgH"].mean(),3),
                })
    return pd.DataFrame(rows)


def stats_top_teams_season_thirds(df: pd.DataFrame) -> pd.DataFrame:
    """Para equipos que terminaron top3, compara su rendimiento en inicio vs final de cada temporada/liga."""
    rows=[]
    for div in sorted(df["Div"].unique()):
        for slbl in df["Season_label"].dropna().unique():
            sub = df[(df["Div"]==div)&(df["Season_label"]==slbl)].sort_values("Date")
            if len(sub)<30: continue
            standing = calc_standings(sub)
            top3 = list(standing.head(3)["equipo"])
            n = len(sub); t1 = sub.iloc[:n//3]; t3 = sub.iloc[2*n//3:]
            for equipo in top3:
                for seg, s in [("inicio",t1),("final",t3)]:
                    es = s[(s["HomeTeam"]==equipo)|(s["AwayTeam"]==equipo)]
                    if len(es)<3: continue
                    pts=0
                    for _,r in es.iterrows():
                        side="H" if r["HomeTeam"]==equipo else "A"
                        if r["FTR"]==side: pts+=3
                        elif r["FTR"]=="D": pts+=1
                    rows.append({
                        "liga":div,"temporada":slbl,"equipo":equipo,
                        "segmento":seg,"pj":len(es),"pts":pts,
                        "pct_vic":round(sum(1 for _,r in es.iterrows() if r["FTR"]==("H" if r["HomeTeam"]==equipo else "A"))/len(es)*100,2),
                        "avg_gf":round(es.apply(lambda r: r["FTHG"] if r["HomeTeam"]==equipo else r["FTAG"],axis=1).mean(),3),
                        "avg_gc":round(es.apply(lambda r: r["FTAG"] if r["HomeTeam"]==equipo else r["FTHG"],axis=1).mean(),3),
                    })
    return pd.DataFrame(rows)


# cuotas con valores llamativos (3.33, 4.44, etc.) y su rendimiento real

def stats_gematric_odds(df: pd.DataFrame) -> pd.DataFrame:
    """Analisis de cuotas con valores 'gematricos' o llamativos (3.33, 4.44, 2.22, etc.)
    y rangos del tipo +333 (cuota 4.33), +200 (3.0), +150 (2.5), recurrentes.
    Comprueba si hay anomalias en el % de acierto alrededor de estas cuotas.""" 
    targets = {
        "2.00 (±0.05)": (1.95,2.05),
        "2.22 (±0.05)": (2.17,2.27),
        "2.50 (±0.05)": (2.45,2.55),
        "3.00 (±0.05)": (2.95,3.05),
        "3.33 (±0.05)": (3.28,3.38),
        "4.00 (±0.05)": (3.95,4.05),
        "4.33 (±0.05)": (4.28,4.38),
        "4.44 (±0.05)": (4.39,4.49),
        "5.00 (±0.05)": (4.95,5.05),
        "6.00 (±0.05)": (5.95,6.05),
        "6.66 (±0.05)": (6.61,6.71),
        "10.0 (±0.2)":  (9.80,10.20),
        "11.0 (±0.2)":  (10.80,11.20),
    }
    rows=[]
    for lbl,(lo,hi) in targets.items():
        # cuota local en ese rango
        mask_h = (df["AvgH"]>=lo)&(df["AvgH"]<hi)
        mask_a = (df["AvgA"]>=lo)&(df["AvgA"]<hi)
        sub_h = df[mask_h]; sub_a = df[mask_a]
        imp_h = round(1/((lo+hi)/2)*100,2)
        if len(sub_h)>=5:
            rows.append({
                "cuota_lbl":lbl,"lado":"local","n":len(sub_h),
                "imp_%":imp_h,
                "real_%":round(sub_h["home_win"].mean()*100,2),
                "edge":round(sub_h["home_win"].mean()*100-imp_h,2),
                "pct_btts":round(sub_h["btts"].mean()*100,2),
                "avg_goles":round(sub_h["total_goals"].mean(),3),
            })
        if len(sub_a)>=5:
            rows.append({
                "cuota_lbl":lbl,"lado":"visitante","n":len(sub_a),
                "imp_%":imp_h,
                "real_%":round(sub_a["away_win"].mean()*100,2),
                "edge":round(sub_a["away_win"].mean()*100-imp_h,2),
                "pct_btts":round(sub_a["btts"].mean()*100,2),
                "avg_goles":round(sub_a["total_goals"].mean(),3),
            })
    return pd.DataFrame(rows).sort_values(["cuota_lbl","lado"]).reset_index(drop=True)


def stats_bet_roi_especial(df: pd.DataFrame) -> pd.DataFrame:
    """ROI real si se apuesta 1 unidad plana a cuotas especiales (3.33, -333, etc.)
    en los tres posibles resultados (local, empate, visitante), apertura y cierre.

    Cuotas americanas de referencia:
      +200 -> decimal 3.0  |  +250 -> 3.5  |  +333 -> 4.33
      -333 -> 1.30         |  -200 -> 1.5  |  -150 -> 1.67

    Por cada cuota objetivo y lado:
      ROI%   = (wins * odd - n) / n * 100
      yield  = ROI% / 100
      edge%  = real_% - imp_%  (positivo = mercado infravalorado)
    """
    targets = {
        1.30: "-333 (fav fuerte)",
        1.50: "-200 (fav claro)",
        1.67: "-150 (fav moderado)",
        2.00: "Evens / +100",
        2.50: "+150",
        3.00: "+200",
        3.33: "+233 / cuota 3.33",
        3.50: "+250",
        4.00: "+300",
        4.33: "+333",
        5.00: "+400",
        6.00: "+500",
        10.0: "+900",
    }
    tolerance = 0.06

    columnas = [
        ("AvgH",  "home_win",  "local_ap"),
        ("AvgA",  "away_win",  "visitante_ap"),
        ("AvgD",  "draw",      "empate_ap"),
        ("AvgCH", "home_win",  "local_ci"),
        ("AvgCA", "away_win",  "visitante_ci"),
        ("AvgCD", "draw",      "empate_ci"),
    ]

    rows = []
    for odd_target, lbl in targets.items():
        lo      = odd_target - tolerance
        hi      = odd_target + tolerance
        imp_pct = round(1 / odd_target * 100, 2)
        for odd_col, res_col, lado in columnas:
            if odd_col not in df.columns or res_col not in df.columns:
                continue
            mask = (df[odd_col] >= lo) & (df[odd_col] < hi)
            sub  = df[mask]
            n    = len(sub)
            if n < 10:
                continue
            wins     = int(sub[res_col].sum())
            profit   = wins * odd_target - n
            roi_pct  = round(profit / n * 100, 2)
            real_pct = round(wins / n * 100, 2)
            edge     = round(real_pct - imp_pct, 2)
            rows.append({
                "cuota":        lbl,
                "decimal":      odd_target,
                "lado":         lado,
                "n":            n,
                "wins":         wins,
                "real_%":       real_pct,
                "imp_%":        imp_pct,
                "edge_%":       edge,
                "roi_%":        roi_pct,
                "yield":        round(roi_pct / 100, 4),
                "avg_odd_real": round(sub[odd_col].mean(), 3),
            })

    df_out = pd.DataFrame(rows)
    if df_out.empty:
        return df_out
    return df_out.sort_values(["decimal", "lado"]).reset_index(drop=True)


# modelo Poisson para estimar probabilidades de marcador

def poisson_model(df: pd.DataFrame, home_team: str, away_team: str,
                  liga: str = None, max_goals: int = 8) -> dict:
    """Modelo Dixon-Coles simplificado basado en Poisson.

    Calcula lambda_home y lambda_away usando el ataque y defensa
    historicos de cada equipo (todo el periodo o solo su liga).
    Devuelve:
      - lambda_h / lambda_a
      - matriz de probabilidades de marcadores
      - P(H), P(D), P(A)
      - P(btts), P(over1.5), P(over2.5), P(over3.5)
      - top marcadores por probabilidad
      - cuota_fair_H/D/A  (1 / probabilidad — sin margen)
    """
    from scipy.stats import poisson as sp_poisson

    sub = df[df["Div"] == liga].copy() if liga else df.copy()

    # promedio global de goles de la liga (referencia)
    mu_h = sub["FTHG"].mean()
    mu_a = sub["FTAG"].mean()

    # ataque y defensa de cada equipo como local
    def ataque_defensa(equipo):
        as_home = sub[sub["HomeTeam"] == equipo]
        as_away = sub[sub["AwayTeam"] == equipo]
        gf_h = as_home["FTHG"].sum(); gc_h = as_home["FTAG"].sum(); n_h = len(as_home)
        gf_a = as_away["FTAG"].sum(); gc_a = as_away["FTHG"].sum(); n_a = len(as_away)
        n_total = n_h + n_a
        if n_total == 0:
            return None
        gf = gf_h + gf_a; gc = gc_h + gc_a
        pj_h = max(n_h, 1); pj_a = max(n_a, 1)
        atk_h = (gf_h / pj_h) / mu_h if mu_h > 0 else 1.0
        def_h = (gc_h / pj_h) / mu_a if mu_a > 0 else 1.0
        atk_a = (gf_a / pj_a) / mu_a if mu_a > 0 else 1.0
        def_a = (gc_a / pj_a) / mu_h if mu_h > 0 else 1.0
        return {
            "atk_home": atk_h, "def_home": def_h,
            "atk_away": atk_a, "def_away": def_a,
            "pj": n_total, "gf": gf, "gc": gc,
        }

    stats_h = ataque_defensa(home_team)
    stats_a = ataque_defensa(away_team)

    if stats_h is None or stats_a is None:
        return {"error": f"equipo no encontrado en el dataset"}

    # lambdas esperados
    lambda_h = mu_h * stats_h["atk_home"] * stats_a["def_away"]
    lambda_a = mu_a * stats_a["atk_away"] * stats_h["def_home"]

    # matriz de marcadores
    max_g = max_goals
    matriz = np.zeros((max_g + 1, max_g + 1))
    for i in range(max_g + 1):
        for j in range(max_g + 1):
            matriz[i, j] = sp_poisson.pmf(i, lambda_h) * sp_poisson.pmf(j, lambda_a)

    p_home = float(np.sum(np.tril(matriz, -1)))
    p_draw = float(np.sum(np.diag(matriz)))
    p_away = float(np.sum(np.triu(matriz, 1)))

    # mercados de goles
    p_btts   = float(sum(matriz[i,j] for i in range(1,max_g+1) for j in range(1,max_g+1)))
    p_over15 = float(sum(matriz[i,j] for i in range(max_g+1) for j in range(max_g+1) if i+j>1))
    p_over25 = float(sum(matriz[i,j] for i in range(max_g+1) for j in range(max_g+1) if i+j>2))
    p_over35 = float(sum(matriz[i,j] for i in range(max_g+1) for j in range(max_g+1) if i+j>3))

    # top 10 marcadores mas probables
    scores = []
    for i in range(max_g + 1):
        for j in range(max_g + 1):
            scores.append((i, j, round(float(matriz[i, j]) * 100, 2)))
    top_scores = sorted(scores, key=lambda x: -x[2])[:10]

    # cuotas fair (sin margen)
    fair_h = round(1 / p_home, 3) if p_home > 0 else None
    fair_d = round(1 / p_draw, 3) if p_draw > 0 else None
    fair_a = round(1 / p_away, 3) if p_away > 0 else None

    return {
        "home":         home_team,
        "away":         away_team,
        "liga":         liga or "global",
        "lambda_h":     round(lambda_h, 3),
        "lambda_a":     round(lambda_a, 3),
        "mu_h_liga":    round(mu_h, 3),
        "mu_a_liga":    round(mu_a, 3),
        "atk_home":     round(stats_h["atk_home"], 3),
        "def_home":     round(stats_h["def_home"], 3),
        "atk_away":     round(stats_a["atk_away"], 3),
        "def_away":     round(stats_a["def_away"], 3),
        "pj_home":      stats_h["pj"],
        "pj_away":      stats_a["pj"],
        "P_H":          round(p_home * 100, 2),
        "P_D":          round(p_draw * 100, 2),
        "P_A":          round(p_away * 100, 2),
        "P_btts":       round(p_btts * 100, 2),
        "P_over15":     round(p_over15 * 100, 2),
        "P_over25":     round(p_over25 * 100, 2),
        "P_over35":     round(p_over35 * 100, 2),
        "fair_H":       fair_h,
        "fair_D":       fair_d,
        "fair_A":       fair_a,
        "top_scores":   top_scores,
        "matriz":       matriz,
    }


def print_poisson(result: dict):
    """Imprime el resultado del modelo Poisson de forma legible."""
    if "error" in result:
        print(f"  ERROR: {result['error']}")
        return
    print(f"\n  {result['home']} vs {result['away']} ({result['liga']})")
    print(f"  lambda local:    {result['lambda_h']}  (atk {result['atk_home']} x def rival {result['def_away']} x mu {result['mu_h_liga']})")
    print(f"  lambda visitante:{result['lambda_a']}  (atk {result['atk_away']} x def rival {result['def_home']} x mu {result['mu_a_liga']})")
    print(f"  partidos usados: local {result['pj_home']} | visitante {result['pj_away']}")
    print(f"")
    print(f"  P(H)={result['P_H']}%   P(D)={result['P_D']}%   P(A)={result['P_A']}%")
    print(f"  cuota fair: H={result['fair_H']}  D={result['fair_D']}  A={result['fair_A']}")
    print(f"")
    print(f"  P(btts)={result['P_btts']}%  P(over1.5)={result['P_over15']}%  P(over2.5)={result['P_over25']}%  P(over3.5)={result['P_over35']}%")
    print(f"")
    print(f"  marcadores mas probables:")
    for h, a, pct in result["top_scores"]:
        bar = "#" * int(pct / 0.5)
        print(f"    {h}-{a}  {pct:5.2f}%  {bar}")


def poisson_vs_market(result: dict, odd_h: float, odd_d: float, odd_a: float) -> pd.DataFrame:
    """Compara cuotas del mercado con probabilidades del modelo Poisson.
    Calcula edge (modelo - implied) y ROI esperado por resultado."""
    if "error" in result:
        return pd.DataFrame()
    rows = []
    for lado, p_modelo, odd in [
        ("local",     result["P_H"] / 100, odd_h),
        ("empate",    result["P_D"] / 100, odd_d),
        ("visitante", result["P_A"] / 100, odd_a),
    ]:
        imp      = 1 / odd
        edge     = round((p_modelo - imp) * 100, 2)
        ev       = round(odd * p_modelo - 1, 4)
        roi_exp  = round(ev * 100, 2)
        rows.append({
            "lado":       lado,
            "P_modelo_%": round(p_modelo * 100, 2),
            "imp_%":      round(imp * 100, 2),
            "edge_%":     edge,
            "cuota_mkt":  odd,
            "cuota_fair": round(1 / p_modelo, 3) if p_modelo > 0 else None,
            "EV_1u":      ev,
            "roi_exp_%":  roi_exp,
            "value_bet":  "SI" if edge > 0 else "no",
        })
    return pd.DataFrame(rows)


# rendimiento real por rango de cuota, apertura vs cierre

def stats_odds_range_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Por cada rango de cuota (apertura y cierre), que % gana realmente cada resultado."""
    bins   = [1.0,1.3,1.5,1.75,2.0,2.25,2.5,3.0,3.5,4.0,5.0,7.0,25.0]
    labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]
    rows=[]
    for ap_col, ci_col, res_col, lbl in [
        ("AvgH","AvgCH","home_win","local"),
        ("AvgA","AvgCA","away_win","visitante"),
    ]:
        d2=df.copy()
        d2["rng_ap"]=pd.cut(d2[ap_col],bins=bins,labels=labels,right=False)
        d2["rng_ci"]=pd.cut(d2[ci_col],bins=bins,labels=labels,right=False)
        for rng in labels:
            sub_ap=d2[d2["rng_ap"]==rng]; sub_ci=d2[d2["rng_ci"]==rng]
            mid=(bins[labels.index(rng)]+bins[labels.index(rng)+1])/2
            imp=round(1/mid*100,2)
            if len(sub_ap)>=10:
                rows.append({
                    "resultado":lbl,"tipo":"apertura","rango_odd":rng,"n":len(sub_ap),
                    "imp_%":imp,
                    "real_%":round(sub_ap[res_col].mean()*100,2),
                    "edge":round(sub_ap[res_col].mean()*100-imp,2),
                    "avg_goles":round(sub_ap["total_goals"].mean(),3),
                    "pct_btts":round(sub_ap["btts"].mean()*100,2),
                })
            if len(sub_ci)>=10:
                rows.append({
                    "resultado":lbl,"tipo":"cierre","rango_odd":rng,"n":len(sub_ci),
                    "imp_%":imp,
                    "real_%":round(sub_ci[res_col].mean()*100,2),
                    "edge":round(sub_ci[res_col].mean()*100-imp,2),
                    "avg_goles":round(sub_ci["total_goals"].mean(),3),
                    "pct_btts":round(sub_ci["btts"].mean()*100,2),
                })
    return pd.DataFrame(rows).sort_values(["resultado","tipo","rango_odd"]).reset_index(drop=True)


# funciones de analisis profundo: primer tiempo, btts, cuotas y equipos

def stats_ht1_effect(df: pd.DataFrame) -> dict:
    """Si hay exactamente 1 gol en el HT, ¿qué pasa en el ST?
    Compara HT=0, HT=1, HT=2+ en términos de ST, FT y flags."""
    total = len(df)
    rows = []
    for lbl, mask in [
        ("HT=0",  df["ht_goals"] == 0),
        ("HT=1",  df["ht_goals"] == 1),
        ("HT=2",  df["ht_goals"] == 2),
        ("HT=3+", df["ht_goals"] >= 3),
    ]:
        sub = df[mask]
        if len(sub) == 0:
            continue
        rows.append({
            "ht_grupo":         lbl,
            "partidos":         len(sub),
            "pct_total":        round(len(sub) / total * 100, 2),
            "avg_st":           round(sub["second_half_goals"].mean(), 3),
            "moda_st":          int(sub["second_half_goals"].mode()[0]) if not sub["second_half_goals"].mode().empty else None,
            "avg_ft":           round(sub["total_goals"].mean(), 3),
            "pct_st_mayor_ht":  round((sub["second_half_goals"] > sub["ht_goals"]).mean() * 100, 2),
            "pct_st_igual_ht":  round((sub["second_half_goals"] == sub["ht_goals"]).mean() * 100, 2),
            "pct_st_menor_ht":  round((sub["second_half_goals"] < sub["ht_goals"]).mean() * 100, 2),
            "pct_btts":         round(sub["btts"].mean() * 100, 2),
            "pct_over25":       round(sub["over25"].mean() * 100, 2),
            "pct_goalless_ft":  round((sub["total_goals"] == 0).mean() * 100, 2),
            "pct_H":            round(sub["home_win"].mean() * 100, 2),
            "pct_D":            round(sub["draw"].mean() * 100, 2),
            "pct_A":            round(sub["away_win"].mean() * 100, 2),
        })
    return pd.DataFrame(rows)


def stats_ht_calma(df: pd.DataFrame) -> pd.DataFrame:
    """¿Si el HT tiene muchos goles, el ST se calma?
    Correlación HT vs ST y tabla de avg_st por valor de ht_goals."""
    rows = []
    for ht_val in range(0, int(df["ht_goals"].max()) + 1):
        sub = df[df["ht_goals"] == ht_val]
        if len(sub) < 5:
            continue
        rows.append({
            "ht_goles":       ht_val,
            "partidos":       len(sub),
            "avg_st":         round(sub["second_half_goals"].mean(), 3),
            "moda_st":        int(sub["second_half_goals"].mode()[0]) if not sub["second_half_goals"].mode().empty else None,
            "avg_ft":         round(sub["total_goals"].mean(), 3),
            "pct_st0":        round((sub["second_half_goals"] == 0).mean() * 100, 2),
            "pct_st1":        round((sub["second_half_goals"] == 1).mean() * 100, 2),
            "pct_st2p":       round((sub["second_half_goals"] >= 2).mean() * 100, 2),
            "pct_st_mayor_ht":round((sub["second_half_goals"] > sub["ht_goals"]).mean() * 100, 2),
            "pct_over25":     round(sub["over25"].mean() * 100, 2),
            "pct_btts":       round(sub["btts"].mean() * 100, 2),
        })
    corr = round(df["ht_goals"].corr(df["second_half_goals"]), 4)
    return pd.DataFrame(rows), corr


def stats_ht_scorers_top(df: pd.DataFrame, top_n: int = 15) -> dict:
    """Top equipos que más goles marcan en HT y top que más reciben en HT."""
    # goles marcados en HT como local (HTHG) + como visitante (HTAG)
    hthg = df.groupby("HomeTeam")["HTHG"].sum()
    htag = df.groupby("AwayTeam")["HTAG"].sum()
    total_ht_gf = hthg.add(htag, fill_value=0).sort_values(ascending=False)

    # goles recibidos en HT como local (HTAG) + como visitante (HTHG)
    htgc_h = df.groupby("HomeTeam")["HTAG"].sum()
    htgc_a = df.groupby("AwayTeam")["HTHG"].sum()
    total_ht_gc = htgc_h.add(htgc_a, fill_value=0).sort_values(ascending=False)

    # ratio HT/FT goles marcados por equipo
    fthg = df.groupby("HomeTeam")["FTHG"].sum()
    ftag = df.groupby("AwayTeam")["FTAG"].sum()
    total_ft_gf = fthg.add(ftag, fill_value=0)
    ratio_ht_ft = (total_ht_gf / total_ft_gf.reindex(total_ht_gf.index)).dropna().sort_values(ascending=False)

    return {
        "top_marcadores_ht":    {k: int(v) for k, v in total_ht_gf.head(top_n).items()},
        "top_reciben_ht":       {k: int(v) for k, v in total_ht_gc.head(top_n).items()},
        "ratio_ht_ft_marcados": {k: round(float(v), 3) for k, v in ratio_ht_ft.head(top_n).items()},
    }


def stats_ht_goal_prob(df: pd.DataFrame) -> dict:
    """Probabilidad de goles en el primer tiempo."""
    total = len(df)
    return {
        "P(HT>=1)":    round((df["ht_goals"] >= 1).mean() * 100, 2),
        "P(HT>=2)":    round((df["ht_goals"] >= 2).mean() * 100, 2),
        "P(HT>=3)":    round((df["ht_goals"] >= 3).mean() * 100, 2),
        "P(HT=0)":     round((df["ht_goals"] == 0).mean() * 100, 2),
        "P(HT=1)":     round((df["ht_goals"] == 1).mean() * 100, 2),
        "P(HT=2)":     round((df["ht_goals"] == 2).mean() * 100, 2),
        "P(HT=0→FT=0)":round(((df["ht_goals"] == 0) & (df["total_goals"] == 0)).mean() * 100, 2),
        "P(FT=0|HT=0)":round(
            ((df["ht_goals"] == 0) & (df["total_goals"] == 0)).sum() /
            max((df["ht_goals"] == 0).sum(), 1) * 100, 2),
        "P(FT>=3|HT=0)": round(
            ((df["ht_goals"] == 0) & (df["total_goals"] >= 3)).sum() /
            max((df["ht_goals"] == 0).sum(), 1) * 100, 2),
        "avg_ht_goles_loc": round(df["HTHG"].mean(), 3),
        "avg_ht_goles_vis": round(df["HTAG"].mean(), 3),
        "pct_local_marca_HT":  round((df["HTHG"] > 0).mean() * 100, 2),
        "pct_visita_marca_HT": round((df["HTAG"] > 0).mean() * 100, 2),
        "pct_btts_HT":         round(((df["HTHG"] > 0) & (df["HTAG"] > 0)).mean() * 100, 2),
    }


def stats_scoreline_00ht_explosion(df: pd.DataFrame) -> dict:
    """Cuando va 0-0 al HT, ¿cuántos goles hay en el ST?
    Distribución del resultado FT partiendo de 0-0 al descanso."""
    sub = df[df["ht_goals"] == 0]
    if len(sub) == 0:
        return {}
    total_ht0 = len(sub)
    sub2 = sub.copy()
    sub2["score_ft"] = sub2["FTHG"].astype(str) + "-" + sub2["FTAG"].astype(str)
    top_scores = sub2["score_ft"].value_counts().head(10)
    return {
        "partidos_ht0":       total_ht0,
        "pct_del_total":      round(total_ht0 / len(df) * 100, 2),
        "pct_ft_0_0":         round((sub["total_goals"] == 0).mean() * 100, 2),
        "pct_ft_1gol":        round((sub["total_goals"] == 1).mean() * 100, 2),
        "pct_ft_2goles":      round((sub["total_goals"] == 2).mean() * 100, 2),
        "pct_ft_3mas":        round((sub["total_goals"] >= 3).mean() * 100, 2),
        "avg_goles_ft":       round(sub["total_goals"].mean(), 3),
        "avg_st_goles":       round(sub["second_half_goals"].mean(), 3),
        "pct_btts_desde_ht0": round(sub["btts"].mean() * 100, 2),
        "pct_over25_ht0":     round(sub["over25"].mean() * 100, 2),
        "pct_H":              round(sub["home_win"].mean() * 100, 2),
        "pct_D":              round(sub["draw"].mean() * 100, 2),
        "pct_A":              round(sub["away_win"].mean() * 100, 2),
        "top_marcadores_ft":  dict(zip(top_scores.index.tolist(), top_scores.values.tolist())),
    }


def stats_btts_by_day(df: pd.DataFrame) -> pd.DataFrame:
    """% BTTS por día de semana y correlación entre días consecutivos."""
    agg = df.groupby("day_name").agg(
        partidos    = ("btts", "count"),
        pct_btts    = ("btts", "mean"),
        pct_over25  = ("over25", "mean"),
        avg_goles   = ("total_goals", "mean"),
        pct_goalless= ("goalless", "mean"),
        pct_H       = ("home_win", "mean"),
        pct_D       = ("draw", "mean"),
    ).round(3)
    for c in ["pct_btts", "pct_over25", "pct_goalless", "pct_H", "pct_D"]:
        agg[c] = round(agg[c] * 100, 2)
    return agg.reindex([d for d in DAY_ORDER if d in agg.index]).reset_index()


def stats_btts_finde_vs_entresemana(df: pd.DataFrame) -> dict:
    """Finde vs entresemana en BTTS y goles."""
    finde = df[df["is_weekend"] == 1]
    entre = df[df["is_weekend"] == 0]
    return {
        "finde_n":        len(finde),
        "finde_btts":     round(finde["btts"].mean() * 100, 2),
        "finde_over25":   round(finde["over25"].mean() * 100, 2),
        "finde_avg_goles":round(finde["total_goals"].mean(), 3),
        "entre_n":        len(entre),
        "entre_btts":     round(entre["btts"].mean() * 100, 2),
        "entre_over25":   round(entre["over25"].mean() * 100, 2),
        "entre_avg_goles":round(entre["total_goals"].mean(), 3),
        "diff_btts":      round((finde["btts"].mean() - entre["btts"].mean()) * 100, 2),
    }


def stats_btts_team_ht(df: pd.DataFrame, top_n: int = 15) -> tuple:
    """Equipos donde más se da BTTS en el primer tiempo (ambos marcan antes del descanso)
    y equipos donde más se da BTTS solo en el segundo tiempo."""
    df2 = df.copy()
    df2["btts_ht"] = ((df2["HTHG"] > 0) & (df2["HTAG"] > 0)).astype(int)
    df2["btts_st_only"] = ((df2["btts"] == 1) & (df2["btts_ht"] == 0)).astype(int)

    equipos = set(df2["HomeTeam"].unique()) | set(df2["AwayTeam"].unique())
    rows_ht = []
    rows_st = []
    for team in equipos:
        sub = df2[(df2["HomeTeam"] == team) | (df2["AwayTeam"] == team)]
        if len(sub) < 15:
            continue
        rows_ht.append({
            "equipo":       team,
            "pj":           len(sub),
            "pct_btts_ht":  round(sub["btts_ht"].mean() * 100, 2),
            "pct_btts_ft":  round(sub["btts"].mean() * 100, 2),
            "avg_goles_ht": round(sub["ht_goals"].mean(), 3),
        })
        rows_st.append({
            "equipo":           team,
            "pj":               len(sub),
            "pct_btts_st_only": round(sub["btts_st_only"].mean() * 100, 2),
            "pct_btts_ft":      round(sub["btts"].mean() * 100, 2),
            "avg_goles_st":     round(sub["second_half_goals"].mean(), 3),
        })

    top_ht = pd.DataFrame(rows_ht).sort_values("pct_btts_ht", ascending=False).head(top_n).reset_index(drop=True)
    top_st = pd.DataFrame(rows_st).sort_values("pct_btts_st_only", ascending=False).head(top_n).reset_index(drop=True)
    return top_ht, top_st


def stats_odds_movement_detail(df: pd.DataFrame) -> pd.DataFrame:
    """Si la cuota baja antes del cierre, ¿es más certero el resultado?
    Clasifica por rango de movimiento con granularidad alta."""
    df2 = df.copy()
    bins  = [-np.inf, -0.20, -0.10, -0.05, 0.05, 0.10, 0.20, np.inf]
    labels = ["baja >0.20", "baja 0.10-0.20", "baja 0.05-0.10",
              "estable ±0.05",
              "sube 0.05-0.10", "sube 0.10-0.20", "sube >0.20"]
    df2["mov_cat"] = pd.cut(df2["odds_move_H"], bins=bins, labels=labels)
    rows = []
    for cat in labels:
        sub = df2[df2["mov_cat"] == cat]
        if len(sub) < 5:
            continue
        rows.append({
            "movimiento":    cat,
            "partidos":      len(sub),
            "pct_total":     round(len(sub) / len(df) * 100, 2),
            "pct_H":         round(sub["home_win"].mean() * 100, 2),
            "pct_D":         round(sub["draw"].mean() * 100, 2),
            "pct_A":         round(sub["away_win"].mean() * 100, 2),
            "avg_goles":     round(sub["total_goals"].mean(), 3),
            "pct_btts":      round(sub["btts"].mean() * 100, 2),
            "pct_over25":    round(sub["over25"].mean() * 100, 2),
            "avg_mov":       round(sub["odds_move_H"].mean(), 4),
        })
    return pd.DataFrame(rows)


def stats_odds_mode_full(df: pd.DataFrame) -> pd.DataFrame:
    """Top 10 modas de cuotas (apertura y cierre) más frecuentes para cada resultado.
    Responde: ¿qué cuota 'redonda' aparece más seguido?"""
    rows = []
    for resultado, ftr, odd_ap, odd_ci in [
        ("local",    "H", "AvgH",  "AvgCH"),
        ("visitante","A", "AvgA",  "AvgCA"),
        ("empate",   "D", "AvgD",  "AvgCD"),
    ]:
        sub = df[df["FTR"] == ftr]
        for tipo, col in [("apertura", odd_ap), ("cierre", odd_ci)]:
            top = sub[col].round(2).value_counts().head(10)
            for cuota, cnt in top.items():
                rows.append({
                    "resultado": resultado,
                    "tipo":      tipo,
                    "cuota":     float(cuota),
                    "frecuencia":int(cnt),
                    "pct":       round(cnt / len(sub) * 100, 2),
                })
    return pd.DataFrame(rows)


def stats_odds_mode_all_results(df: pd.DataFrame) -> pd.DataFrame:
    """Moda de cuotas para todos los partidos (sin filtrar por resultado).
    ¿Cuál es la cuota local más común que sale en el tablero?"""
    rows = []
    for nombre, col in [("AvgH", "local_ap"), ("AvgCH", "local_ci"),
                         ("AvgA", "visita_ap"), ("AvgCA", "visita_ci"),
                         ("AvgD", "empate_ap"), ("AvgCD", "empate_ci")]:
        top = df[nombre].round(2).value_counts().head(10)
        for cuota, cnt in top.items():
            rows.append({
                "cuota_tipo": col,
                "cuota":      float(cuota),
                "frecuencia": int(cnt),
                "pct":        round(cnt / len(df) * 100, 2),
            })
    return pd.DataFrame(rows)


def stats_clean_sheet_teams(df: pd.DataFrame, top_n: int = 15) -> dict:
    """Top equipos por % de clean sheets como local y como visitante."""
    # como local: no reciben gol (FTAG=0)
    cs_loc = df.groupby("HomeTeam").agg(
        pj=("FTAG", "count"),
        cs=("clean_sheet_h", "sum"),
    ).reset_index()
    cs_loc = cs_loc[cs_loc["pj"] >= 20]
    cs_loc["pct_cs"] = round(cs_loc["cs"] / cs_loc["pj"] * 100, 2)
    cs_loc = cs_loc.sort_values("pct_cs", ascending=False).head(top_n)

    # como visitante: no reciben gol (FTHG=0)
    cs_vis = df.groupby("AwayTeam").agg(
        pj=("FTHG", "count"),
        cs=("clean_sheet_a", "sum"),
    ).reset_index().rename(columns={"AwayTeam": "AwayTeam"})
    cs_vis = cs_vis[cs_vis["pj"] >= 20]
    cs_vis["pct_cs"] = round(cs_vis["cs"] / cs_vis["pj"] * 100, 2)
    cs_vis = cs_vis.sort_values("pct_cs", ascending=False).head(top_n)

    return {
        "top_cs_local":    cs_loc[["HomeTeam", "pj", "cs", "pct_cs"]].rename(columns={"HomeTeam": "equipo"}),
        "top_cs_visitante":cs_vis[["AwayTeam",  "pj", "cs", "pct_cs"]].rename(columns={"AwayTeam":  "equipo"}),
    }


def stats_late_season_goals(df: pd.DataFrame) -> pd.DataFrame:
    """Meses de inicio (ago-oct) vs final (mar-may): ¿más abiertos al final?"""
    mes_inicio = [8, 9, 10]
    mes_final  = [3, 4, 5]
    rows = []
    for lbl, meses in [("inicio (ago-oct)", mes_inicio), ("medio (nov-feb)", [11, 12, 1, 2]),
                        ("final (mar-may)", mes_final)]:
        sub = df[df["month"].isin(meses)]
        if len(sub) == 0:
            continue
        rows.append({
            "periodo":    lbl,
            "partidos":   len(sub),
            "avg_goles":  round(sub["total_goals"].mean(), 3),
            "avg_ht":     round(sub["ht_goals"].mean(), 3),
            "avg_st":     round(sub["second_half_goals"].mean(), 3),
            "pct_btts":   round(sub["btts"].mean() * 100, 2),
            "pct_over25": round(sub["over25"].mean() * 100, 2),
            "pct_H":      round(sub["home_win"].mean() * 100, 2),
            "pct_D":      round(sub["draw"].mean() * 100, 2),
            "pct_goalless":round(sub["goalless"].mean() * 100, 2),
        })
    return pd.DataFrame(rows)


def stats_team_variance_goals(df: pd.DataFrame, top_n: int = 10) -> dict:
    """Equipos más consistentes vs más impredecibles (varianza de goles marcados)."""
    equipos = set(df["HomeTeam"].unique()) | set(df["AwayTeam"].unique())
    rows = []
    for team in equipos:
        loc = df[df["HomeTeam"] == team]["FTHG"]
        vis = df[df["AwayTeam"] == team]["FTAG"]
        goles = pd.concat([loc, vis])
        if len(goles) < 20:
            continue
        rows.append({
            "equipo":    team,
            "pj":        len(goles),
            "avg_gf":    round(goles.mean(), 3),
            "std_gf":    round(goles.std(), 3),
            "var_gf":    round(goles.var(), 3),
            "cv_gf":     round(goles.std() / goles.mean() * 100, 2) if goles.mean() > 0 else None,
            "max_gf":    int(goles.max()),
            "pct_0goles":round((goles == 0).mean() * 100, 2),
        })
    df_out = pd.DataFrame(rows)
    return {
        "mas_consistentes":    df_out.sort_values("std_gf").head(top_n).reset_index(drop=True),
        "mas_impredecibles":   df_out.sort_values("std_gf", ascending=False).head(top_n).reset_index(drop=True),
    }


def stats_scoreline_1_0_ht_extended(df: pd.DataFrame) -> dict:
    """Marcador 1-0 al HT: qué pasa en el FT (ya existe básica, esta añade más granularidad)."""
    sub = df[(df["HTHG"] == 1) & (df["HTAG"] == 0)]
    if len(sub) == 0:
        return {}
    return {
        "total":            int(len(sub)),
        "pct_1_0_ft":       round(((sub["FTHG"]==1)&(sub["FTAG"]==0)).mean()*100, 2),
        "pct_2_0_ft":       round(((sub["FTHG"]==2)&(sub["FTAG"]==0)).mean()*100, 2),
        "pct_3_0_ft":       round(((sub["FTHG"]==3)&(sub["FTAG"]==0)).mean()*100, 2),
        "pct_1_1_ft":       round(((sub["FTHG"]==1)&(sub["FTAG"]==1)).mean()*100, 2),
        "pct_2_1_ft":       round(((sub["FTHG"]==2)&(sub["FTAG"]==1)).mean()*100, 2),
        "pct_remonta_A":    round((sub["FTR"]=="A").mean()*100, 2),
        "pct_empata":       round((sub["FTR"]=="D").mean()*100, 2),
        "pct_mantiene_H":   round((sub["FTR"]=="H").mean()*100, 2),
        "avg_goles_ft":     round(sub["total_goals"].mean(), 3),
        "pct_btts":         round(sub["btts"].mean()*100, 2),
        "pct_over25":       round(sub["over25"].mean()*100, 2),
    }


def stats_scoreline_0_0_ht_deep(df: pd.DataFrame) -> pd.DataFrame:
    """Distribución completa del FT cuando va 0-0 al HT, por liga."""
    ht0 = df[df["ht_goals"] == 0].copy()
    ht0["score_ft"] = ht0["FTHG"].astype(str) + "-" + ht0["FTAG"].astype(str)
    rows = []
    for liga in sorted(df["Div"].unique()):
        sub = ht0[ht0["Div"] == liga]
        if len(sub) == 0:
            continue
        top5 = sub["score_ft"].value_counts().head(5)
        rows.append({
            "liga":          liga,
            "partidos_ht0":  len(sub),
            "pct_ft_0_0":    round((sub["total_goals"]==0).mean()*100, 2),
            "pct_ft_1gol":   round((sub["total_goals"]==1).mean()*100, 2),
            "pct_ft_2goles": round((sub["total_goals"]==2).mean()*100, 2),
            "pct_ft_3mas":   round((sub["total_goals"]>=3).mean()*100, 2),
            "avg_goles_ft":  round(sub["total_goals"].mean(), 3),
            "pct_btts":      round(sub["btts"].mean()*100, 2),
            "top1_marcador": top5.index[0] if len(top5)>0 else None,
            "top1_pct":      round(top5.iloc[0]/len(sub)*100,2) if len(top5)>0 else None,
        })
    return pd.DataFrame(rows)


def draw_er_diagram(fp):
    fig,ax=plt.subplots(figsize=(14,9)); ax.set_xlim(0,14); ax.set_ylim(0,9); ax.axis("off")
    def entity(ax,x,y,title,fields,w=2.2,h=2.6):
        ax.add_patch(mpatches.FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle="round,pad=0.1",
            linewidth=2,edgecolor="#2c3e50",facecolor="#d6eaf8"))
        ax.text(x,y+h/2-0.25,title,ha="center",va="center",fontsize=10,fontweight="bold",color="#2c3e50")
        ax.plot([x-w/2,x+w/2],[y+h/2-0.45,y+h/2-0.45],color="#2c3e50",linewidth=1)
        for i,f in enumerate(fields):
            ax.text(x,y+h/2-0.7-i*0.32,f,ha="center",va="center",fontsize=7.5,color="#1a252f")
    def rel(ax,x,y,lbl,w=1.3,h=0.55):
        ax.add_patch(plt.Polygon([[x,y+h],[x+w/2,y],[x,y-h],[x-w/2,y]],
            closed=True,linewidth=1.5,edgecolor="#2c3e50",facecolor="#fef9e7"))
        ax.text(x,y,lbl,ha="center",va="center",fontsize=8,color="#2c3e50")
    def arr(ax,x1,y1,x2,y2,lbl=""):
        ax.annotate("",xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle="-|>",color="#2c3e50",lw=1.5))
        if lbl: ax.text((x1+x2)/2+0.1,(y1+y2)/2+0.1,lbl,fontsize=7,color="#555")
    entity(ax,7,4.5,"Match",["Date","FTHG/FTAG","FTR","HTHG/HTAG","HTR"])
    entity(ax,2,7,"Team",["HomeTeam / AwayTeam"])
    entity(ax,12,7,"Odds",["B365H/D/A","MaxH/D/A","AvgH/D/A","B365CH/D/A","MaxCH/D/A","AvgCH/D/A"])
    entity(ax,2,2,"League",["Div (E0/SP1/D1/I1/F1)"])
    entity(ax,12,2,"Season",["Season (1920..2526)"])
    rel(ax,4.3,6.1,"plays"); rel(ax,9.7,6.1,"has odds")
    rel(ax,4.3,3.2,"belongs to"); rel(ax,9.7,3.2,"played in")
    arr(ax,2,6.7,3.5,6.3); arr(ax,5.1,6.1,5.9,5.1)
    arr(ax,12,6.7,10.5,6.3); arr(ax,8.8,5.5,10.3,6.1,"1:1")
    arr(ax,2,2.7,3.5,3.1); arr(ax,5.1,3.2,5.9,3.9,"N:1")
    arr(ax,12,2.7,10.5,3.1); arr(ax,8.8,3.9,10.3,3.2,"N:1")
    for x,y,t in [(3,6.6,"N"),(5.5,5.6,"1"),(11,6.6,"1"),(3,2.9,"N"),(11,2.9,"N")]:
        ax.text(x,y,t,fontsize=8,color="#e74c3c",fontweight="bold")
    ax.set_title("Diagrama Entidad-Relacion - European Football Dataset",fontsize=13,fontweight="bold",pad=15)
    plt.tight_layout(); plt.savefig(fp,dpi=150,bbox_inches="tight"); plt.close()
    print(f"  ER -> {fp}")

def scatter_group_by(fp,df,x_col,y_col,label_col):
    fig,ax=plt.subplots(figsize=(10,6))
    labels=pd.unique(df[label_col]); cmap=get_cmap(len(labels)+1)
    for i,lbl in enumerate(labels):
        sub=df[df[label_col]==lbl]
        ax.scatter(sub[x_col],sub[y_col],label=lbl,color=cmap(i),alpha=0.5,s=10)
    ax.set_xlabel(x_col); ax.set_ylabel(y_col); ax.legend()
    plt.savefig(fp); plt.close()

# carga del dataset y calculo de variables derivadas

os.makedirs("img",exist_ok=True)
df=pd.read_csv("../Practica 1/data/clean/football_clean.csv",parse_dates=["Date"])
pd.set_option("display.float_format",lambda x:f"{x:.3f}")

df["total_goals"]       = df["FTHG"]+df["FTAG"]
df["ht_goals"]          = df["HTHG"]+df["HTAG"]
df["second_half_goals"] = df["total_goals"]-df["ht_goals"]
df["goal_diff"]         = df["FTHG"]-df["FTAG"]
df["home_win"]          = (df["FTR"]=="H").astype(int)
df["draw"]              = (df["FTR"]=="D").astype(int)
df["away_win"]          = (df["FTR"]=="A").astype(int)
df["btts"]              = ((df["FTHG"]>0)&(df["FTAG"]>0)).astype(int)
df["over15"]            = (df["total_goals"]>1).astype(int)
df["over25"]            = (df["total_goals"]>2).astype(int)
df["over35"]            = (df["total_goals"]>3).astype(int)
df["over45"]            = (df["total_goals"]>4).astype(int)
df["under15"]           = (df["total_goals"]<=1).astype(int)
df["under25"]           = (df["total_goals"]<=2).astype(int)
df["under35"]           = (df["total_goals"]<=3).astype(int)
df["clean_sheet_h"]     = (df["FTAG"]==0).astype(int)
df["clean_sheet_a"]     = (df["FTHG"]==0).astype(int)
df["high_scoring"]      = (df["total_goals"]>=5).astype(int)
df["goalless"]          = (df["total_goals"]==0).astype(int)
df["ht_goalless"]       = (df["ht_goals"]==0).astype(int)
df["ht_high"]           = (df["ht_goals"]>=3).astype(int)
df["imp_prob_H"]        = round(1/df["AvgH"],4)
df["imp_prob_D"]        = round(1/df["AvgD"],4)
df["imp_prob_A"]        = round(1/df["AvgA"],4)
df["overround"]         = round(df["imp_prob_H"]+df["imp_prob_D"]+df["imp_prob_A"],4)
df["odds_move_H"]       = round(df["AvgCH"]-df["AvgH"],4)
df["odds_move_A"]       = round(df["AvgCA"]-df["AvgA"],4)
df["odds_move_D"]       = round(df["AvgCD"]-df["AvgD"],4)
df["is_underdog_away"]  = (df["AvgA"]>4).astype(int)
df["is_underdog_home"]  = (df["AvgH"]>4).astype(int)
df["month"]             = df["Date"].dt.month
df["dayofweek"]         = df["Date"].dt.dayofweek
df["day_name"]          = df["dayofweek"].map(DAY_NAMES)
df["is_weekend"]        = df["dayofweek"].isin([5,6]).astype(int)
df["day_type"]          = df["dayofweek"].map(
    lambda d:"Finde" if d in [5,6] else "Viernes" if d==4 else "Entresemana")
df["Season_label"] = df["Season"].map({
    1920:"2019/20",2021:"2020/21",2122:"2021/22",
    2223:"2022/23",2324:"2023/24",2425:"2024/25",2526:"2025/26"})

goal_flags=["btts","over15","over25","over35","over45","under15","under25","under35",
            "clean_sheet_h","clean_sheet_a","high_scoring","goalless"]
ligas=sorted(df["Div"].unique())
temporadas=sorted(df["Season"].unique())
season_labels={1920:"2019/20",2021:"2020/21",2122:"2021/22",
               2223:"2022/23",2324:"2023/24",2425:"2024/25",2526:"2025/26"}

resumen_temporadas=[]; resumen_ligas_temp=[]

# ciclo principal: itera por temporada y dentro de cada una por liga

for temporada in temporadas:
    slabel=season_labels.get(temporada,str(temporada))
    df_t=df[df["Season"]==temporada]

    print(f"\n{'='*70}")
    print(f"  TEMPORADA {slabel}  ({len(df_t)} partidos)")
    print(f"{'='*70}")

    # --- estadisticas numericas basicas ---
    print(f"\n--- estadisticas numericas {slabel} ---")
    print_tabulate(describe_numeric(df_t,["FTHG","FTAG","HTHG","HTAG",
        "total_goals","ht_goals","second_half_goals","goal_diff"]))

    # --- moda vs media de goles ---
    print(f"\n--- moda vs media de goles {slabel} ---")
    print_tabulate(stats_goals_mode_table(df_t))

    g_t=stats_goals(df_t)
    print(f"  ratio HT/ST: {g_t['ratio_ht_st']} | corr HT->FT: {g_t['corr_ht_ft']} | corr HT->ST: {g_t['corr_ht_st']}")

    # --- analisis HT vs FT ---
    print(f"\n--- analisis primer tiempo vs segundo tiempo {slabel} ---")
    print_tabulate(stats_halftime_analysis(df_t))

    # --- partidos 5+ goles ---
    hs_t=stats_high_scoring_analysis(df_t)
    if hs_t:
        print(f"\n--- partidos 5+ goles {slabel} ---")
        for k,v in hs_t.items(): print(f"  {k}: {v}")

    # --- over vs under: media de goles ---
    print(f"\n--- over vs under: media de goles {slabel} ---")
    print_tabulate(stats_over_under_goals(df_t))

    # --- btts vs no btts ---
    print(f"\n--- btts vs no btts {slabel} ---")
    print_tabulate(stats_btts_profile(df_t))

    # --- resultados ---
    print(f"\n--- resultados FT {slabel} ---")
    print_tabulate(describe_categorical(df_t,"FTR"))
    print(f"\n--- resultados HT {slabel} ---")
    print_tabulate(describe_categorical(df_t,"HTR"))

    # --- flags ---
    print(f"\n--- btts / over / under / flags {slabel} ---")
    for flag in goal_flags:
        print(f"  {flag}: {int(df_t[flag].sum())} ({round(df_t[flag].mean()*100,2)}%)")

    # --- remontadas ---
    cb=stats_comeback(df_t)
    print(f"\n--- remontadas {slabel} ---")
    print(f"  local remonta: {cb['remontada_loc']} ({cb['pct_remontada_loc']}%)")
    print(f"  visit remonta: {cb['remontada_vis']} ({cb['pct_remontada_vis']}%)")
    print(f"  gana HT pierde FT: {cb['ht_win_ft_lose']} ({cb['pct_ht_win_ft_lose']}%)")
    print(f"  empate HT -> H: {cb['ht_draw_ft_win_h']}  empate HT -> A: {cb['ht_draw_ft_win_a']}")

    # --- odds ---
    print(f"\n--- odds {slabel} ---")
    print_tabulate(describe_numeric(df_t,["AvgH","AvgD","AvgA","AvgCH","AvgCD","AvgCA","overround"]))

    print(f"\n--- movimiento de mercado {slabel} ---")
    print_tabulate(describe_numeric(df_t,["odds_move_H","odds_move_D","odds_move_A"]))

    # --- moda y mediana cuotas ganadoras ---
    om=stats_odds_mode(df_t)
    print(f"\n--- moda y mediana cuotas ganadoras {slabel} ---")
    print(f"  local  | ap moda {om['moda_ap_H']} (x{om['cnt_ap_H']}) med {om['med_ap_H']} | ci moda {om['moda_ci_H']} med {om['med_ci_H']}")
    print(f"  visita | ap moda {om['moda_ap_A']} (x{om['cnt_ap_A']}) med {om['med_ap_A']} | ci moda {om['moda_ci_A']} med {om['med_ci_A']}")
    print(f"  empate | ap moda {om['moda_ap_D']} med {om['med_ap_D']}")
    print(f"  ud vis | ap moda {om['moda_ud_ap_A']} (x{om['cnt_ud_ap_A']}) med {om['med_ud_ap_A']}")

    # --- equilibrio de cuotas ---
    print(f"\n--- equilibrio de cuotas H vs A {slabel} ---")
    print_tabulate(stats_odds_gap(df_t))

    # --- implied vs real ---
    iv=stats_implied_vs_real(df_t)
    print(f"\n--- implied vs real {slabel} ---")
    print(f"  local:  implied {iv['imp_H']}% | real {iv['real_H']}% | diff {iv['diff_H']}%")
    print(f"  visita: implied {iv['imp_A']}% | real {iv['real_A']}% | diff {iv['diff_A']}%")
    print(f"  empate: implied {iv['imp_D']}% | real {iv['real_D']}% | diff {iv['diff_D']}%")

    # --- underdogs ---
    ud_t=stats_underdogs(df_t)
    print(f"\n--- underdogs {slabel} ---")
    print(f"  visit (>4): {ud_t['ud_away_total']} | ganan {ud_t['ud_away_wins']} ({ud_t['ud_away_pct']}%) avg odd {ud_t['ud_away_avgOdd']}")
    print(f"  local (>4): {ud_t['ud_home_total']} | ganan {ud_t['ud_home_wins']} ({ud_t['ud_home_pct']}%) avg odd {ud_t['ud_home_avgOdd']}")
    print(f"  extremos (>8): {ud_t['ud_ext_total']} | ganan {ud_t['ud_ext_wins']} ({ud_t['ud_ext_pct']}%)")

    ude_t=df_t[(df_t["AvgA"]>8)&(df_t["away_win"]==1)]
    if len(ude_t):
        print(f"\n--- underdogs extremos ganadores {slabel} ---")
        print_tabulate(ude_t[["Div","Date","HomeTeam","AwayTeam","AvgH","AvgA","FTHG","FTAG"]].sort_values("AvgA",ascending=False))

    # --- smart money ---
    sm_t=stats_smart_money(df_t)
    print(f"\n--- smart money {slabel} ---")
    print(f"  local:  {sm_t['sm_loc_n']} | gana {sm_t['sm_loc_pct']}% | avg goles {sm_t['sm_loc_goles']}")
    print(f"  visita: {sm_t['sm_vis_n']} | gana {sm_t['sm_vis_pct']}% | avg goles {sm_t['sm_vis_goles']}")
    print(f"  empate: {sm_t['sm_draw_n']} | empata {sm_t['sm_draw_pct']}%")

    # --- efecto racha ---
    print(f"\n--- efecto racha equipo local {slabel} ---")
    print_tabulate(stats_streak_effect(df_t))

    # --- dias de la semana ---
    print(f"\n--- dias de la semana {slabel} ---")
    print_tabulate(stats_day_of_week(df_t))

    # --- goles por mes ---
    print(f"\n--- goles por mes {slabel} ---")
    print_tabulate(stats_month_season(df_t))

    # --- analisis profundo primer tiempo ---
    print(f"\n--- efecto del marcador al HT: calma en el ST {slabel} ---")
    ht_eff, corr_ht_st2 = stats_ht_calma(df_t)
    print(f"  correlacion ht_goals vs second_half_goals: {corr_ht_st2}")
    print_tabulate(ht_eff)

    print(f"\n--- si hay 1 gol en HT, ¿qué pasa en el ST? {slabel} ---")
    print_tabulate(stats_ht1_effect(df_t))

    ht_prob = stats_ht_goal_prob(df_t)
    print(f"\n--- probabilidad de goles en el primer tiempo {slabel} ---")
    for k, v in ht_prob.items():
        print(f"  {k}: {v}")

    ht00 = stats_scoreline_00ht_explosion(df_t)
    if ht00:
        print(f"\n--- partidos 0-0 al HT: explosion en el ST {slabel} ---")
        for k, v in ht00.items():
            if k != "top_marcadores_ft":
                print(f"  {k}: {v}")
        if "top_marcadores_ft" in ht00:
            print(f"  top marcadores FT desde 0-0 HT: {ht00['top_marcadores_ft']}")

    s10 = stats_scoreline_1_0_ht_extended(df_t)
    if s10:
        print(f"\n--- marcador 1-0 al HT: como termina {slabel} ---")
        for k, v in s10.items():
            print(f"  {k}: {v}")

    print(f"\n--- btts por dia de semana {slabel} ---")
    print_tabulate(stats_btts_by_day(df_t))
    bfd = stats_btts_finde_vs_entresemana(df_t)
    print(f"  finde btts {bfd['finde_btts']}% | entresemana btts {bfd['entre_btts']}% | diff {bfd['diff_btts']}%")

    print(f"\n--- inicio vs final de temporada {slabel} ---")
    print_tabulate(stats_late_season_goals(df_t))

    print(f"\n--- movimiento cuota local (granular) {slabel} ---")
    print_tabulate(stats_odds_movement_detail(df_t))

    # --- distribuciones ---
    print(f"\n--- distribucion goles {slabel} ---")
    print_tabulate(stats_distribution(df_t,"total_goals",[0,1,2,3,4,5,6,7,8,9,10,20]))
    print(f"\n--- distribucion odds local {slabel} ---")
    print_tabulate(stats_distribution(df_t,"AvgH",[1.0,1.3,1.5,1.75,2.0,2.5,3.0,4.0,6.0,25.0]))
    print(f"\n--- distribucion odds visitante {slabel} ---")
    print_tabulate(stats_distribution(df_t,"AvgA",[1.0,1.5,2.0,2.5,3.0,4.0,6.0,10.0,40.0]))

    # --- partidos alta anotacion listado ---
    ha=df_t[df_t["high_scoring"]==1]
    print(f"\n--- top 10 partidos mas goleadores {slabel} ---")
    if len(ha):
        print_tabulate(ha.sort_values("total_goals",ascending=False).head(10)[
            ["Div","Date","HomeTeam","AwayTeam","FTHG","FTAG","HTHG","HTAG","FTR"]])

    # acumular resumen temporada
    row_t={"season":slabel}
    row_t.update(g_t); row_t.update(stats_flags(df_t,goal_flags))
    row_t.update(stats_results(df_t)); row_t.update(stats_odds(df_t))
    row_t.update(ud_t); row_t.update(sm_t); row_t.update(cb)
    resumen_temporadas.append(row_t)

    # ==== CICLO INTERIOR: LIGA ====
    for liga in ligas:
        df_tl=df_t[df_t["Div"]==liga]
        if df_tl.empty: continue

        print(f"\n  {'-'*60}")
        print(f"  {liga} ({LIGAS_NAME.get(liga,liga)}) | {slabel}  ({len(df_tl)} partidos)")
        print(f"  {'-'*60}")

        g_tl=stats_goals(df_tl); r_tl=stats_results(df_tl)
        f_tl=stats_flags(df_tl,goal_flags); o_tl=stats_odds(df_tl)
        ud_tl=stats_underdogs(df_tl); sm_tl=stats_smart_money(df_tl)
        cb_tl=stats_comeback(df_tl); om_tl=stats_odds_mode(df_tl)
        iv_tl=stats_implied_vs_real(df_tl)

        # goles: resumen en una linea + tabla moda vs media
        print(f"  avg goles: total {g_tl['avg_total']} (moda {g_tl['moda_total']}) | "
              f"local {g_tl['avg_local']} | visit {g_tl['avg_visitante']} | max {g_tl['max_goles']}")
        print(f"  HT: avg {g_tl['avg_ht']} moda {g_tl['moda_ht']} | "
              f"ST: avg {g_tl['avg_st']} moda {g_tl['moda_st']} | ratio HT/ST {g_tl['ratio_ht_st']}")
        print(f"  corr HT->FT: {g_tl['corr_ht_ft']} | corr HT->ST: {g_tl['corr_ht_st']}")

        # analisis HT vs FT
        print(f"  tabla HT vs FT:")
        print_tabulate(stats_halftime_analysis(df_tl))

        # 5+ goles
        hs_tl=stats_high_scoring_analysis(df_tl)
        if hs_tl:
            print(f"  5+goles: {hs_tl['total_5plus']} ({hs_tl['pct_5plus']}%) | "
                  f"ht0 en 5+: {hs_tl['pct_ht0_en_5plus']}% | "
                  f"ht3+ en 5+: {hs_tl['pct_ht3p_en_5plus']}% | "
                  f"avg ST en 5+: {hs_tl['avg_st_en_5plus']}")

        # over vs under goles
        print(f"  over vs under goles:")
        print_tabulate(stats_over_under_goals(df_tl))

        # btts profile
        print(f"  btts vs no btts:")
        print_tabulate(stats_btts_profile(df_tl))

        # resultados y flags
        print(f"  FT: H {r_tl['H']} ({r_tl['pct_H']}%) D {r_tl['D']} ({r_tl['pct_D']}%) A {r_tl['A']} ({r_tl['pct_A']}%)")
        print(f"  HT: H {r_tl['ht_H']} D {r_tl['ht_D']} A {r_tl['ht_A']}")
        print(f"  btts {f_tl['btts']}% | over2.5 {f_tl['over25']}% | over3.5 {f_tl['over35']}% | under2.5 {f_tl['under25']}%")
        print(f"  cs_h {f_tl['clean_sheet_h']}% | cs_a {f_tl['clean_sheet_a']}% | goalless {f_tl['goalless']}% | 5+ {f_tl['high_scoring']}%")
        print(f"  remontada loc {cb_tl['remontada_loc']} ({cb_tl['pct_remontada_loc']}%) | "
              f"vis {cb_tl['remontada_vis']} ({cb_tl['pct_remontada_vis']}%)")

        # odds
        print(f"  odds: H {o_tl['avg_AvgH']} D {o_tl['avg_AvgD']} A {o_tl['avg_AvgA']} | "
              f"CH {o_tl['avg_AvgCH']} CA {o_tl['avg_AvgCA']} | overround {o_tl['avg_overround']}")
        print(f"  mov: H {round(df_tl['odds_move_H'].mean(),4)} D {round(df_tl['odds_move_D'].mean(),4)} A {round(df_tl['odds_move_A'].mean(),4)}")
        print(f"  moda odd local  | ap {om_tl['moda_ap_H']} med {om_tl['med_ap_H']} | ci {om_tl['moda_ci_H']} med {om_tl['med_ci_H']}")
        print(f"  moda odd visita | ap {om_tl['moda_ap_A']} med {om_tl['med_ap_A']} | ci {om_tl['moda_ci_A']} med {om_tl['med_ci_A']}")
        print(f"  implied vs real: H {iv_tl['imp_H']}% vs {iv_tl['real_H']}% (diff {iv_tl['diff_H']}%) | "
              f"A {iv_tl['imp_A']}% vs {iv_tl['real_A']}% (diff {iv_tl['diff_A']}%)")

        # equilibrio cuotas
        print(f"  equilibrio cuotas H vs A:")
        print_tabulate(stats_odds_gap(df_tl))

        # underdogs y smart money
        print(f"  ud visit: {ud_tl['ud_away_total']} ({ud_tl['ud_away_pct']}%) | "
              f"ud local: {ud_tl['ud_home_total']} ({ud_tl['ud_home_pct']}%) | "
              f"extremos: {ud_tl['ud_ext_total']} ({ud_tl['ud_ext_pct']}%)")
        print(f"  sm local: {sm_tl['sm_loc_n']} ({sm_tl['sm_loc_pct']}% {sm_tl['sm_loc_goles']}g) | "
              f"visit: {sm_tl['sm_vis_n']} ({sm_tl['sm_vis_pct']}% {sm_tl['sm_vis_goles']}g)")

        # efecto racha
        print(f"  efecto racha equipo local:")
        print_tabulate(stats_streak_effect(df_tl))

        # dias de la semana + jornada
        print(f"  dias de la semana:")
        print_tabulate(stats_day_of_week(df_tl))
        print(f"  jornada segun n partidos ese dia:")
        print_tabulate(stats_jornada_size(df_tl))
        print(f"  cruce dia x jornada:")
        print_tabulate(stats_day_x_jornada(df_tl))

        # goles por mes
        print(f"  goles por mes {liga} {slabel}:")
        print_tabulate(stats_month_season(df_tl))

        # top goleadores
        print(f"  top goles local:  {top_scorers(df_tl,'HomeTeam','FTHG')}")
        print(f"  top goles visit:  {top_scorers(df_tl,'AwayTeam','FTAG')}")
        print(f"  top cs local:     {top_scorers(df_tl,'HomeTeam','clean_sheet_h')}")

        # ht calma
        ht_eff_tl, corr_tl = stats_ht_calma(df_tl)
        print(f"  ht calma (corr ht->st: {corr_tl}):")
        print_tabulate(ht_eff_tl[["ht_goles","partidos","avg_st","pct_st_mayor_ht","pct_btts","pct_over25"]])

        # probabilidad goles HT
        htpr = stats_ht_goal_prob(df_tl)
        print(f"  P(HT=0)={htpr['P(HT=0)']}% P(HT>=1)={htpr['P(HT>=1)']}% "
              f"P(HT>=2)={htpr['P(HT>=2)']}% P(FT=0|HT=0)={htpr['P(FT=0|HT=0)']}% "
              f"P(FT>=3|HT=0)={htpr['P(FT>=3|HT=0)']}%")

        # 0-0 HT
        ht00_tl = stats_scoreline_00ht_explosion(df_tl)
        if ht00_tl:
            print(f"  0-0 HT → avg_goles_ft {ht00_tl['avg_goles_ft']} | "
                  f"pct_ft_0-0 {ht00_tl['pct_ft_0_0']}% | "
                  f"pct_3mas {ht00_tl['pct_ft_3mas']}% | "
                  f"btts {ht00_tl['pct_btts_desde_ht0']}%")

        # btts por dia
        print(f"  btts por dia:")
        print_tabulate(stats_btts_by_day(df_tl)[["day_name","partidos","pct_btts","avg_goles","pct_over25"]])

        # tabla posiciones + segmentos
        standing=calc_standings(df_tl)
        print(f"  tabla de posiciones {liga} {slabel}:")
        standing_print = standing.copy()
        standing_print.insert(0, "pos", range(1, len(standing_print)+1))
        print_tabulate(standing_print)

        n_eq=len(standing)
        top5=list(standing.head(5)["equipo"]); bot5=list(standing.tail(5)["equipo"])
        mid=list(standing.iloc[5:n_eq-5]["equipo"]) if n_eq>10 else []
        segs=[
            stats_segment(df_tl,"todos"),
            stats_segment(df_tl[df_tl["HomeTeam"].isin(top5)|df_tl["AwayTeam"].isin(top5)],"top5"),
            stats_segment(df_tl[df_tl["HomeTeam"].isin(bot5)|df_tl["AwayTeam"].isin(bot5)],"bottom5"),
        ]
        if mid: segs.append(stats_segment(df_tl[df_tl["HomeTeam"].isin(mid)|df_tl["AwayTeam"].isin(mid)],"medio"))
        print(f"  top5 vs bottom5 vs media:")
        print_tabulate(pd.DataFrame(segs))

        ude_tl=df_tl[(df_tl["AvgA"]>8)&(df_tl["away_win"]==1)]
        if len(ude_tl):
            print(f"  underdogs extremos ganadores:")
            print_tabulate(ude_tl[["Date","HomeTeam","AwayTeam","AvgH","AvgA","FTHG","FTAG","HTHG","HTAG"]].sort_values("AvgA",ascending=False))

        row_tl={"season":slabel,"liga":liga}
        row_tl.update(g_tl); row_tl.update(f_tl); row_tl.update(r_tl)
        row_tl.update(o_tl); row_tl.update(ud_tl); row_tl.update(sm_tl); row_tl.update(cb_tl)
        resumen_ligas_temp.append(row_tl)


# comparacion entre temporadas

print(f"\n{'='*70}\n  COMPARACION ENTRE TEMPORADAS\n{'='*70}")
df_rt=pd.DataFrame(resumen_temporadas)

print("\n=== goles por temporada ===")
print_tabulate(df_rt[["season","partidos","avg_local","avg_visitante","avg_total","moda_total","std_total","avg_ht","moda_ht","avg_st","moda_st","ratio_ht_st"]])

print("\n=== moda vs media goles por temporada (tabla completa) ===")
for row in resumen_temporadas:
    s=row["season"]; sub=df[df["Season_label"]==s]
    print(f"\n  {s}:"); print_tabulate(stats_goals_mode_table(sub))

print("\n=== analisis HT vs FT por temporada ===")
for row in resumen_temporadas:
    s=row["season"]; sub=df[df["Season_label"]==s]
    print(f"\n  {s}:"); print_tabulate(stats_halftime_analysis(sub))

print("\n=== over vs under goles por temporada ===")
for row in resumen_temporadas:
    s=row["season"]; sub=df[df["Season_label"]==s]
    print(f"\n  {s}:"); print_tabulate(stats_over_under_goals(sub))

print("\n=== btts vs no btts por temporada ===")
for row in resumen_temporadas:
    s=row["season"]; sub=df[df["Season_label"]==s]
    print(f"\n  {s}:"); print_tabulate(stats_btts_profile(sub))

print("\n=== equilibrio cuotas por temporada ===")
for row in resumen_temporadas:
    s=row["season"]; sub=df[df["Season_label"]==s]
    print(f"\n  {s}:"); print_tabulate(stats_odds_gap(sub))

print("\n=== efecto racha por temporada ===")
for row in resumen_temporadas:
    s=row["season"]; sub=df[df["Season_label"]==s]
    print(f"\n  {s}:"); print_tabulate(stats_streak_effect(sub))

print("\n=== btts / over / under por temporada ===")
print_tabulate(df_rt[["season","btts","over15","over25","over35","over45","under15","under25","under35"]])

print("\n=== clean sheets y especiales por temporada ===")
print_tabulate(df_rt[["season","clean_sheet_h","clean_sheet_a","high_scoring","goalless"]])

print("\n=== resultados FT por temporada ===")
print_tabulate(df_rt[["season","H","pct_H","D","pct_D","A","pct_A"]])

print("\n=== resultados HT por temporada ===")
print_tabulate(df_rt[["season","ht_H","ht_D","ht_A"]])

print("\n=== remontadas por temporada ===")
print_tabulate(df_rt[["season","remontada_loc","pct_remontada_loc","remontada_vis","pct_remontada_vis","ht_win_ft_lose","pct_ht_win_ft_lose"]])

print("\n=== odds y overround por temporada ===")
print_tabulate(df_rt[["season","avg_AvgH","avg_AvgD","avg_AvgA","avg_AvgCH","avg_AvgCA","avg_overround"]])

print("\n=== underdogs por temporada ===")
print_tabulate(df_rt[["season","ud_away_total","ud_away_wins","ud_away_pct","ud_away_avgOdd","ud_ext_total","ud_ext_wins","ud_ext_pct"]])

print("\n=== smart money por temporada ===")
print_tabulate(df_rt[["season","sm_loc_n","sm_loc_pct","sm_loc_goles","sm_vis_n","sm_vis_pct","sm_vis_goles"]])


# comparacion entre ligas (todo el periodo)

print(f"\n{'='*70}\n  COMPARACION ENTRE LIGAS (todo el periodo)\n{'='*70}")

print("\n=== goles por liga ===")
print_tabulate(df.groupby("Div").agg(
    partidos=("total_goals","count"),
    avg_local=("FTHG","mean"), avg_visitante=("FTAG","mean"),
    avg_total=("total_goals","mean"), std_total=("total_goals","std"),
    avg_ht=("ht_goals","mean"), avg_st=("second_half_goals","mean"), max_goles=("total_goals","max"),
).round(3).reset_index())

print("\n=== moda vs media goles por liga ===")
for liga in ligas:
    print(f"\n  {liga} ({LIGAS_NAME.get(liga,liga)}):")
    print_tabulate(stats_goals_mode_table(df[df["Div"]==liga]))

print("\n=== varianza y dispersion goles por liga ===")
var_rows=[]
for liga in ligas:
    s=df[df["Div"]==liga]["total_goals"]
    var_rows.append({"liga":liga,"mean":round(s.mean(),3),"moda":int(s.mode()[0]),
        "var":round(s.var(),3),"std":round(s.std(),3),"cv_%":round(s.std()/s.mean()*100,2),
        "iqr":round(s.quantile(0.75)-s.quantile(0.25),3),
        "p10":round(s.quantile(0.10),3),"p90":round(s.quantile(0.90),3),
        "skew":round(s.skew(),3),"kurt":round(s.kurt(),3)})
print_tabulate(pd.DataFrame(var_rows))

print("\n=== analisis HT vs FT por liga ===")
for liga in ligas:
    print(f"\n  {liga}:"); print_tabulate(stats_halftime_analysis(df[df["Div"]==liga]))

print("\n=== partidos 5+ goles por liga ===")
hs_rows=[]
for liga in ligas:
    h=stats_high_scoring_analysis(df[df["Div"]==liga])
    if h: h["liga"]=liga; hs_rows.append(h)
print_tabulate(pd.DataFrame(hs_rows)[["liga","total_5plus","pct_5plus","avg_ht_en_5plus",
    "pct_ht0_en_5plus","pct_ht1_en_5plus","pct_ht2_en_5plus","pct_ht3p_en_5plus","avg_st_en_5plus"]])

print("\n=== over vs under goles por liga ===")
for liga in ligas:
    print(f"\n  {liga}:"); print_tabulate(stats_over_under_goals(df[df["Div"]==liga]))

print("\n=== btts vs no btts por liga ===")
for liga in ligas:
    print(f"\n  {liga}:"); print_tabulate(stats_btts_profile(df[df["Div"]==liga]))

print("\n=== equilibrio cuotas por liga ===")
for liga in ligas:
    print(f"\n  {liga}:"); print_tabulate(stats_odds_gap(df[df["Div"]==liga]))

print("\n=== efecto racha por liga ===")
for liga in ligas:
    print(f"\n  {liga}:"); print_tabulate(stats_streak_effect(df[df["Div"]==liga]))

print("\n=== correlacion HT vs FT por liga ===")
for liga in ligas:
    sub=df[df["Div"]==liga]
    print(f"  {liga} — HT->FT: {round(sub['ht_goals'].corr(sub['total_goals']),4)} | HT->ST: {round(sub['ht_goals'].corr(sub['second_half_goals']),4)}")

print("\n=== varianza odds por liga ===")
vo=[]
for liga in ligas:
    sub=df[df["Div"]==liga]
    vo.append({"liga":liga,
        "var_H":round(sub["AvgH"].var(),3),"cv_H":round(sub["AvgH"].std()/sub["AvgH"].mean()*100,2),
        "iqr_H":round(sub["AvgH"].quantile(0.75)-sub["AvgH"].quantile(0.25),3),"skew_H":round(sub["AvgH"].skew(),3),
        "var_A":round(sub["AvgA"].var(),3),"cv_A":round(sub["AvgA"].std()/sub["AvgA"].mean()*100,2),
        "iqr_A":round(sub["AvgA"].quantile(0.75)-sub["AvgA"].quantile(0.25),3),"skew_A":round(sub["AvgA"].skew(),3),
    })
print_tabulate(pd.DataFrame(vo))

print("\n=== distribucion goles (todo el periodo) ===")
print_tabulate(stats_distribution(df,"total_goals",[0,1,2,3,4,5,6,7,8,9,10,20]))
print("\n=== distribucion goles HT ===")
print_tabulate(stats_distribution(df,"ht_goals",[0,1,2,3,4,5,10]))
print("\n=== distribucion goles ST ===")
print_tabulate(stats_distribution(df,"second_half_goals",[0,1,2,3,4,5,10]))
print("\n=== distribucion odds local ===")
print_tabulate(stats_distribution(df,"AvgH",[1.0,1.3,1.5,1.75,2.0,2.5,3.0,4.0,6.0,25.0]))
print("\n=== distribucion odds visitante ===")
print_tabulate(stats_distribution(df,"AvgA",[1.0,1.5,2.0,2.5,3.0,4.0,6.0,10.0,40.0]))

print("\n=== tasa acierto real vs implied por rango odd ===")
print_tabulate(stats_win_rate_by_odd(df))

print("\n=== implied vs real por liga ===")
iv_rows=[]
for liga in ligas:
    iv=stats_implied_vs_real(df[df["Div"]==liga]); iv["liga"]=liga; iv_rows.append(iv)
print_tabulate(pd.DataFrame(iv_rows)[["liga","imp_H","real_H","diff_H","imp_A","real_A","diff_A","imp_D","real_D","diff_D"]])

print("\n=== moda y mediana cuotas ganadoras por liga ===")
om_l=[]
for liga in ligas:
    om=stats_odds_mode(df[df["Div"]==liga]); om["liga"]=liga; om_l.append(om)
print_tabulate(pd.DataFrame(om_l)[["liga","moda_ap_H","med_ap_H","moda_ci_H","med_ci_H",
    "moda_ap_A","med_ap_A","moda_ci_A","med_ci_A","moda_ud_ap_A","med_ud_ap_A"]])

print("\n=== btts / over / under por liga ===")
print_tabulate((df.groupby("Div")[goal_flags].mean()*100).round(2).reset_index())

print("\n=== resultados por liga ===")
ftr_l=df.groupby(["Div","FTR"]).size().unstack(fill_value=0).reset_index()
ftr_l["total"]=ftr_l[["A","D","H"]].sum(axis=1)
for c in ["H","D","A"]: ftr_l[f"pct_{c}"]=round(ftr_l[c]/ftr_l["total"]*100,2)
print_tabulate(ftr_l)

print("\n=== remontadas por liga ===")
cb_l=[]
for liga in ligas:
    cb=stats_comeback(df[df["Div"]==liga]); cb["liga"]=liga; cb_l.append(cb)
print_tabulate(pd.DataFrame(cb_l)[["liga","remontada_loc","pct_remontada_loc","remontada_vis","pct_remontada_vis","ht_win_ft_lose","pct_ht_win_ft_lose"]])

print("\n=== equipos que mas remontan y mas dejan escapar (todo el periodo) ===")
cb_teams=stats_comeback_teams(df)
print("  top 15 remontan:")
print_tabulate(cb_teams.head(15))
print("  top 15 pierden ventaja HT:")
print_tabulate(cb_teams.sort_values("total_choke",ascending=False).head(15))

print("\n=== odds por liga ===")
print_tabulate(df.groupby("Div")[["AvgH","AvgD","AvgA","AvgCH","AvgCD","AvgCA","overround"]].mean().round(3).reset_index())

print("\n=== movimiento mercado por liga ===")
print_tabulate(df.groupby("Div")[["odds_move_H","odds_move_D","odds_move_A"]].agg(["mean","std"]).round(4).reset_index())

print("\n=== underdogs por liga ===")
for liga in ligas:
    ud=stats_underdogs(df[df["Div"]==liga])
    print(f"  {liga} — visit: {ud['ud_away_total']} ({ud['ud_away_pct']}% avg {ud['ud_away_avgOdd']}) | local: {ud['ud_home_total']} ({ud['ud_home_pct']}%) | ext: {ud['ud_ext_total']} ({ud['ud_ext_pct']}%)")

print("\n=== smart money por liga ===")
for liga in ligas:
    sm=stats_smart_money(df[df["Div"]==liga])
    print(f"  {liga} — local: {sm['sm_loc_n']} ({sm['sm_loc_pct']}% {sm['sm_loc_goles']}g) | visit: {sm['sm_vis_n']} ({sm['sm_vis_pct']}% {sm['sm_vis_goles']}g)")

print("\n=== dias de la semana por liga ===")
for liga in ligas:
    print(f"\n  {liga}:"); print_tabulate(stats_day_of_week(df[df["Div"]==liga]))

print("\n=== jornada segun n partidos ese dia por liga ===")
for liga in ligas:
    print(f"\n  {liga}:"); print_tabulate(stats_jornada_size(df[df["Div"]==liga]))

print("\n=== cruce dia x jornada por liga ===")
for liga in ligas:
    print(f"\n  {liga}:"); print_tabulate(stats_day_x_jornada(df[df["Div"]==liga]))

print("\n=== goles por mes del anio por liga ===")
for liga in ligas:
    print(f"\n  {liga}:"); print_tabulate(stats_month_season(df[df["Div"]==liga]))

print("\n=== top5 vs bottom5 vs media por liga (todo el periodo) ===")
for liga in ligas:
    sub=df[df["Div"]==liga]; stnd=calc_standings(sub); n_eq=len(stnd)
    top5=list(stnd.head(5)["equipo"]); bot5=list(stnd.tail(5)["equipo"])
    mid=list(stnd.iloc[5:n_eq-5]["equipo"]) if n_eq>10 else []
    segs=[stats_segment(sub,"todos"),
          stats_segment(sub[sub["HomeTeam"].isin(top5)|sub["AwayTeam"].isin(top5)],"top5"),
          stats_segment(sub[sub["HomeTeam"].isin(bot5)|sub["AwayTeam"].isin(bot5)],"bottom5")]
    if mid: segs.append(stats_segment(sub[sub["HomeTeam"].isin(mid)|sub["AwayTeam"].isin(mid)],"medio"))
    print(f"\n  {liga}:"); print_tabulate(pd.DataFrame(segs))

# secciones adicionales de analisis

print(f"\n{'='*70}\n  ANALISIS PROFUNDO PRIMER TIEMPO — COMPARACION ENTRE LIGAS\n{'='*70}")

print("\n=== calma en el ST segun goles en HT (todo el periodo) ===")
ht_eff_all, corr_all = stats_ht_calma(df)
print(f"  correlacion ht_goals vs second_half_goals (global): {corr_all}")
print_tabulate(ht_eff_all)
print("\n  por liga:")
for liga in ligas:
    sub = df[df["Div"]==liga]
    _, c = stats_ht_calma(sub)
    print(f"  {LIGAS_NAME.get(liga,liga):<20} corr ht->st: {c}")

print("\n=== efecto del marcador al HT en el ST ===")
print_tabulate(stats_ht1_effect(df))

print("\n=== probabilidad de goles en el primer tiempo por liga ===")
htpr_rows = []
for liga in ligas:
    p = stats_ht_goal_prob(df[df["Div"]==liga])
    p["liga"] = liga
    htpr_rows.append(p)
print_tabulate(pd.DataFrame(htpr_rows)[["liga","P(HT=0)","P(HT=1)","P(HT>=2)",
    "pct_btts_HT","P(FT=0|HT=0)","P(FT>=3|HT=0)"]])

print("\n=== cuando va 0-0 al HT: explosion en el segundo tiempo por liga ===")
print_tabulate(stats_scoreline_0_0_ht_deep(df))

print("\n=== marcador 1-0 al HT: como termina el partido por liga ===")
s10_rows = []
for liga in ligas:
    r = stats_scoreline_1_0_ht_extended(df[df["Div"]==liga])
    if r:
        r["liga"] = liga
        s10_rows.append(r)
if s10_rows:
    print_tabulate(pd.DataFrame(s10_rows)[["liga","total","pct_mantiene_H","pct_empata",
        "pct_remonta_A","pct_1_0_ft","pct_1_1_ft","avg_goles_ft","pct_btts"]])

print("\n=== top equipos que marcan en HT y que reciben goles en HT ===")
ht_sc = stats_ht_scorers_top(df)
print("  top marcadores en HT:")
for k, v in list(ht_sc["top_marcadores_ht"].items())[:15]:
    print(f"    {k}: {v}")
print("  top que reciben en HT:")
for k, v in list(ht_sc["top_reciben_ht"].items())[:15]:
    print(f"    {k}: {v}")
print("  ratio HT/FT de goles marcados (marcan temprano):")
for k, v in list(ht_sc["ratio_ht_ft_marcados"].items())[:15]:
    print(f"    {k}: {v}")

print(f"\n{'='*70}\n  BTTS POR DIA Y PATRONES\n{'='*70}")

print("\n=== btts por dia de semana (todo el periodo) ===")
print_tabulate(stats_btts_by_day(df))

bfd_all = stats_btts_finde_vs_entresemana(df)
print(f"\n=== finde vs entresemana ===")
for k, v in bfd_all.items():
    print(f"  {k}: {v}")

print("\n=== btts por dia por liga ===")
for liga in ligas:
    print(f"\n  {liga}:")
    print_tabulate(stats_btts_by_day(df[df["Div"]==liga])[["day_name","partidos","pct_btts","avg_goles","pct_over25"]])

print("\n=== equipos donde mas se da BTTS en el primer tiempo ===")
btts_ht_top, btts_st_top = stats_btts_team_ht(df)
print("  top BTTS en HT:")
print_tabulate(btts_ht_top)
print("  top BTTS solo en ST (no en HT):")
print_tabulate(btts_st_top)

print(f"\n{'='*70}\n  CUOTAS: MODAS Y MOVIMIENTO DETALLADO\n{'='*70}")

print("\n=== modas de cuotas mas frecuentes (todos los partidos) ===")
print_tabulate(stats_odds_mode_all_results(df))

print("\n=== modas de cuotas ganadoras por resultado (top 10 cada uno) ===")
print_tabulate(stats_odds_mode_full(df))

print("\n=== modas de cuotas ganadoras por liga ===")
om_l=[]
for liga in ligas:
    om=stats_odds_mode(df[df["Div"]==liga]); om["liga"]=liga; om_l.append(om)
print_tabulate(pd.DataFrame(om_l)[["liga","moda_ap_H","med_ap_H","moda_ci_H","med_ci_H",
    "moda_ap_A","med_ap_A","moda_ci_A","med_ci_A","moda_ud_ap_A","med_ud_ap_A"]])

print("\n=== movimiento de cuota local (granularidad alta) — todo el periodo ===")
print_tabulate(stats_odds_movement_detail(df))

print("\n=== movimiento de cuota por liga ===")
for liga in ligas:
    print(f"\n  {liga}:")
    print_tabulate(stats_odds_movement_detail(df[df["Div"]==liga]))

print(f"\n{'='*70}\n  CLEAN SHEETS POR EQUIPO\n{'='*70}")

print("\n=== top equipos por clean sheet (local y visitante) ===")
cs_dict = stats_clean_sheet_teams(df)
print("  top clean sheet como local:")
print_tabulate(cs_dict["top_cs_local"])
print("  top clean sheet como visitante:")
print_tabulate(cs_dict["top_cs_visitante"])

print("\n=== clean sheet por mes del año (¿más defensivos en invierno?) ===")
print_tabulate(stats_clean_sheet_by_month(df))

print(f"\n{'='*70}\n  VARIANZA DE GOLES POR EQUIPO (CONSISTENTES VS IMPREDECIBLES)\n{'='*70}")
var_dict = stats_team_variance_goals(df)
print("\n  mas consistentes (baja std):")
print_tabulate(var_dict["mas_consistentes"])
print("\n  mas impredecibles (alta std):")
print_tabulate(var_dict["mas_impredecibles"])

print(f"\n{'='*70}\n  INICIO VS FINAL DE TEMPORADA\n{'='*70}")
print("\n=== inicio (ago-oct) vs medio (nov-feb) vs final (mar-may) — todo el periodo ===")
print_tabulate(stats_late_season_goals(df))
print("\n  por liga:")
for liga in ligas:
    print(f"\n  {liga}:")
    print_tabulate(stats_late_season_goals(df[df["Div"]==liga]))


# rachas extendidas: comparacion global y por liga

print(f"\n{'='*70}")
print(f"  RACHAS — ANALISIS EXTENDIDO")
print(f"{'='*70}")

print("\n=== efecto de la racha previa del local: goles y resultados ===")
print("  (racha neta: positivo=victorias, negativo=derrotas, 0=neutro)")
print_tabulate(stats_bad_streak_analysis(df))

print("\n=== por liga ===")
for liga in ligas:
    print(f"\n  {liga}:")
    print_tabulate(stats_bad_streak_analysis(df[df["Div"]==liga]))

print("\n=== patron de goles por valor exacto de racha previa (-5 a +5) ===")
print("  ¿Cuánto golea y cuánto recibe el local según su historial reciente?")
print_tabulate(stats_streak_goals_pattern(df))

print("\n=== momentum: partido anterior del local → resultado de hoy ===")
print("  ¿Qué pasa HOY según lo que pasó el partido anterior (goleada/derrota/etc.)?")
print_tabulate(stats_streak_momentum(df))

print("\n=== por liga ===")
for liga in ligas:
    print(f"\n  {liga}:")
    print_tabulate(stats_streak_momentum(df[df["Div"]==liga]))

print("\n=== FENOMENO: racha mala → goleada → siguiente partido (análisis profundo) ===")
print("  (racha mala ≥2 partidos sin ganar, goleada = marcar 3+ goles)")
bse = stats_bad_streak_explosion_deep(df, min_bad=2, min_goles_goleada=3)
if "sin_eventos" not in bse and "sin_siguiente" not in bse:
    print(f"  total eventos detectados: {bse['total_eventos']}")
    print(f"  tras goleada: gana {bse['pct_gana_siguiente']}% | "
          f"pierde {bse['pct_pierde_siguiente']}% | "
          f"empata {bse['pct_empata_siguiente']}%")
    print(f"  avg goles a favor siguiente: {bse['avg_gf_siguiente']} | "
          f"en contra: {bse['avg_gc_siguiente']}")
    print(f"  % que repite goleada en siguiente: {bse['pct_repite_goleada']}%")
    print(f"  baseline % vic tras cualquier victoria: {bse['baseline_pct_vic_tras_cualquier_victoria']}%")
    print("\n  por intensidad de racha previa:")
    print_tabulate(bse["tabla_por_intensidad"])

print("\n  [variante más estricta: racha ≥3, goleada 4+]")
bse2 = stats_bad_streak_explosion_deep(df, min_bad=3, min_goles_goleada=4)
if "sin_eventos" not in bse2 and "sin_siguiente" not in bse2:
    print(f"  total eventos: {bse2['total_eventos']} | "
          f"gana {bse2['pct_gana_siguiente']}% | "
          f"pierde {bse2['pct_pierde_siguiente']}% | "
          f"avg goles sig: {bse2['avg_gf_siguiente']}")
    if not bse2["tabla_por_intensidad"].empty:
        print_tabulate(bse2["tabla_por_intensidad"])

print("\n=== rebote tras derrota según la magnitud de la derrota ===")
print("  ¿Tras una goleada recibida el equipo local reacciona con más goles?")
print_tabulate(stats_rebound_after_loss(df))

print("\n  por liga:")
for liga in ligas:
    sub = df[df["Div"]==liga]
    rbl = stats_rebound_after_loss(sub)
    if not rbl.empty:
        print(f"\n  {liga}:")
        print_tabulate(rbl)

print("\n=== como se rompen las rachas ganadoras ===")
print("  Cuando un equipo lleva N victorias seguidas, ¿cómo lo frenan?")
print_tabulate(stats_winning_streak_end(df))

print("\n  por liga:")
for liga in ligas:
    wse = stats_winning_streak_end(df[df["Div"]==liga])
    if not wse.empty:
        print(f"\n  {liga}:")
        print_tabulate(wse)

print("\n=== equipos que mejor reaccionan tras mala racha (rebote como local) ===")
print("  rebote = pct_vic_tras_racha_mala - pct_vic_neutro")
sby = stats_streak_by_team(df)
if not sby.empty:
    print("  top 15 con mayor rebote:")
    print_tabulate(sby.head(15)[["equipo","pj_local","n_tras_mala","pct_vic_tras_mala",
                                   "pct_vic_neutro","pct_vic_tras_buena","rebote","caida"]])
    print("  top 15 con mayor caída tras buena racha:")
    print_tabulate(sby.sort_values("caida").head(15)[
        ["equipo","pj_local","n_tras_buena","pct_vic_tras_buena",
         "pct_vic_neutro","rebote","caida"]])

print("\n=== efecto de empates consecutivos del local ===")
print("  ¿Después de N empates seguidos el local es más explosivo?")
print_tabulate(stats_draw_streak_effect(df))

print("\n  por liga:")
for liga in ligas:
    dse = stats_draw_streak_effect(df[df["Div"]==liga])
    if not dse.empty:
        print(f"\n  {liga}:")
        print_tabulate(dse)

print("\n=== top 15 goleadores ===")
hg=df.groupby("HomeTeam")["FTHG"].sum().reset_index().rename(columns={"HomeTeam":"equipo","FTHG":"g_loc"})
ag=df.groupby("AwayTeam")["FTAG"].sum().reset_index().rename(columns={"AwayTeam":"equipo","FTAG":"g_vis"})
tg=hg.merge(ag,on="equipo",how="outer").fillna(0)
tg[["g_loc","g_vis"]]=tg[["g_loc","g_vis"]].astype(int); tg["total"]=tg["g_loc"]+tg["g_vis"]
print_tabulate(tg.sort_values("total",ascending=False).head(15))

print("\n=== mayor % victorias local (min 50 pj) ===")
hw=df.groupby("HomeTeam").agg(pj=("home_win","count"),v=("home_win","sum"))
hw=hw[hw["pj"]>=50]; hw["pct"]=round(hw["v"]/hw["pj"]*100,2)
print_tabulate(hw.sort_values("pct",ascending=False).head(15).reset_index())

print("\n=== mayor % victorias visitante (min 50 pj) ===")
aw=df.groupby("AwayTeam").agg(pj=("away_win","count"),v=("away_win","sum"))
aw=aw[aw["pj"]>=50]; aw["pct"]=round(aw["v"]/aw["pj"]*100,2)
print_tabulate(aw.sort_values("pct",ascending=False).head(15).reset_index())

print("\n=== clean sheets local (min 50 pj) ===")
cs=df.groupby("HomeTeam").agg(pj=("clean_sheet_h","count"),cs=("clean_sheet_h","sum"))
cs=cs[cs["pj"]>=50]; cs["pct"]=round(cs["cs"]/cs["pj"]*100,2)
print_tabulate(cs.sort_values("pct",ascending=False).head(10).reset_index())

print("\n=== dias de la semana general ===")
print_tabulate(stats_day_of_week(df))

print("\n=== finde vs entresemana vs viernes (general) ===")
dtype=df.groupby("day_type").agg(
    partidos=("total_goals","count"), avg_goles=("total_goals","mean"),
    var_goles=("total_goals","var"),  avg_ht=("ht_goals","mean"),
    avg_st=("second_half_goals","mean"),
    pct_btts=("btts","mean"), pct_over25=("over25","mean"), pct_H=("home_win","mean"),
).round(3).reset_index()
for c in ["pct_btts","pct_over25","pct_H"]: dtype[c]=round(dtype[c]*100,2)
print_tabulate(dtype)

print("\n=== goles por mes general ===")
print_tabulate(stats_month_season(df))

print("\n=== correlacion dia semana vs goles/resultado ===")
print(round(df[["dayofweek","total_goals","ht_goals","second_half_goals",
                "home_win","draw","away_win","over25","btts"]].corr(),3))

print("\n=== correlacion general ===")
corr_cols=["total_goals","FTHG","FTAG","ht_goals","second_half_goals",
           "AvgH","AvgA","AvgD","overround","odds_move_H","odds_move_A",
           "imp_prob_H","imp_prob_A","btts","over25","clean_sheet_h","clean_sheet_a"]
print(round(df[corr_cols].corr(),3))


# analisis generales para todo el periodo y todas las ligas

print(f"\n{'='*70}\n  ANALISIS GENERALES — TODO EL PERIODO\n{'='*70}")

print("\n=== marcadores exactos (todo el periodo) ===")
print_tabulate(stats_exact_scores(df))

print("\n=== marcadores exactos HT (todo el periodo) ===")
print_tabulate(stats_exact_scores_ht(df))

print("\n=== marcadores exactos por liga ===")
for liga in ligas:
    print(f"\n  {liga}:"); print_tabulate(stats_exact_scores(df[df["Div"]==liga], top_n=10))

print("\n=== marcadores HT mas frecuentes por liga ===")
for liga in ligas:
    print(f"\n  {liga}:"); print_tabulate(stats_exact_scores_ht(df[df["Div"]==liga]))

print("\n=== matriz HTR -> FTR por liga ===")
for liga in ligas:
    print(f"\n  {liga}:"); print_tabulate(stats_ht_to_ft_matrix(df[df["Div"]==liga]))

print("\n=== HTR predice FTR por liga ===")
htc_rows=[]
for liga in ligas:
    h=stats_ht_consistency(df[df["Div"]==liga]); h["liga"]=liga; htc_rows.append(h)
print_tabulate(pd.DataFrame(htc_rows)[["liga","total","pct_mismo","si_htH_pct_ftH","si_htD_pct_ftD","si_htA_pct_ftA","si_htH_remonta_A","si_htA_remonta_H"]])

print("\n=== si va 1-0 al HT: que pasa al FT, por liga ===")
for liga in ligas:
    sc=stats_scoreline_1_0_ht(df[df["Div"]==liga])
    if sc: print(f"  {liga} — mantiene H: {sc['pct_mantiene_H']}% | empata: {sc['pct_empata']}% | remonta: {sc['pct_remonta_A']}% | avg goles FT: {sc['avg_goles_ft']} | top: {sc['top_marcadores']}")

print("\n=== diferencia goles FT por liga ===")
for liga in ligas:
    print(f"\n  {liga}:"); print_tabulate(stats_goal_diff_distribution(df[df["Div"]==liga]))

print("\n=== quien marca por liga ===")
has_l=[]
for liga in ligas:
    h=stats_home_away_scoring(df[df["Div"]==liga]); h["liga"]=liga; has_l.append(h)
print_tabulate(pd.DataFrame(has_l)[["liga","pct_solo_local","pct_solo_visit","pct_ambos_marcan","pct_ninguno","avg_goles_si_local_marca","avg_goles_si_visit_marca"]])

print("\n=== ritmo HT vs ST por liga ===")
rht_l=[]
for liga in ligas:
    r=stats_st_vs_ht_rhythm(df[df["Div"]==liga]); r["liga"]=liga; rht_l.append(r)
print_tabulate(pd.DataFrame(rht_l)[["liga","pct_st_mayor","pct_st_igual","pct_st_menor","avg_goles_st_mayor","avg_goles_st_menor"]])

print("\n=== perfil cuota local por liga ===")
for liga in ligas:
    print(f"\n  {liga}:"); print_tabulate(stats_goles_by_fav_profile(df[df["Div"]==liga]))

print("\n=== superfavorito local por liga ===")
sf_rows=[]
for liga in ligas:
    s=stats_superfav_analysis(df[df["Div"]==liga]); s["liga"]=liga; sf_rows.append(s)
print_tabulate(pd.DataFrame(sf_rows)[["liga","superfav_n","pct_del_total","pct_H","pct_D","pct_A","avg_goles","pct_cs_h","pct_btts"]])

print("\n=== overround vs goles por liga ===")
for liga in ligas:
    print(f"\n  {liga}:"); print_tabulate(stats_overround_vs_goals(df[df["Div"]==liga]))

print("\n=== overround evolucion por temporada ===")
print_tabulate(stats_overround_evolution(df))

print("\n=== decaimiento ventaja local por temporada ===")
print_tabulate(stats_home_advantage_decay(df))

print("\n=== apertura vs cierre de cuotas (todo el periodo) ===")
print_tabulate(stats_apertura_vs_cierre(df))

print("\n=== apertura vs cierre de cuotas por liga ===")
for liga in ligas:
    print(f"\n  {liga}:"); print_tabulate(stats_apertura_vs_cierre(df[df["Div"]==liga]))

print("\n=== tasa sorpresa segun dif cuotas (todo el periodo) ===")
print_tabulate(stats_upset_rate(df))

print("\n=== tasa sorpresa por liga ===")
for liga in ligas:
    print(f"\n  {liga}:"); print_tabulate(stats_upset_rate(df[df["Div"]==liga]))

print("\n=== empate real vs implied por rango cuota empate (todo el periodo) ===")
print_tabulate(stats_draw_by_odd_range(df))

print("\n=== empate real vs implied por liga ===")
for liga in ligas:
    print(f"\n  {liga}:"); print_tabulate(stats_draw_by_odd_range(df[df["Div"]==liga]))

print("\n=== value betting por liga ===")
for liga in ligas:
    print(f"\n  {liga}:"); print_tabulate(stats_value_betting(df[df["Div"]==liga]))

print("\n=== volatilidad mercado por liga ===")
mv_l=[]
for liga in ligas:
    m=stats_market_volatility(df[df["Div"]==liga]); m["liga"]=liga; mv_l.append(m)
print_tabulate(pd.DataFrame(mv_l)[["liga","alta_vol_n","alta_pct_H","alta_avg_goles","alta_pct_btts","baja_vol_n","baja_avg_goles","corr_move_goles"]])

print("\n=== discrepancia B365 vs mercado (todo el periodo) ===")
cons_table, cons_top = stats_market_consensus(df)
print_tabulate(cons_table)
print("  top 10 partidos con mayor discrepancia:")
print_tabulate(cons_top)

print("\n=== discrepancia B365 vs mercado por liga ===")
for liga in ligas:
    ct,_ = stats_market_consensus(df[df["Div"]==liga])
    print(f"  {liga}:"); print_tabulate(ct)

print("\n=== top 20 partidos mas sorpresivos (todo el periodo) ===")
print_tabulate(stats_surprise_index(df))

print("\n=== top 10 sorpresas por liga ===")
for liga in ligas:
    print(f"\n  {liga}:"); print_tabulate(stats_surprise_index(df[df["Div"]==liga], top_n=10))

print("\n=== entropia de resultados por liga ===")
print_tabulate(stats_entropy_results(df))

print("\n=== concentracion goles: Gini por liga ===")
for liga in ligas:
    g=stats_gini_goals(df[df["Div"]==liga])
    if g: print(f"  {liga} — gini={g['gini_goles']} | mas goles: {g['max_equipo']} ({g['max_goles']}) | menos: {g['min_equipo']} ({g['min_goles']}) | top3={g['pct_top3']}%")

print("\n=== concentracion goles: Gini todo el periodo ===")
g_all=stats_gini_goals(df)
if g_all: print(f"  gini={g_all['gini_goles']} | mas goles: {g_all['max_equipo']} ({g_all['max_goles']}) | top3={g_all['pct_top3']}%")

print("\n=== rachas ganadoras maximas (todo el periodo) ===")
print_tabulate(stats_consecutive_streaks(df))

print("\n=== clean sheets por mes (todo el periodo) ===")
print_tabulate(stats_clean_sheet_by_month(df))

print("\n=== clean sheets por mes por liga ===")
for liga in ligas:
    print(f"\n  {liga}:"); print_tabulate(stats_clean_sheet_by_month(df[df["Div"]==liga]))

print("\n=== ratio goles local/visitante por liga y temporada ===")
print_tabulate(stats_home_local_ratio(df))

print("\n=== btts segun smart money (todo el periodo) ===")
bsm_all=stats_btts_smart_money(df)
print(f"  sm draw: btts={bsm_all['sm_draw_btts']}% goles={bsm_all['sm_draw_goles']} | sin sm: {bsm_all['no_sm_draw_btts']}%")
print(f"  sm H: btts={bsm_all['sm_H_btts']}% goles={bsm_all['sm_H_goles']} | sm A: btts={bsm_all['sm_A_btts']}% goles={bsm_all['sm_A_goles']}")
print(f"  corr move_D->btts: {bsm_all['corr_move_D_btts']} | corr move_D->goles: {bsm_all['corr_move_D_goles']}")


# rachas detalladas por equipo

print(f"\n{'='*70}\n  RACHAS DETALLADAS\n{'='*70}")

print("\n=== rachas maximas de victorias, derrotas y empates por equipo ===")
print_tabulate(stats_max_streaks_all(df).head(30))

print("\n=== rachas maximas derrotas por equipo (top 20) ===")
print_tabulate(stats_max_streaks_all(df).sort_values("max_racha_der",ascending=False).head(20).reset_index(drop=True))

print("\n=== rachas maximas empates por equipo (top 20) ===")
print_tabulate(stats_max_streaks_all(df).sort_values("max_racha_emp",ascending=False).head(20).reset_index(drop=True))

print("\n=== fenomeno: mala racha -> goleada -> siguiente partido ===")
bstg = stats_bad_streak_then_goleada(df, min_bad=3, min_goals=4)
if not bstg.empty:
    total_ev = len(bstg)
    pct_gana = round(bstg["gano_siguiente"].mean()*100,2)
    pct_pierde = round(bstg["perdio_siguiente"].mean()*100,2)
    pct_empata = round(bstg["empato_siguiente"].mean()*100,2)
    print(f"  total eventos (racha mala >=3 -> golea >=4 -> siguiente): {total_ev}")
    print(f"  resultado siguiente: gana {pct_gana}% | pierde {pct_pierde}% | empata {pct_empata}%")
    print("  top 15 eventos:")
    print_tabulate(bstg[["equipo","racha_mala_previa","marcador_goleada","rival_siguiente","resultado_siguiente"]].head(15))
    print("  por equipo (cuantas veces ocurrio):")
    print_tabulate(bstg.groupby("equipo").agg(
        veces=("perdio_siguiente","count"),
        gano_sig=("gano_siguiente","sum"),
        perdio_sig=("perdio_siguiente","sum"),
        empato_sig=("empato_siguiente","sum"),
    ).sort_values("veces",ascending=False).head(15).reset_index())
else:
    print("  no hay eventos suficientes con esos filtros")

# variante con umbral mas permisivo para capturar mas casos
print("\n  [variante: racha>=2 derrotas/sin gana, golea >=3]")
bstg2 = stats_bad_streak_then_goleada(df, min_bad=2, min_goals=3)
if not bstg2.empty:
    print(f"  total: {len(bstg2)} | gana sig: {round(bstg2['gano_siguiente'].mean()*100,2)}% | pierde: {round(bstg2['perdio_siguiente'].mean()*100,2)}%")


# btts segun como iba el marcador al descanso

print(f"\n{'='*70}\n  BTTS SEGUN ESTADO HT\n{'='*70}")

print("\n=== btts segun si el HT termino 0-0 o con goles (todo el periodo) ===")
bht = stats_btts_by_ht_state(df)
print(f"  HT sin goles ({bht['ht_goalless_n']} partidos): btts={bht['btts_si_ht0']}% | over2.5={bht['over25_si_ht0']}% | avg goles={bht['avg_goles_si_ht0']}")
print(f"  HT con goles ({bht['ht_con_gol_n']} partidos): btts={bht['btts_si_ht1p']}% | over2.5={bht['over25_si_ht1p']}% | avg goles={bht['avg_goles_si_ht1p']}")
print(f"  desglose: HT=1 -> btts={bht['btts_si_ht1']}% | HT=2+ -> btts={bht['btts_si_ht2p']}%")

print("\n=== btts segun HT por liga ===")
bht_l=[]
for liga in ligas:
    b=stats_btts_by_ht_state(df[df["Div"]==liga]); b["liga"]=liga; bht_l.append(b)
print_tabulate(pd.DataFrame(bht_l)[["liga","ht_goalless_n","btts_si_ht0","over25_si_ht0","avg_goles_si_ht0",
    "ht_con_gol_n","btts_si_ht1p","avg_goles_si_ht1p"]])

print("\n=== equipos donde mas se da BTTS viniendo de HT 0-0 ===")
print_tabulate(stats_btts_ht_by_team(df))

print("\n=== equipos cuyo 0-0 al HT explota en goles en el ST ===")
print_tabulate(stats_teams_explosion_after_00ht(df))

print("\n=== equipos que con mas frecuencia van ganando al descanso ===")
print_tabulate(stats_teams_winning_at_ht(df))


# analisis de remontadas, con foco en favoritos y Barcelona

print(f"\n{'='*70}\n  REMONTADAS: FAVORITOS Y BARCELONA\n{'='*70}")

print("\n=== remontadas segun si el equipo era favorito o no ===")
print_tabulate(stats_comeback_by_fav(df))

print("\n=== remontadas del favorito por liga ===")
for liga in ligas:
    print(f"\n  {liga}:"); print_tabulate(stats_comeback_by_fav(df[df["Div"]==liga]))

print("\n=== barcelona: analisis completo de remontadas ===")
barca_stats = stats_barcelona_remontadas(df)
for k,v in barca_stats.items():
    print(f"  {k}: {v}")

print("\n=== barcelona: partidos donde iba perdiendo al HT (detalle) ===")
barca_detail = stats_barcelona_remontadas_detail(df)
if not barca_detail.empty:
    print_tabulate(barca_detail)
    print(f"\n  resumen barcelona perdiendo HT:")
    print(f"  total: {len(barca_detail)} | remonto: {barca_detail['remonto'].sum()} ({round(barca_detail['remonto'].mean()*100,2)}%)")
    print(f"  por temporada:")
    print_tabulate(barca_detail.groupby("Season_label").agg(
        veces_perdiendo_ht=("remonto","count"),
        remontadas=("remonto","sum"),
        pct_remonta=("remonto","mean"),
    ).round(3).reset_index())
else:
    print("  Barcelona no aparece en el dataset (puede ser que no este en La Liga en este periodo)")


# comparativa entre el arranque y el cierre de cada temporada

print(f"\n{'='*70}\n  INICIO VS FINAL DE TEMPORADA\n{'='*70}")

print("\n=== comparativa inicio vs final de temporada por liga ===")
thirds = stats_season_thirds(df)
if not thirds.empty:
    pivot = thirds.pivot_table(
        index=["liga","temporada"],
        columns="segmento",
        values=["pct_H","pct_D","pct_A","avg_goles","pct_btts"]
    ).round(2)
    print_tabulate(thirds.sort_values(["liga","temporada","segmento"]))

print("\n=== diferencia inicio vs final (pct_H): quien mas cambia ===")
if not thirds.empty:
    ini = thirds[thirds["segmento"]=="inicio"][["liga","temporada","pct_H","avg_goles"]].rename(columns={"pct_H":"pct_H_ini","avg_goles":"goles_ini"})
    fin = thirds[thirds["segmento"]=="final"][["liga","temporada","pct_H","avg_goles"]].rename(columns={"pct_H":"pct_H_fin","avg_goles":"goles_fin"})
    comp = ini.merge(fin,on=["liga","temporada"])
    comp["delta_H"]     = round(comp["pct_H_fin"]-comp["pct_H_ini"],2)
    comp["delta_goles"] = round(comp["goles_fin"]-comp["goles_ini"],3)
    print_tabulate(comp.sort_values("delta_H"))

print("\n=== equipos top3: rendimiento inicio vs final de temporada ===")
top_thirds = stats_top_teams_season_thirds(df)
if not top_thirds.empty:
    ini_t = top_thirds[top_thirds["segmento"]=="inicio"][["liga","temporada","equipo","pj","pts","pct_vic","avg_gf"]].rename(columns={"pj":"pj_i","pts":"pts_i","pct_vic":"vic_i","avg_gf":"gf_i"})
    fin_t = top_thirds[top_thirds["segmento"]=="final"][["liga","temporada","equipo","pj","pts","pct_vic","avg_gf"]].rename(columns={"pj":"pj_f","pts":"pts_f","pct_vic":"vic_f","avg_gf":"gf_f"})
    cmp_t = ini_t.merge(fin_t,on=["liga","temporada","equipo"],how="outer")
    cmp_t["delta_vic"]=round(cmp_t["vic_f"]-cmp_t["vic_i"],2)
    cmp_t["delta_gf"] =round(cmp_t["gf_f"]-cmp_t["gf_i"],3)
    print("  equipos que mas bajan al final (delta_vic negativo = rinden menos):")
    print_tabulate(cmp_t.sort_values("delta_vic").head(15))
    print("  equipos que mas suben al final:")
    print_tabulate(cmp_t.sort_values("delta_vic",ascending=False).head(15))


# rendimiento por rango de cuota y analisis de valores especiales

print(f"\n{'='*70}\n  CUOTAS — RENDIMIENTO POR RANGO Y VALORES ESPECIALES\n{'='*70}")

print("\n=== rendimiento real vs implied por rango de cuota (apertura vs cierre) ===")
orp = stats_odds_range_performance(df)
print("  local — apertura:")
print_tabulate(orp[(orp["resultado"]=="local")&(orp["tipo"]=="apertura")].drop(columns=["resultado","tipo"]))
print("  local — cierre:")
print_tabulate(orp[(orp["resultado"]=="local")&(orp["tipo"]=="cierre")].drop(columns=["resultado","tipo"]))
print("  visitante — apertura:")
print_tabulate(orp[(orp["resultado"]=="visitante")&(orp["tipo"]=="apertura")].drop(columns=["resultado","tipo"]))
print("  visitante — cierre:")
print_tabulate(orp[(orp["resultado"]=="visitante")&(orp["tipo"]=="cierre")].drop(columns=["resultado","tipo"]))

print("\n=== apertura vs cierre: que cuota conviene mas para apostar al local ===")
loc_ap = orp[(orp["resultado"]=="local")&(orp["tipo"]=="apertura")][["rango_odd","n","imp_%","real_%","edge"]].rename(columns={"n":"n_ap","real_%":"real_ap","edge":"edge_ap"})
loc_ci = orp[(orp["resultado"]=="local")&(orp["tipo"]=="cierre")][["rango_odd","real_%","edge"]].rename(columns={"real_%":"real_ci","edge":"edge_ci"})
cmp_ac = loc_ap.merge(loc_ci,on="rango_odd",how="outer")
cmp_ac["mejor"]= cmp_ac.apply(lambda r: "cierre" if r.get("edge_ci",0)>r.get("edge_ap",0) else "apertura",axis=1)
print_tabulate(cmp_ac)

print("\n=== cuotas gematricas / valores especiales: anomalias de mercado ===")
gm = stats_gematric_odds(df)
if not gm.empty:
    print_tabulate(gm)
    print("\n  cuotas con edge positivo para el local:")
    print_tabulate(gm[(gm["lado"]=="local")&(gm["edge"]>0)].sort_values("edge",ascending=False))
    print("  cuotas con edge positivo para el visitante:")
    print_tabulate(gm[(gm["lado"]=="visitante")&(gm["edge"]>0)].sort_values("edge",ascending=False))

print("\n=== cuotas gematricas por liga ===")
for liga in ligas:
    gm_l = stats_gematric_odds(df[df["Div"]==liga])
    if not gm_l.empty:
        print(f"\n  {liga}:")
        print_tabulate(gm_l[gm_l["n"]>=5])

print(f"\n{'='*70}")
print("  ROI APUESTA PLANA A CUOTAS ESPECIALES (3.33 / +333 / -333 / etc.)")
print(f"{'='*70}")
print("  Interpretacion: roi_% > 0 = estrategia rentable | edge_% > 0 = mercado infravalorado\n")

roi_esp = stats_bet_roi_especial(df)
if not roi_esp.empty:
    print("\n  Tabla completa (apertura + cierre, los 3 resultados):")
    print_tabulate(roi_esp)

    pos_roi = roi_esp[roi_esp["roi_%"] > 0].sort_values("roi_%", ascending=False)
    print("\n  Cuotas CON ROI positivo (todo el dataset):")
    if not pos_roi.empty:
        print_tabulate(pos_roi)
    else:
        print("  (ninguna cuota especial tiene ROI positivo en el dataset completo)")

    print("\n  Resumen apertura local:")
    local_ap = roi_esp[roi_esp["lado"] == "local_ap"][
        ["cuota","decimal","n","wins","real_%","imp_%","edge_%","roi_%","yield"]
    ].sort_values("decimal").reset_index(drop=True)
    if not local_ap.empty:
        print_tabulate(local_ap)

    print("\n  Resumen apertura empate:")
    draw_ap = roi_esp[roi_esp["lado"] == "empate_ap"][
        ["cuota","decimal","n","wins","real_%","imp_%","edge_%","roi_%","yield"]
    ].sort_values("decimal").reset_index(drop=True)
    if not draw_ap.empty:
        print_tabulate(draw_ap)

    print("\n  Resumen apertura visitante:")
    away_ap = roi_esp[roi_esp["lado"] == "visitante_ap"][
        ["cuota","decimal","n","wins","real_%","imp_%","edge_%","roi_%","yield"]
    ].sort_values("decimal").reset_index(drop=True)
    if not away_ap.empty:
        print_tabulate(away_ap)

print("\n=== ROI cuotas especiales por liga ===")
for liga in ligas:
    roi_l = stats_bet_roi_especial(df[df["Div"] == liga])
    if roi_l.empty:
        continue
    print(f"\n  {LIGAS_NAME[liga]}:")
    sub_l = roi_l[roi_l["lado"].isin(["local_ap","empate_ap","visitante_ap"])][
        ["cuota","lado","n","real_%","imp_%","edge_%","roi_%"]
    ].reset_index(drop=True)
    if not sub_l.empty:
        print_tabulate(sub_l)

print(f"\n{'='*70}")
print("  MODELO POISSON — PREDICCION DE MARCADORES Y PROBABILIDADES")
print(f"{'='*70}")
print("""
  El modelo Poisson estima lambda_home y lambda_away para cada partido
  usando el ataque y defensa historicos de cada equipo relativizados
  por el promedio de goles de la liga (referencia de rendimiento).

  Formula:
    lambda_H = mu_H_liga * ataque_local * defensa_visitante
    lambda_A = mu_A_liga * ataque_visitante * defensa_local

  Donde ataque = (goles_marcados / partidos) / mu_liga
        defensa = (goles_recibidos / partidos) / mu_rival_liga

  P(marcador i-j) = Poisson(i; lambda_H) * Poisson(j; lambda_A)
  P(H)  = suma de P(i-j) con i > j
  P(D)  = suma de P(i-j) con i == j
  P(A)  = suma de P(i-j) con i < j
""")

# un clasico por liga para mostrar el modelo en accion
ejemplos_poisson = [
    ("E0",  "Man City",    "Arsenal"),
    ("SP1", "Real Madrid", "Barcelona"),
    ("D1",  "Bayern Munich","Dortmund"),
    ("I1",  "Inter",       "Juventus"),
    ("F1",  "Paris SG",    "Marseille"),
]

print("\n=== Poisson: ejemplos por liga (clásicos) ===")
for liga, home, away in ejemplos_poisson:
    res = poisson_model(df, home, away, liga=liga)
    print_poisson(res)

print("\n=== Poisson: todos los equipos de la liga (ranking de ataque y defensa) ===")
for liga in ligas:
    sub_liga = df[df["Div"] == liga]
    mu_h = sub_liga["FTHG"].mean()
    mu_a = sub_liga["FTAG"].mean()
    equipos_liga = sorted(sub_liga["HomeTeam"].unique())
    rows_rank = []
    for eq in equipos_liga:
        as_home = sub_liga[sub_liga["HomeTeam"] == eq]
        as_away = sub_liga[sub_liga["AwayTeam"] == eq]
        n_h = len(as_home); n_a = len(as_away)
        if n_h + n_a < 10:
            continue
        gf_h = as_home["FTHG"].sum(); gc_h = as_home["FTAG"].sum()
        gf_a = as_away["FTAG"].sum(); gc_a = as_away["FTHG"].sum()
        pj_h = max(n_h, 1); pj_a = max(n_a, 1)
        atk = round(((gf_h/pj_h) + (gf_a/pj_a)) / 2 / ((mu_h + mu_a) / 2), 3) if (mu_h+mu_a)>0 else 1.0
        dfn = round(((gc_h/pj_h) + (gc_a/pj_a)) / 2 / ((mu_h + mu_a) / 2), 3) if (mu_h+mu_a)>0 else 1.0
        rows_rank.append({
            "equipo":     eq,
            "pj":         n_h + n_a,
            "ataque":     atk,
            "defensa":    dfn,
            "gf_pg":      round((gf_h + gf_a) / (n_h + n_a), 3),
            "gc_pg":      round((gc_h + gc_a) / (n_h + n_a), 3),
            "poder":      round(atk - dfn, 3),
        })
    if not rows_rank:
        continue
    rank_df = pd.DataFrame(rows_rank).sort_values("poder", ascending=False).reset_index(drop=True)
    rank_df.insert(0, "pos", range(1, len(rank_df)+1))
    print(f"\n  {LIGAS_NAME[liga]} — ranking ataque/defensa/poder (todo el periodo):")
    print_tabulate(rank_df)

print("\n=== Poisson: matriz de marcadores completa — ejemplo Premier League ===")
res_ex = poisson_model(df, "Man City", "Arsenal", liga="E0")
if "error" not in res_ex:
    print(f"\n  {res_ex['home']} vs {res_ex['away']} — lambda_H={res_ex['lambda_h']} lambda_A={res_ex['lambda_a']}")
    print(f"  Matriz P(i-j) en % — filas=goles local, columnas=goles visitante:\n")
    max_show = 7
    header = ["loc\\vis"] + [str(j) for j in range(max_show+1)]
    mat_rows = []
    for i in range(max_show+1):
        row_data = [str(i)] + [f"{res_ex['matriz'][i,j]*100:.2f}%" for j in range(max_show+1)]
        mat_rows.append(row_data)
    print(tabulate(mat_rows, headers=header, tablefmt="orgtbl"))


# generacion de graficas


print("\n--- generando graficas ---")
cmap=get_cmap(len(ligas)+1)

draw_er_diagram("img/er_diagram.png")

scatter_group_by("img/odds_apertura_por_liga.png",    df,"AvgH","AvgA","Div")
scatter_group_by("img/goles_local_vs_visitante.png",  df,"FTHG","FTAG","Div")
scatter_group_by("img/odds_apertura_vs_cierre_H.png", df,"AvgH","AvgCH","Div")
scatter_group_by("img/odds_apertura_vs_cierre_A.png", df,"AvgA","AvgCA","Div")
scatter_group_by("img/implied_prob_H_vs_A.png",       df,"imp_prob_H","imp_prob_A","Div")
scatter_group_by("img/movimiento_mercado.png",         df,"odds_move_H","odds_move_A","Div")
scatter_group_by("img/ht_vs_ft_goles.png",            df,"ht_goals","total_goals","Div")

for col,ylabel,fname in [
    ("total_goals","goles promedio","goles_por_temporada_y_liga"),
    ("over25","% over 2.5","over25_por_temporada_y_liga"),
    ("btts","% btts","btts_por_temporada_y_liga"),
    ("home_win","% victoria local","pct_home_win_por_temporada_liga"),
    ("clean_sheet_h","% clean sheet local","clean_sheet_por_temporada_liga"),
]:
    fig,ax=plt.subplots(figsize=(12,6))
    for i,liga in enumerate(ligas):
        sub=df[df["Div"]==liga].groupby("Season")[col].mean()
        if col!="total_goals": sub=sub*100
        ax.plot(sub.index.astype(str),sub.values,marker="o",label=liga,color=cmap(i))
    ax.set_xlabel("temporada"); ax.set_ylabel(ylabel); ax.legend()
    plt.savefig(f"img/{fname}.png"); plt.close()

fig,axes=plt.subplots(1,5,figsize=(20,5))
for i,liga in enumerate(ligas):
    sub=df[df["Div"]==liga]["FTR"].value_counts()
    axes[i].pie(sub.values,labels=sub.index,autopct="%1.1f%%"); axes[i].set_title(liga)
plt.savefig("img/resultados_pie_por_liga.png"); plt.close()

fig,ax=plt.subplots(figsize=(13,5))
x=np.arange(len(ligas)); w=0.13
for j,(ml,col) in enumerate([("btts","btts"),("over1.5","over15"),("over2.5","over25"),
                               ("over3.5","over35"),("under2.5","under25"),("cs_h","clean_sheet_h")]):
    ax.bar(x+j*w,[df[df["Div"]==l][col].mean()*100 for l in ligas],w,label=ml)
ax.set_xticks(x+w*2.5); ax.set_xticklabels(ligas); ax.set_ylabel("%"); ax.legend()
plt.savefig("img/btts_over_under_cs_por_liga.png"); plt.close()

fig,axes=plt.subplots(1,5,figsize=(20,5))
for i,liga in enumerate(ligas):
    axes[i].hist(df[df["Div"]==liga]["total_goals"],bins=range(0,12),edgecolor="black",align="left")
    axes[i].set_title(liga); axes[i].set_xlabel("goles")
plt.savefig("img/distribucion_goles_por_liga.png"); plt.close()

fig,axes=plt.subplots(1,2,figsize=(14,5))
for i,liga in enumerate(ligas):
    axes[0].hist(df[df["Div"]==liga]["ht_goals"],bins=range(0,8),alpha=0.5,label=liga,align="left")
    axes[1].hist(df[df["Div"]==liga]["second_half_goals"],bins=range(0,8),alpha=0.5,label=liga,align="left")
axes[0].set_title("goles 1er tiempo"); axes[0].set_xlabel("goles HT"); axes[0].legend()
axes[1].set_title("goles 2do tiempo"); axes[1].set_xlabel("goles ST"); axes[1].legend()
plt.tight_layout(); plt.savefig("img/distribucion_ht_st_por_liga.png"); plt.close()

fig,ax=plt.subplots(figsize=(10,5))
for liga in ligas:
    ax.hist(df[df["Div"]==liga]["overround"],bins=40,alpha=0.5,label=liga)
ax.set_xlabel("overround"); ax.legend()
plt.savefig("img/overround_por_liga.png"); plt.close()

fig,ax=plt.subplots(figsize=(10,5))
ax.hist(df[(df["is_underdog_away"]==1)&(df["away_win"]==1)]["AvgA"],bins=30,alpha=0.6,label="gana")
ax.hist(df[(df["is_underdog_away"]==1)&(df["away_win"]==0)]["AvgA"],bins=30,alpha=0.6,label="pierde")
ax.set_xlabel("AvgA"); ax.legend(); ax.set_title("odds underdogs: gana vs pierde")
plt.savefig("img/underdog_win_vs_lose.png"); plt.close()

fig,ax=plt.subplots(figsize=(10,5))
ax.hist(df["total_goals"],bins=range(0,15),edgecolor="black",align="left")
ax.set_xlabel("goles"); ax.set_ylabel("frecuencia")
plt.savefig("img/distribucion_goles_totales.png"); plt.close()

# grafica: como cambia el resultado final segun los goles del primer tiempo
fig,axes=plt.subplots(1,3,figsize=(18,5))
ht_groups=["HT=0","HT=1","HT=2","HT=3+"]
masks=[df["ht_goals"]==0,df["ht_goals"]==1,df["ht_goals"]==2,df["ht_goals"]>=3]
avg_ft=[df[m]["total_goals"].mean() for m in masks]
pct_bt=[df[m]["btts"].mean()*100 for m in masks]
pct_5p=[df[m]["high_scoring"].mean()*100 for m in masks]
axes[0].bar(ht_groups,avg_ft,color="#3498db",alpha=0.8)
axes[0].set_title("avg goles FT segun HT"); axes[0].set_ylabel("avg goles FT")
axes[1].bar(ht_groups,pct_bt,color="#e74c3c",alpha=0.8)
axes[1].set_title("% btts segun HT"); axes[1].set_ylabel("% btts")
axes[2].bar(ht_groups,pct_5p,color="#2ecc71",alpha=0.8)
axes[2].set_title("% 5+ goles segun HT"); axes[2].set_ylabel("% 5+")
plt.tight_layout(); plt.savefig("img/ht_vs_ft_analisis.png"); plt.close()

# grafica: relacion entre el equilibrio de cuotas y los goles del partido
gap_df=stats_odds_gap(df)
fig,axes=plt.subplots(1,3,figsize=(18,5))
axes[0].bar(gap_df["equilibrio"],gap_df["avg_goles"],color="#3498db",alpha=0.8)
axes[0].set_title("avg goles segun equilibrio cuotas"); axes[0].tick_params(axis="x",rotation=45)
axes[1].bar(gap_df["equilibrio"],gap_df["pct_btts"],color="#e74c3c",alpha=0.8)
axes[1].set_title("% btts segun equilibrio"); axes[1].tick_params(axis="x",rotation=45)
axes[2].bar(gap_df["equilibrio"],gap_df["pct_D"],color="#9b59b6",alpha=0.8)
axes[2].set_title("% empate segun equilibrio"); axes[2].tick_params(axis="x",rotation=45)
plt.tight_layout(); plt.savefig("img/odds_gap_analisis.png"); plt.close()

# grafica: goles promedio y otros indicadores segun el dia de la semana
day_plot=stats_day_of_week(df)
fig,ax=plt.subplots(figsize=(11,5))
dp=day_plot.set_index("day_name")
ax.bar(dp.index,dp["avg_goles"],color="#3498db",alpha=0.8,label="avg goles")
ax2=ax.twinx()
ax2.plot(dp.index,dp["pct_btts"],marker="o",color="#e74c3c",label="% btts")
ax2.plot(dp.index,dp["pct_over25"],marker="s",color="#2ecc71",label="% over2.5")
ax.set_ylabel("goles promedio"); ax2.set_ylabel("%")
ax2.legend(loc="upper right"); ax.legend(loc="upper left")
plt.title("goles y resultados por dia de la semana")
plt.tight_layout(); plt.savefig("img/goles_por_dia_semana.png"); plt.close()

# grafica: como influye la racha previa del equipo local en el resultado
streak_df=stats_streak_effect(df)
fig,axes=plt.subplots(1,2,figsize=(14,5))
axes[0].bar(streak_df["streak_cat"],streak_df["pct_H"],color="#3498db",alpha=0.8)
axes[0].set_title("% victoria local segun racha previa"); axes[0].set_ylabel("% H")
axes[1].bar(streak_df["streak_cat"],streak_df["avg_goles"],color="#e74c3c",alpha=0.8)
axes[1].set_title("avg goles segun racha previa"); axes[1].set_ylabel("avg goles")
plt.tight_layout(); plt.savefig("img/efecto_racha.png"); plt.close()

print("guardadas imagenes en img/")

# grafica: marcadores exactos mas frecuentes en todo el periodo
scores_all = stats_exact_scores(df, top_n=12)
fig, ax = plt.subplots(figsize=(11,5))
ax.barh(scores_all["marcador"][::-1], scores_all["pct"][::-1], color="#3498db", alpha=0.85)
ax.set_xlabel("% del total de partidos")
ax.set_title("Marcadores exactos mas frecuentes (todo el periodo)")
for i,(v,p) in enumerate(zip(scores_all["marcador"][::-1], scores_all["pct"][::-1])):
    ax.text(p+0.05, i, f"{p}%", va="center", fontsize=8)
plt.tight_layout(); plt.savefig("img/marcadores_exactos.png"); plt.close()

# grafica: distribucion de la diferencia de goles por partido
gdiff = stats_goal_diff_distribution(df)
gdiff = gdiff[gdiff["diff_goles"] <= 6]
fig, ax = plt.subplots(figsize=(9,5))
ax.bar(gdiff["diff_goles"].astype(str), gdiff["pct"], color="#2ecc71", alpha=0.85, edgecolor="white")
ax.set_xlabel("diferencia de goles (valor absoluto)"); ax.set_ylabel("% partidos")
ax.set_title("Distribucion de diferencia de goles en el resultado final")
for i,(v,p) in enumerate(zip(gdiff["diff_goles"], gdiff["pct"])):
    ax.text(i, p+0.2, f"{p}%", ha="center", fontsize=8)
plt.tight_layout(); plt.savefig("img/diferencia_goles_ft.png"); plt.close()

# grafica: tasa de sorpresas segun que tan dispares eran las cuotas
upset_df = stats_upset_rate(df)
if not upset_df.empty:
    fig, ax = plt.subplots(figsize=(10,5))
    x = np.arange(len(upset_df)); w = 0.28
    ax.bar(x - w, upset_df["pct_fav_gana"], w, label="favorito gana", color="#3498db", alpha=0.85)
    ax.bar(x,     upset_df["pct_upset"],    w, label="sorpresa",       color="#e74c3c", alpha=0.85)
    ax.bar(x + w, upset_df["pct_empate"],   w, label="empate",         color="#95a5a6", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(upset_df["dif_cuotas"], rotation=30, ha="right")
    ax.set_ylabel("%"); ax.legend()
    ax.set_title("Favorito gana vs sorpresa vs empate segun diferencia de cuotas")
    plt.tight_layout(); plt.savefig("img/upset_rate_por_dif_cuotas.png"); plt.close()

# grafica: evolucion del overround a lo largo del tiempo
oe = stats_overround_evolution(df)
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(oe["temporada"], oe["avg_overround"], marker="o", color="#8e44ad", linewidth=2)
ax.fill_between(oe["temporada"],
    oe["avg_overround"] - oe["std_overround"],
    oe["avg_overround"] + oe["std_overround"],
    alpha=0.15, color="#8e44ad")
ax.set_xlabel("temporada"); ax.set_ylabel("overround promedio")
ax.set_title("Evolucion del overround por temporada (margen de la casa)")
plt.tight_layout(); plt.savefig("img/overround_evolucion.png"); plt.close()

# grafica: si la ventaja de jugar en casa se esta perdiendo con los anos
had = stats_home_advantage_decay(df)
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(had["Season_label"], had["pct_H"],  marker="o", label="% H",  color="#3498db", linewidth=2)
ax.plot(had["Season_label"], had["pct_D"],  marker="s", label="% D",  color="#95a5a6", linewidth=2)
ax.plot(had["Season_label"], had["pct_A"],  marker="^", label="% A",  color="#e74c3c", linewidth=2)
ax.set_xlabel("temporada"); ax.set_ylabel("%"); ax.legend()
ax.set_title("Evolucion de resultados por temporada (ventaja local)")
plt.tight_layout(); plt.savefig("img/ventaja_local_evolucion.png"); plt.close()

# grafica: perfil de la cuota local por liga
pfav = stats_goles_by_fav_profile(df)
if not pfav.empty:
    fig, axes = plt.subplots(1,3,figsize=(16,5))
    axes[0].bar(pfav["perfil"], pfav["avg_total"],  color="#3498db", alpha=0.85)
    axes[0].set_title("avg goles segun perfil cuota local"); axes[0].tick_params(axis="x",rotation=20)
    axes[1].bar(pfav["perfil"], pfav["pct_btts"],   color="#e74c3c", alpha=0.85)
    axes[1].set_title("% btts segun perfil cuota"); axes[1].tick_params(axis="x",rotation=20)
    axes[2].bar(pfav["perfil"], pfav["pct_over25"], color="#2ecc71", alpha=0.85)
    axes[2].set_title("% over2.5 segun perfil cuota"); axes[2].tick_params(axis="x",rotation=20)
    plt.tight_layout(); plt.savefig("img/goles_por_perfil_cuota.png"); plt.close()

# grafica: matriz de transicion de resultado al descanso a resultado final
hft = stats_ht_to_ft_matrix(df)
if not hft.empty:
    fig, ax = plt.subplots(figsize=(6,4))
    hft_mat = hft.set_index("HTR")[["pct_H","pct_D","pct_A"]].rename(columns={"pct_H":"FT=H","pct_D":"FT=D","pct_A":"FT=A"})
    im = ax.imshow(hft_mat.values, cmap="Blues", vmin=0, vmax=80)
    ax.set_xticks(range(3)); ax.set_xticklabels(hft_mat.columns)
    ax.set_yticks(range(len(hft_mat))); ax.set_yticklabels(hft_mat.index)
    for i in range(len(hft_mat)):
        for j in range(3):
            ax.text(j, i, f"{hft_mat.values[i,j]}%", ha="center", va="center", fontsize=11)
    ax.set_xlabel("resultado FT"); ax.set_ylabel("resultado HT")
    ax.set_title("Matriz de transicion HTR → FTR (%)")
    plt.colorbar(im, ax=ax); plt.tight_layout()
    plt.savefig("img/matriz_htr_ftr.png"); plt.close()

# grafica: ratio goles local vs visitante por liga y temporada
hlr = stats_home_local_ratio(df)
fig, ax = plt.subplots(figsize=(12,5))
cmap = get_cmap(len(ligas)+1)
for i, liga in enumerate(ligas):
    sub = hlr[hlr["Div"]==liga]
    ax.plot(sub["Season_label"], sub["ratio"], marker="o", label=liga, color=cmap(i), linewidth=2)
ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
ax.set_xlabel("temporada"); ax.set_ylabel("ratio goles local / visitante")
ax.set_title("Evolucion del ratio goles local/visitante por liga")
ax.legend(); plt.tight_layout(); plt.savefig("img/ratio_goles_local_visitante.png"); plt.close()

# grafica: entropia de resultados por liga
ent_df = stats_entropy_results(df)
fig, ax = plt.subplots(figsize=(8,4))
bars = ax.bar(ent_df["liga"], ent_df["entropia"], color="#9b59b6", alpha=0.85)
ax.axhline(float(ent_df["max_posible"].iloc[0]), color="gray", linestyle="--", linewidth=0.8)
ax.set_ylabel("entropia (bits)"); ax.set_ylim(0, 1.7)
ax.set_title("Entropia de resultados por liga (impredecibilidad)")
for bar, val in zip(bars, ent_df["entropia"]):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.01, f"{val}", ha="center", fontsize=9)
plt.tight_layout(); plt.savefig("img/entropia_por_liga.png"); plt.close()

# grafica: calibracion del mercado para el empate
draw_rng = stats_draw_by_odd_range(df)
if not draw_rng.empty:
    fig, ax = plt.subplots(figsize=(10,5))
    x = np.arange(len(draw_rng)); w = 0.35
    ax.bar(x-w/2, draw_rng["imp_%"],  w, label="implied %",  color="#3498db", alpha=0.85)
    ax.bar(x+w/2, draw_rng["real_%"], w, label="real %",     color="#e74c3c", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(draw_rng["rango_AvgD"], rotation=35, ha="right")
    ax.set_ylabel("%"); ax.legend()
    ax.set_title("Empate: probabilidad implicita vs resultado real por rango de cuota")
    plt.tight_layout(); plt.savefig("img/empate_implied_vs_real.png"); plt.close()

print("guardadas imagenes nuevas en img/")

# grafica: top 20 equipos con las rachas ganadoras mas largas
streak_all = stats_max_streaks_all(df).head(20)
fig, ax = plt.subplots(figsize=(14,6))
x = np.arange(len(streak_all)); w = 0.28
ax.bar(x-w, streak_all["max_racha_vic"], w, label="victorias", color="#3498db", alpha=0.85)
ax.bar(x,   streak_all["max_racha_der"], w, label="derrotas",  color="#e74c3c", alpha=0.85)
ax.bar(x+w, streak_all["max_racha_emp"], w, label="empates",   color="#95a5a6", alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(streak_all["equipo"], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("partidos consecutivos"); ax.legend()
ax.set_title("Rachas maximas por equipo (victorias / derrotas / empates)")
plt.tight_layout(); plt.savefig("img/rachas_maximas_equipos.png"); plt.close()

# grafica: btts segun como iba el marcador al descanso
bht_rows=[]
for liga in ligas:
    b=stats_btts_by_ht_state(df[df["Div"]==liga]); b["liga"]=liga; bht_rows.append(b)
bht_df = pd.DataFrame(bht_rows)
fig, axes = plt.subplots(1,2,figsize=(14,5))
x = np.arange(len(ligas)); w = 0.35
axes[0].bar(x-w/2, bht_df["btts_si_ht0"],  w, label="HT sin goles", color="#3498db", alpha=0.85)
axes[0].bar(x+w/2, bht_df["btts_si_ht1p"], w, label="HT con goles", color="#e74c3c", alpha=0.85)
axes[0].set_xticks(x); axes[0].set_xticklabels(ligas)
axes[0].set_ylabel("% BTTS"); axes[0].set_title("BTTS segun estado del primer tiempo")
axes[0].legend()
axes[1].bar(x-w/2, bht_df["avg_goles_si_ht0"],  w, label="HT sin goles", color="#3498db", alpha=0.85)
axes[1].bar(x+w/2, bht_df["avg_goles_si_ht1p"], w, label="HT con goles", color="#e74c3c", alpha=0.85)
axes[1].set_xticks(x); axes[1].set_xticklabels(ligas)
axes[1].set_ylabel("avg goles FT"); axes[1].set_title("Avg goles FT segun estado del HT")
axes[1].legend()
plt.tight_layout(); plt.savefig("img/btts_segun_estado_ht.png"); plt.close()

# grafica: cuantos goles meten en el segundo tiempo los equipos que van 0-0 al descanso
exp_df = stats_teams_explosion_after_00ht(df).head(15)
if not exp_df.empty:
    fig, ax = plt.subplots(figsize=(12,5))
    ax.barh(exp_df["equipo"][::-1], exp_df["avg_goles_ft"][::-1], color="#e67e22", alpha=0.85)
    ax.set_xlabel("avg goles FT cuando va 0-0 al HT")
    ax.set_title("Equipos cuyos partidos 0-0 al HT explotan en goles (FT)")
    for i,(e,v) in enumerate(zip(exp_df["equipo"][::-1], exp_df["avg_goles_ft"][::-1])):
        ax.text(v+0.01, i, f"{v}", va="center", fontsize=8)
    plt.tight_layout(); plt.savefig("img/explosion_00ht_por_equipo.png"); plt.close()

# grafica: diferencias entre el inicio y el final de temporada
thirds = stats_season_thirds(df)
if not thirds.empty:
    ini = thirds[thirds["segmento"]=="inicio"].groupby("liga")["pct_H"].mean()
    fin = thirds[thirds["segmento"]=="final"].groupby("liga")["pct_H"].mean()
    fig, ax = plt.subplots(figsize=(10,5))
    x = np.arange(len(ligas)); w = 0.35
    ini_vals = [ini.get(l,0) for l in ligas]
    fin_vals  = [fin.get(l,0) for l in ligas]
    ax.bar(x-w/2, ini_vals, w, label="inicio temporada", color="#3498db", alpha=0.85)
    ax.bar(x+w/2, fin_vals, w, label="final temporada",  color="#e74c3c", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(ligas)
    ax.set_ylabel("% victoria local"); ax.legend()
    ax.set_title("Victoria local: inicio vs final de temporada por liga")
    plt.tight_layout(); plt.savefig("img/inicio_vs_final_temporada.png"); plt.close()

# grafica: edge real por cuota especial
gm = stats_gematric_odds(df)
if not gm.empty:
    gm_loc = gm[gm["lado"]=="local"].copy()
    fig, ax = plt.subplots(figsize=(13,5))
    colors = ["#2ecc71" if e>0 else "#e74c3c" for e in gm_loc["edge"]]
    bars = ax.bar(gm_loc["cuota_lbl"], gm_loc["edge"], color=colors, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("cuota"); ax.set_ylabel("edge (real% - implied%)")
    ax.set_title("Edge en cuotas especiales/gematricas — lado local")
    ax.tick_params(axis="x", rotation=35)
    plt.tight_layout(); plt.savefig("img/cuotas_gematricas_edge.png"); plt.close()

print("guardadas imagenes finales en img/")

# ejecucion completa del analisis de rachas binarias

print(f"\n{'='*70}")
print(f"  RACHAS BINARIAS — MOTOR COMPLETO")
print(f"  (over/under, btts, goles HT/ST, cuotas altas, CS, resultados)")
print(f"{'='*70}")

# construccion de flags y ejecucion del motor
binary_results, df_ext = run_all_binary_streaks(df)

# 1. resumen global de todas las rachas
print("\n=== RESUMEN GLOBAL: max racha ON y OFF por flag ===")
print_tabulate(streak_summary_table(binary_results))

# 2. reporte por grupos tematicos

# grupo A: resultados a tiempo completo
print(f"\n{'─'*70}")
print("  GRUPO A — RESULTADOS FULL TIME")
print(f"{'─'*70}")
print_binary_streak_report(binary_results, [
    "over25","over35","over15","over45",
    "under25","under15",
    "btts","no_btts",
    "goalless","high_scoring",
    "home_win","away_win","draw",
])

# grupo B: primer tiempo
print(f"\n{'─'*70}")
print("  GRUPO B — PRIMER TIEMPO (HT)")
print(f"{'─'*70}")
print_binary_streak_report(binary_results, [
    "gol_ht","under05_ht","under15_ht","over15_ht",
    "btts_ht","no_btts_ht",
])

# grupo C: segundo tiempo
print(f"\n{'─'*70}")
print("  GRUPO C — SEGUNDO TIEMPO (ST)")
print(f"{'─'*70}")
print_binary_streak_report(binary_results, [
    "gol_st","under05_st","under15_st","over15_st","over25_st",
    "btts_st","no_btts_st","st_mas_ht",
])

# grupo D: cuotas altas y resultados sorpresa
print(f"\n{'─'*70}")
print("  GRUPO D — CUOTAS ALTAS Y SORPRESAS")
print(f"{'─'*70}")
print_binary_streak_report(binary_results, [
    "high_odd_30_win","high_odd_40_win","high_odd_50_win","high_odd_60_win",
    "high_draw_win","extreme_upset","balanced_high",
])

# grupo E: porterias a cero
print(f"\n{'─'*70}")
print("  GRUPO E — CLEAN SHEETS")
print(f"{'─'*70}")
print_binary_streak_report(binary_results, [
    "clean_sheet_h","clean_sheet_a",
])

# 3. correlaciones cruzadas entre flags
print(f"\n{'='*70}")
print("  CORRELACIONES CRUZADAS ENTRE RACHAS")
print(f"{'='*70}")
print_tabulate(streak_cross_flags(df))

# 4. cuotas altas en contexto de racha
print(f"\n{'='*70}")
print("  CUOTAS ALTAS: ¿TRAS N SORPRESAS CONSECUTIVAS, HAY OTRA?")
print(f"{'='*70}")
print_tabulate(streak_odds_high_analysis(df))

# 5. rachas maximas por liga para los flags principales
FLAGS_CLAVE = [
    ("over25",       "over2.5_FT"),
    ("btts",         "btts_FT"),
    ("under25",      "under2.5_FT"),
    ("goalless",     "0-0_FT"),
    ("gol_ht",       "gol_en_HT"),
    ("under05_ht",   "0goles_HT"),
    ("gol_st",       "gol_en_ST"),
    ("high_odd_30_win","cuota_alta_gana"),
]

print(f"\n{'='*70}")
print("  RACHAS MÁXIMAS POR LIGA — FLAGS CLAVE")
print(f"{'='*70}")
for flag, label in FLAGS_CLAVE:
    if flag not in df_ext.columns: continue
    print(f"\n  [{label}]")
    print_tabulate(streak_max_by_league(df_ext, flag, label))

# 6. rachas maximas por temporada
print(f"\n{'='*70}")
print("  RACHAS MÁXIMAS POR TEMPORADA — FLAGS CLAVE")
print(f"{'='*70}")
for flag, label in FLAGS_CLAVE:
    if flag not in df_ext.columns: continue
    rows_t = []
    for slbl in sorted(df["Season_label"].dropna().unique()):
        sub = df_ext[df_ext["Season_label"] == slbl].sort_values("Date")
        if len(sub) < 10: continue
        vals = sub[flag].tolist()
        mx_on = 0; mx_off = 0; c_on = 0; c_off = 0
        for v in vals:
            if v == 1:
                c_on += 1; c_off = 0; mx_on = max(mx_on, c_on)
            else:
                c_off += 1; c_on = 0; mx_off = max(mx_off, c_off)
        rows_t.append({
            "temporada": slbl,
            "max_racha_ON":  mx_on,
            "max_racha_OFF": mx_off,
            f"pct_{flag}":   round(sub[flag].mean() * 100, 2),
            "partidos":      len(sub),
        })
    if rows_t:
        print(f"\n  [{label}]")
        print_tabulate(pd.DataFrame(rows_t))

# 7. equipos con rachas mas largas en los flags clave
print(f"\n{'='*70}")
print("  TOP EQUIPOS CON RACHAS MÁS LARGAS — FLAGS CLAVE")
print(f"{'='*70}")
for flag, label in FLAGS_CLAVE:
    if flag not in df_ext.columns: continue
    print(f"\n  [{label}]")
    print_tabulate(streak_max_by_team(df_ext, flag, label).head(12))

# 8. perfil de equipos calientes y frios en btts y over25
print(f"\n{'='*70}")
print("  PERFIL EQUIPOS CALIENTES / FRÍOS (racha actual ≥3)")
print(f"{'='*70}")
for flag, label in [("btts","btts_FT"), ("over25","over2.5_FT"),
                     ("gol_ht","gol_HT"), ("clean_sheet_h","cs_local")]:
    if flag not in df_ext.columns: continue
    print(f"\n  [{label}] — equipos con racha activa más larga:")
    print_tabulate(streak_team_flag_profile(df_ext, flag, label).head(10))

# 9. estado actual: que rachas historicas siguen activas
print(f"\n{'='*70}")
print("  ESTADO VIVO AL FINAL DEL DATASET — TODOS LOS FLAGS")
print(f"{'='*70}")
estado_rows = []
for flag, r in binary_results.items():
    est = r["estado_vivo"].copy()
    est.insert(0, "flag", r["label"])
    estado_rows.append(est)
if estado_rows:
    estado_all = pd.concat(estado_rows, ignore_index=True)
    # mostrar solo los que tienen racha activa larga (≥3)
    print("\n  rachas activas de 3+ partidos al final del dataset:")
    activas = estado_all[estado_all["partidos_activos"] >= 3].sort_values(
        "partidos_activos", ascending=False)
    if not activas.empty:
        print_tabulate(activas)
    else:
        print("  (ninguna racha de 3+ activa al cierre del dataset)")

print("\n=== fin bloque rachas binarias ===")