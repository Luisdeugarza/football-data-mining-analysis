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

# FUNCIONES AUXILIARES — ESTADISTICA DESCRIPTIVA


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

#  GOLES ---

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

#  CUOTAS ---

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

#  RESULTADOS / FLAGS / REMONTADAS --

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

#  UNDERDOGS / SMART MONEY -

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

#  DIAS / JORNADA --

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

#  RACHAS --

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

#  TABLA DE POSICIONES / SEGMENTOS -

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

#  GRAFICAS 


#  MARCADORES EXACTOS --

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

#  CUOTAS Y MERCADO EXTRA --

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

#  RACHAS DETALLADAS ---

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


#  BTTS SEGUN HT ---

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


#  0-0 HT -> EXPLOSION EN SegundoTiempo ---

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


#  EQUIPOS QUE YA GANAN AL HT -

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


#  REMONTADAS DEL FAVORITO + BARCELONA es mi equipo favorito y remontan mucho, lo cual tiene valor en mi opinio -

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


#  INICIO VS FIN DE TEMPORADA --

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


#  CUOTAS GEMATRICAS / NUMEROLOGIA 

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


#  CUOTAS: QUE RANGO SE GANA MAS (apert + cierre) -

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

# SETUP

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

# CICLO PRINCIPAL: temporada > liga

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

        # tabla posiciones + segmentos
        standing=calc_standings(df_tl)
        print(f"  tabla de posiciones {liga} {slabel}:")
        print_tabulate(standing.reset_index().rename(columns={"index":"pos"}))

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


# COMPARACION ENTRE TEMPORADAS

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


# COMPARACION ENTRE LIGAS

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


# ANALISIS GENERALES (todo el periodo, todas las ligas)

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


# RACHAS DETALLADAS

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

# variante mas permisiva
print("\n  [variante: racha>=2 derrotas/sin gana, golea >=3]")
bstg2 = stats_bad_streak_then_goleada(df, min_bad=2, min_goals=3)
if not bstg2.empty:
    print(f"  total: {len(bstg2)} | gana sig: {round(bstg2['gano_siguiente'].mean()*100,2)}% | pierde: {round(bstg2['perdio_siguiente'].mean()*100,2)}%")


# BTTS SEGUN ESTADO DEL PRIMER TIEMPO

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


# REMONTADAS: FAVORITOS + BARCELONA

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


# INICIO VS FIN DE TEMPORADA

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


# CUOTAS: RANGOS DE PERFORMANCE + VALORES GEMATRICOS

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


# GRAFICAS


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

# HT vs FT analisis grafico
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

# odds gap grafico
gap_df=stats_odds_gap(df)
fig,axes=plt.subplots(1,3,figsize=(18,5))
axes[0].bar(gap_df["equilibrio"],gap_df["avg_goles"],color="#3498db",alpha=0.8)
axes[0].set_title("avg goles segun equilibrio cuotas"); axes[0].tick_params(axis="x",rotation=45)
axes[1].bar(gap_df["equilibrio"],gap_df["pct_btts"],color="#e74c3c",alpha=0.8)
axes[1].set_title("% btts segun equilibrio"); axes[1].tick_params(axis="x",rotation=45)
axes[2].bar(gap_df["equilibrio"],gap_df["pct_D"],color="#9b59b6",alpha=0.8)
axes[2].set_title("% empate segun equilibrio"); axes[2].tick_params(axis="x",rotation=45)
plt.tight_layout(); plt.savefig("img/odds_gap_analisis.png"); plt.close()

# dias de la semana grafico
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

# racha efecto grafico
streak_df=stats_streak_effect(df)
fig,axes=plt.subplots(1,2,figsize=(14,5))
axes[0].bar(streak_df["streak_cat"],streak_df["pct_H"],color="#3498db",alpha=0.8)
axes[0].set_title("% victoria local segun racha previa"); axes[0].set_ylabel("% H")
axes[1].bar(streak_df["streak_cat"],streak_df["avg_goles"],color="#e74c3c",alpha=0.8)
axes[1].set_title("avg goles segun racha previa"); axes[1].set_ylabel("avg goles")
plt.tight_layout(); plt.savefig("img/efecto_racha.png"); plt.close()

print("guardadas imagenes en img/")

#  graficas: marcadores exactos (todo el periodo) 
scores_all = stats_exact_scores(df, top_n=12)
fig, ax = plt.subplots(figsize=(11,5))
ax.barh(scores_all["marcador"][::-1], scores_all["pct"][::-1], color="#3498db", alpha=0.85)
ax.set_xlabel("% del total de partidos")
ax.set_title("Marcadores exactos mas frecuentes (todo el periodo)")
for i,(v,p) in enumerate(zip(scores_all["marcador"][::-1], scores_all["pct"][::-1])):
    ax.text(p+0.05, i, f"{p}%", va="center", fontsize=8)
plt.tight_layout(); plt.savefig("img/marcadores_exactos.png"); plt.close()

#  graficas: diferencia de goles 
gdiff = stats_goal_diff_distribution(df)
gdiff = gdiff[gdiff["diff_goles"] <= 6]
fig, ax = plt.subplots(figsize=(9,5))
ax.bar(gdiff["diff_goles"].astype(str), gdiff["pct"], color="#2ecc71", alpha=0.85, edgecolor="white")
ax.set_xlabel("diferencia de goles (valor absoluto)"); ax.set_ylabel("% partidos")
ax.set_title("Distribucion de diferencia de goles en el resultado final")
for i,(v,p) in enumerate(zip(gdiff["diff_goles"], gdiff["pct"])):
    ax.text(i, p+0.2, f"{p}%", ha="center", fontsize=8)
plt.tight_layout(); plt.savefig("img/diferencia_goles_ft.png"); plt.close()

#  graficas: upset rate (tasa sorpresa) por diferencia de cuotas 
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

#  graficas: overround evolucion 
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

#  graficas: decaimiento ventaja local 
had = stats_home_advantage_decay(df)
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(had["Season_label"], had["pct_H"],  marker="o", label="% H",  color="#3498db", linewidth=2)
ax.plot(had["Season_label"], had["pct_D"],  marker="s", label="% D",  color="#95a5a6", linewidth=2)
ax.plot(had["Season_label"], had["pct_A"],  marker="^", label="% A",  color="#e74c3c", linewidth=2)
ax.set_xlabel("temporada"); ax.set_ylabel("%"); ax.legend()
ax.set_title("Evolucion de resultados por temporada (ventaja local)")
plt.tight_layout(); plt.savefig("img/ventaja_local_evolucion.png"); plt.close()

#  graficas: perfil cuota local 
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

#  graficas: matriz HTR -> FTR 
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

#  graficas: ratio goles local/visita por liga y temporada 
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

#  graficas: entropia por liga 
ent_df = stats_entropy_results(df)
fig, ax = plt.subplots(figsize=(8,4))
bars = ax.bar(ent_df["liga"], ent_df["entropia"], color="#9b59b6", alpha=0.85)
ax.axhline(float(ent_df["max_posible"].iloc[0]), color="gray", linestyle="--", linewidth=0.8)
ax.set_ylabel("entropia (bits)"); ax.set_ylim(0, 1.7)
ax.set_title("Entropia de resultados por liga (impredecibilidad)")
for bar, val in zip(bars, ent_df["entropia"]):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.01, f"{val}", ha="center", fontsize=9)
plt.tight_layout(); plt.savefig("img/entropia_por_liga.png"); plt.close()

#  graficas: empate real vs implied 
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

#  graficas: rachas maximas top 20 equipos 
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

#  graficas: btts segun estado HT por liga 
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

#  graficas: explosion 0-0 HT por equipo 
exp_df = stats_teams_explosion_after_00ht(df).head(15)
if not exp_df.empty:
    fig, ax = plt.subplots(figsize=(12,5))
    ax.barh(exp_df["equipo"][::-1], exp_df["avg_goles_ft"][::-1], color="#e67e22", alpha=0.85)
    ax.set_xlabel("avg goles FT cuando va 0-0 al HT")
    ax.set_title("Equipos cuyos partidos 0-0 al HT explotan en goles (FT)")
    for i,(e,v) in enumerate(zip(exp_df["equipo"][::-1], exp_df["avg_goles_ft"][::-1])):
        ax.text(v+0.01, i, f"{v}", va="center", fontsize=8)
    plt.tight_layout(); plt.savefig("img/explosion_00ht_por_equipo.png"); plt.close()

#  graficas: comparativa inicio vs final temporada 
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

#  graficas: cuotas gematricas — edge por valor 
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