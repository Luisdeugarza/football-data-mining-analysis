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
import os
from tabulate import tabulate

# resolví el problema de rutas relativas usando __file__ para que funcione
# desde cualquier directorio sin importar desde dónde se ejecute el script

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
SEASON_MAP = {
    1920: "2019/20", 2021: "2020/21", 2122: "2021/22",
    2223: "2022/23", 2324: "2023/24", 2425: "2024/25", 2526: "2025/26",
}

# cargué el dataset y construí todas las variables necesarias para los 11 análisis
# incluí odds derivadas, probabilidades implícitas, movimiento de mercado y bias

df = pd.read_csv(DATA_PATH, parse_dates=["Date"])

df["total_goals"]  = df["FTHG"] + df["FTAG"]
df["home_win"]     = (df["FTR"] == "H").astype(int)
df["draw"]         = (df["FTR"] == "D").astype(int)
df["away_win"]     = (df["FTR"] == "A").astype(int)
df["over25"]       = (df["total_goals"] > 2).astype(int)
df["btts"]         = ((df["FTHG"] > 0) & (df["FTAG"] > 0)).astype(int)
df["imp_prob_H"]   = round(1 / df["AvgH"], 4)
df["imp_prob_D"]   = round(1 / df["AvgD"], 4)
df["imp_prob_A"]   = round(1 / df["AvgA"], 4)
df["overround"]    = round(df["imp_prob_H"] + df["imp_prob_D"] + df["imp_prob_A"], 4)
df["odds_move_H"]  = round(df["AvgCH"] - df["AvgH"], 4)
df["odds_move_A"]  = round(df["AvgCA"] - df["AvgA"], 4)
df["odds_move_D"]  = round(df["AvgCD"] - df["AvgD"], 4)
df["bias_H"]       = round(df["imp_prob_H"] - df["home_win"], 4)
df["bias_A"]       = round(df["imp_prob_A"] - df["away_win"], 4)
df["bias_D"]       = round(df["imp_prob_D"] - df["draw"],     4)
df["Season_label"] = df["Season"].map(SEASON_MAP)
df["month"]        = df["Date"].dt.month
df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)

# marqué las semanas post fecha FIFA — son las semanas donde se juega
# inmediatamente después de un parón internacional (meses 3, 6, 9, 10, 11)

fifa_months     = [3, 6, 9, 10, 11]
df["post_fifa"] = df["month"].isin(fifa_months).astype(int)

# identifiqué underdogs visitantes como aquellos con cuota promedio mayor a 4.0

df["underdog_away"] = (df["AvgA"] > 4.0).astype(int)


# encapsulé la lógica de regresión en una función reutilizable para poder
# aplicarla sobre cualquier métrica y cualquier liga sin repetir código

def forecast_serie(data: pd.DataFrame, col_fecha: str, col_valor: str,
                   n_weeks: int = 12, freq: str = "W"):
    ts = (
        data.set_index(col_fecha)[col_valor]
        .resample(freq)
        .mean()
        .dropna()
        .reset_index()
    )
    ts.columns = ["Fecha", "valor"]
    ts["t"]    = range(len(ts))
    X          = sm.add_constant(ts["t"])
    model      = sm.OLS(ts["valor"], X).fit()
    m          = model.params["t"]
    b          = model.params["const"]
    t_last     = ts["t"].max()
    f_last     = ts["Fecha"].max()
    fut_t      = np.arange(t_last + 1, t_last + 1 + n_weeks)
    fut_dates  = pd.date_range(start=f_last + pd.Timedelta(weeks=1),
                               periods=n_weeks, freq=freq)
    df_fut = pd.DataFrame({
        "Fecha": fut_dates,
        "t":     fut_t,
        "pred":  np.round(m * fut_t + b, 4),
    })
    return model, ts, df_fut


# separé cada serie en 80% train / 20% test para validar el modelo
# sobre datos que nunca vio y obtener MAE y RMSE reales

def train_test_ols(ts: pd.DataFrame, test_pct: float = 0.20):
    split     = int(len(ts) * (1 - test_pct))
    train     = ts.iloc[:split].copy()
    test      = ts.iloc[split:].copy()
    X_train   = sm.add_constant(train["t"])
    model     = sm.OLS(train["valor"], X_train).fit()
    X_test    = sm.add_constant(test["t"])
    pred_test = model.predict(X_test)
    mae       = float(np.mean(np.abs(test["valor"].values - pred_test.values)))
    rmse      = float(np.sqrt(np.mean((test["valor"].values - pred_test.values) ** 2)))
    return model, train, test, pred_test, mae, rmse


ligas   = list(LIGAS_NAME.keys())
equipos = [
    ("Man City",      "E0",  "#6caddf"),
    ("Arsenal",       "E0",  "#ef0107"),
    ("Real Madrid",   "SP1", "#ffd700"),
    ("Barcelona",     "SP1", "#a50044"),
    ("Bayern Munich", "D1",  "#dc052d"),
    ("Juventus",      "I1",  "#000000"),
    ("Paris SG",      "F1",  "#004170"),
]


# analicé el % over 2.5 por liga como serie de tiempo semanal
# elegí esta métrica porque es la línea más transaccionada en el mercado europeo

print("1. over 2.5 por liga")

resumen_o25 = []
fig, axes   = plt.subplots(len(ligas), 1, figsize=(14, 4 * len(ligas)), sharex=False)

for i, liga in enumerate(ligas):
    sub = df[df["Div"] == liga].copy()
    model_tr, train, test, pred_test, mae, rmse = train_test_ols(
        forecast_serie(sub, "Date", "over25", n_weeks=16)[1]
    )
    model_full, ts_full, fut_full = forecast_serie(sub, "Date", "over25", n_weeks=16)
    m_f     = model_full.params["t"]
    fut_pct = fut_full["pred"] * 100
    col     = LIGA_COLORS[liga]
    resumen_o25.append({
        "Liga":          LIGAS_NAME[liga],
        "Media":         f"{ts_full['valor'].mean()*100:.1f}%",
        "Tendencia/sem": round(m_f * 100, 5),
        "R²":            round(model_full.rsquared, 4),
        "p-value":       round(model_full.pvalues["t"], 4),
        "MAE test":      round(mae * 100, 2),
        "RMSE test":     round(rmse * 100, 2),
        "Pred +16w":     f"{fut_pct.iloc[-1]:.1f}%",
    })
    ax = axes[i]
    ax.plot(ts_full["Fecha"], ts_full["valor"] * 100,
            color=col, alpha=0.45, linewidth=1, label="observado")
    ax.plot(train["Fecha"], model_tr.predict(sm.add_constant(train["t"])) * 100,
            color="red", linewidth=1.8, label=f"tendencia train  R²={model_tr.rsquared:.4f}")
    ax.plot(test["Fecha"], pred_test * 100,
            color="red", linewidth=1.8, linestyle="--")
    ax.scatter(test["Fecha"], test["valor"] * 100,
               color="black", s=12, zorder=5, label=f"test  MAE={mae*100:.1f}pp")
    ax.plot(fut_full["Fecha"], fut_pct, color="orange",
            linewidth=2, linestyle="--", label=f"pred +16w  {fut_pct.iloc[-1]:.1f}%")
    ax.axvline(test["Fecha"].iloc[0], color="gray", linestyle=":", linewidth=1)
    ax.set_ylabel("% over 2.5"); ax.legend(fontsize=7); ax.set_ylim(20, 90)
    ax.set_title(
        f"{LIGAS_NAME[liga]}  tendencia: {m_f*100:+.4f}pp/sem  R²={model_full.rsquared:.4f}  p={model_full.pvalues['t']:.4f}",
        fontweight="bold")
    print(f"  {LIGAS_NAME[liga]}: {m_f*100:+.5f}pp/sem  R²={model_full.rsquared:.4f}  p={model_full.pvalues['t']:.4f}  MAE={mae*100:.2f}pp  pred+16w={fut_pct.iloc[-1]:.1f}%")

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "forecasting_over25_por_liga.png"), dpi=130)
plt.close()
print("  guardado: img/forecasting_over25_por_liga.png\n")
print_tabulate(pd.DataFrame(resumen_o25))


# analicé el % de victorias local por liga para detectar si la home advantage
# tiene tendencia lineal — un descenso sostenido obligaría a reajustar cuotas

print("\n2. victorias local por liga")

resumen_hw = []
fig, axes  = plt.subplots(len(ligas), 1, figsize=(14, 4 * len(ligas)), sharex=False)

for i, liga in enumerate(ligas):
    sub = df[df["Div"] == liga].copy()
    model_tr, train, test, pred_test, mae, rmse = train_test_ols(
        forecast_serie(sub, "Date", "home_win", n_weeks=16)[1]
    )
    model_full, ts_full, fut_full = forecast_serie(sub, "Date", "home_win", n_weeks=16)
    m_f     = model_full.params["t"]
    fut_pct = fut_full["pred"] * 100
    col     = LIGA_COLORS[liga]
    resumen_hw.append({
        "Liga":          LIGAS_NAME[liga],
        "Media":         f"{ts_full['valor'].mean()*100:.1f}%",
        "Tendencia/sem": round(m_f * 100, 5),
        "R²":            round(model_full.rsquared, 4),
        "p-value":       round(model_full.pvalues["t"], 4),
        "MAE test":      round(mae * 100, 2),
        "RMSE test":     round(rmse * 100, 2),
        "Pred +16w":     f"{fut_pct.iloc[-1]:.1f}%",
    })
    ax = axes[i]
    ax.plot(ts_full["Fecha"], ts_full["valor"] * 100,
            color=col, alpha=0.45, linewidth=1, label="observado")
    ax.plot(train["Fecha"], model_tr.predict(sm.add_constant(train["t"])) * 100,
            color="red", linewidth=1.8, label=f"tendencia train  R²={model_tr.rsquared:.4f}")
    ax.plot(test["Fecha"], pred_test * 100,
            color="red", linewidth=1.8, linestyle="--")
    ax.scatter(test["Fecha"], test["valor"] * 100,
               color="black", s=12, zorder=5, label=f"test  MAE={mae*100:.1f}pp")
    ax.plot(fut_full["Fecha"], fut_pct, color="orange",
            linewidth=2, linestyle="--", label=f"pred +16w  {fut_pct.iloc[-1]:.1f}%")
    ax.axvline(test["Fecha"].iloc[0], color="gray", linestyle=":", linewidth=1)
    ax.set_ylabel("% victoria local"); ax.legend(fontsize=7); ax.set_ylim(20, 75)
    ax.set_title(
        f"{LIGAS_NAME[liga]}  tendencia: {m_f*100:+.4f}pp/sem  R²={model_full.rsquared:.4f}  p={model_full.pvalues['t']:.4f}",
        fontweight="bold")
    print(f"  {LIGAS_NAME[liga]}: {m_f*100:+.5f}pp/sem  R²={model_full.rsquared:.4f}  p={model_full.pvalues['t']:.4f}  MAE={mae*100:.2f}pp  pred+16w={fut_pct.iloc[-1]:.1f}%")

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "forecasting_homewin_por_liga.png"), dpi=130)
plt.close()
print("  guardado: img/forecasting_homewin_por_liga.png\n")
print_tabulate(pd.DataFrame(resumen_hw))


# analicé el movimiento de cuotas de apertura a cierre por liga
# si la tendencia es consistente en una dirección el smart money está corrigiendo
# un sesgo sistemático en la apertura de la casa

print("\n3. movimiento de odds apertura a cierre por liga")

resumen_mov = []
fig, axes   = plt.subplots(len(ligas), 1, figsize=(14, 4 * len(ligas)), sharex=False)

for i, liga in enumerate(ligas):
    sub = df[df["Div"] == liga].copy()
    model_full, ts_full, fut_full = forecast_serie(sub, "Date", "odds_move_H", n_weeks=16)
    model_tr, train, test, pred_test, mae, rmse = train_test_ols(ts_full)
    m_f = model_full.params["t"]
    col = LIGA_COLORS[liga]
    yp  = model_full.predict(sm.add_constant(ts_full["t"]))
    resumen_mov.append({
        "Liga":          LIGAS_NAME[liga],
        "Media mov_H":   round(ts_full["valor"].mean(), 5),
        "Tendencia/sem": round(m_f, 6),
        "R²":            round(model_full.rsquared, 4),
        "p-value":       round(model_full.pvalues["t"], 4),
        "MAE test":      round(mae, 4),
        "Pred +16w":     round(fut_full["pred"].iloc[-1], 5),
    })
    ax = axes[i]
    ax.plot(ts_full["Fecha"], ts_full["valor"],
            color=col, alpha=0.45, linewidth=1, label="mov semanal avg")
    ax.plot(ts_full["Fecha"], yp, color="red", linewidth=1.8,
            label=f"tendencia  R²={model_full.rsquared:.4f}  p={model_full.pvalues['t']:.4f}")
    ax.plot(fut_full["Fecha"], fut_full["pred"], color="orange",
            linewidth=2, linestyle="--", label=f"pred +16w  {fut_full['pred'].iloc[-1]:.4f}")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axvline(test["Fecha"].iloc[0], color="gray", linestyle=":", linewidth=1)
    ax.set_ylabel("odds_move_H (cierre - apertura)"); ax.legend(fontsize=7)
    ax.set_title(
        f"{LIGAS_NAME[liga]}  tendencia: {m_f:+.6f}/sem  R²={model_full.rsquared:.4f}  p={model_full.pvalues['t']:.4f}",
        fontweight="bold")
    print(f"  {LIGAS_NAME[liga]}: media={ts_full['valor'].mean():+.5f}  tendencia={m_f:+.6f}/sem  R²={model_full.rsquared:.4f}  p={model_full.pvalues['t']:.4f}")

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "forecasting_odds_move.png"), dpi=130)
plt.close()
print("  guardado: img/forecasting_odds_move.png\n")
print_tabulate(pd.DataFrame(resumen_mov))


# analicé el overround semanal para ver si el margen de la casa sube o baja
# un margen creciente indica que la casa se vuelve menos competitiva con el tiempo

print("\n4. overround semanal por liga")

resumen_or = []
fig, axes  = plt.subplots(len(ligas), 1, figsize=(14, 4 * len(ligas)), sharex=False)

for i, liga in enumerate(ligas):
    sub = df[df["Div"] == liga].copy()
    model_full, ts_full, fut_full = forecast_serie(sub, "Date", "overround", n_weeks=16)
    model_tr, train, test, pred_test, mae, rmse = train_test_ols(ts_full)
    m_f          = model_full.params["t"]
    col          = LIGA_COLORS[liga]
    yp           = model_full.predict(sm.add_constant(ts_full["t"]))
    margen_medio = (ts_full["valor"].mean() - 1) * 100
    resumen_or.append({
        "Liga":          LIGAS_NAME[liga],
        "Overround avg": round(ts_full["valor"].mean(), 5),
        "Margen casa %": f"{margen_medio:.2f}%",
        "Tendencia/sem": round(m_f, 6),
        "R²":            round(model_full.rsquared, 4),
        "p-value":       round(model_full.pvalues["t"], 4),
        "Pred +16w":     round(fut_full["pred"].iloc[-1], 5),
    })
    ax = axes[i]
    ax.plot(ts_full["Fecha"], (ts_full["valor"] - 1) * 100,
            color=col, alpha=0.45, linewidth=1, label="margen semanal %")
    ax.plot(ts_full["Fecha"], (yp - 1) * 100, color="red", linewidth=1.8,
            label=f"tendencia  R²={model_full.rsquared:.4f}  p={model_full.pvalues['t']:.4f}")
    ax.plot(fut_full["Fecha"], (fut_full["pred"] - 1) * 100, color="orange",
            linewidth=2, linestyle="--",
            label=f"pred +16w  {(fut_full['pred'].iloc[-1]-1)*100:.2f}%")
    ax.axvline(test["Fecha"].iloc[0], color="gray", linestyle=":", linewidth=1)
    ax.set_ylabel("margen casa (%)"); ax.legend(fontsize=7)
    ax.set_title(
        f"{LIGAS_NAME[liga]}  margen medio: {margen_medio:.2f}%  tendencia: {m_f*100:+.5f}pp/sem  R²={model_full.rsquared:.4f}",
        fontweight="bold")
    print(f"  {LIGAS_NAME[liga]}: margen={margen_medio:.2f}%  tendencia={m_f*100:+.5f}pp/sem  R²={model_full.rsquared:.4f}  p={model_full.pvalues['t']:.4f}")

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "forecasting_overround.png"), dpi=130)
plt.close()
print("  guardado: img/forecasting_overround.png\n")
print_tabulate(pd.DataFrame(resumen_or))


# bajé al nivel de equipo específico y comparé rendimiento de local vs visitante
# usé períodos de 4 semanas porque cada equipo juega pocos partidos de local por mes

print("\n5. victorias local vs visitante por equipo")

resumen_eq = []
fig, axes  = plt.subplots(len(equipos), 1, figsize=(14, 4 * len(equipos)), sharex=False)

for i, (equipo, liga, col) in enumerate(equipos):
    sub_h = df[df["HomeTeam"] == equipo].copy()
    sub_a = df[df["AwayTeam"] == equipo].copy()

    if len(sub_h) < 20 or len(sub_a) < 20:
        print(f"  {equipo}: pocos datos, omitido")
        continue

    model_h, ts_h, fut_h = forecast_serie(sub_h, "Date", "home_win",  n_weeks=16, freq="4W")
    model_a, ts_a, fut_a = forecast_serie(sub_a, "Date", "away_win",  n_weeks=16, freq="4W")
    _, _, test_h, _, mae_h, _ = train_test_ols(ts_h)
    _, _, test_a, _, mae_a, _ = train_test_ols(ts_a)

    m_h = model_h.params["t"]
    m_a = model_a.params["t"]
    yp_h = model_h.predict(sm.add_constant(ts_h["t"]))
    yp_a = model_a.predict(sm.add_constant(ts_a["t"]))

    resumen_eq.append({
        "Equipo":         equipo,
        "Media local":    f"{ts_h['valor'].mean()*100:.1f}%",
        "Tend local/per": round(m_h * 100, 4),
        "R² local":       round(model_h.rsquared, 4),
        "p local":        round(model_h.pvalues["t"], 4),
        "Media visita":   f"{ts_a['valor'].mean()*100:.1f}%",
        "Tend visita/per":round(m_a * 100, 4),
        "R² visita":      round(model_a.rsquared, 4),
        "p visita":       round(model_a.pvalues["t"], 4),
    })

    ax = axes[i]
    ax.plot(ts_h["Fecha"], ts_h["valor"] * 100,
            color=col, alpha=0.4, linewidth=1, label="local observado")
    ax.plot(ts_h["Fecha"], yp_h * 100,
            color=col, linewidth=2,
            label=f"local tendencia  R²={model_h.rsquared:.3f}  {m_h*100:+.3f}pp/per")
    ax.plot(fut_h["Fecha"], fut_h["pred"] * 100,
            color=col, linewidth=1.8, linestyle="--")
    ax.plot(ts_a["Fecha"], ts_a["valor"] * 100,
            color="gray", alpha=0.4, linewidth=1, label="visita observado")
    ax.plot(ts_a["Fecha"], yp_a * 100,
            color="black", linewidth=2,
            label=f"visita tendencia  R²={model_a.rsquared:.3f}  {m_a*100:+.3f}pp/per")
    ax.plot(fut_a["Fecha"], fut_a["pred"] * 100,
            color="black", linewidth=1.8, linestyle="--")
    ax.axvline(ts_h["Fecha"].max(), color="gray", linestyle=":", linewidth=1)
    ax.set_ylabel("% victorias"); ax.legend(fontsize=7); ax.set_ylim(-10, 120)
    ax.set_title(f"{equipo} ({LIGAS_NAME[liga]})  local vs visitante", fontweight="bold")
    print(f"  {equipo}: local={ts_h['valor'].mean()*100:.1f}% ({m_h*100:+.4f}pp/per p={model_h.pvalues['t']:.4f})  visita={ts_a['valor'].mean()*100:.1f}% ({m_a*100:+.4f}pp/per p={model_a.pvalues['t']:.4f})")

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "forecasting_equipos.png"), dpi=130)
plt.close()
print("  guardado: img/forecasting_equipos.png\n")
print_tabulate(pd.DataFrame(resumen_eq))


# analicé el bias por temporada para ver si las casas han ido corrigiendo
# su sobreestimación del local con el paso de los años

print("\n6. bias probabilidad implícita vs resultado real por liga")

resumen_bias = []
fig, axes    = plt.subplots(len(ligas), 1, figsize=(14, 4 * len(ligas)), sharex=False)

for i, liga in enumerate(ligas):
    sub = df[df["Div"] == liga].copy()
    model_full, ts_full, fut_full = forecast_serie(sub, "Date", "bias_H", n_weeks=16)
    model_tr, train, test, pred_test, mae, rmse = train_test_ols(ts_full)
    m_f       = model_full.params["t"]
    col       = LIGA_COLORS[liga]
    yp        = model_full.predict(sm.add_constant(ts_full["t"]))
    media     = ts_full["valor"].mean()
    direccion = "sobreestima local" if media > 0 else "subestima local"
    tendencia = "bias creciente"    if m_f > 0   else "bias decreciente"
    resumen_bias.append({
        "Liga":          LIGAS_NAME[liga],
        "Bias medio":    round(media, 5),
        "Dirección":     direccion,
        "Tendencia/sem": round(m_f, 6),
        "Tendencia dir": tendencia,
        "R²":            round(model_full.rsquared, 4),
        "p-value":       round(model_full.pvalues["t"], 4),
        "Pred +16w":     round(fut_full["pred"].iloc[-1], 5),
    })
    ax = axes[i]
    ax.plot(ts_full["Fecha"], ts_full["valor"],
            color=col, alpha=0.45, linewidth=1, label="bias semanal")
    ax.plot(ts_full["Fecha"], yp, color="red", linewidth=1.8,
            label=f"tendencia  R²={model_full.rsquared:.4f}  p={model_full.pvalues['t']:.4f}")
    ax.plot(fut_full["Fecha"], fut_full["pred"], color="orange",
            linewidth=2, linestyle="--", label=f"pred +16w  {fut_full['pred'].iloc[-1]:.4f}")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.9, alpha=0.7,
               label="bias=0 (mercado perfecto)")
    ax.axvline(test["Fecha"].iloc[0], color="gray", linestyle=":", linewidth=1)
    ax.set_ylabel("bias (imp_prob_H - home_win)"); ax.legend(fontsize=7)
    ax.set_title(
        f"{LIGAS_NAME[liga]}  bias medio: {media:+.4f} ({direccion})  R²={model_full.rsquared:.4f}  p={model_full.pvalues['t']:.4f}",
        fontweight="bold")
    print(f"  {LIGAS_NAME[liga]}: bias={media:+.5f} ({direccion})  tendencia={m_f:+.6f}/sem  R²={model_full.rsquared:.4f}  p={model_full.pvalues['t']:.4f}")

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "forecasting_bias.png"), dpi=130)
plt.close()
print("  guardado: img/forecasting_bias.png\n")
print_tabulate(pd.DataFrame(resumen_bias))


# calculé el bias por temporada para ver si las casas han ido corrigiendo
# su sobreestimación del local año a año — si baja significa que el mercado aprende

print("\n7. calibración del bias por temporada")

seasons     = sorted(df["Season_label"].dropna().unique())
resumen_cal = []

for liga in ligas:
    sub = df[df["Div"] == liga].copy()
    for season in seasons:
        s = sub[sub["Season_label"] == season]
        if len(s) < 10:
            continue
        bias_medio = (1 / s["AvgH"] - s["home_win"]).mean()
        resumen_cal.append({
            "Liga":       LIGAS_NAME[liga],
            "Temporada":  season,
            "Bias local": round(bias_medio, 5),
            "Partidos":   len(s),
        })

df_cal = pd.DataFrame(resumen_cal)
print_tabulate(df_cal)

fig, ax = plt.subplots(figsize=(13, 6))
for liga in ligas:
    sub_cal = df_cal[df_cal["Liga"] == LIGAS_NAME[liga]]
    ax.plot(sub_cal["Temporada"], sub_cal["Bias local"],
            marker="o", linewidth=2, color=LIGA_COLORS[liga],
            label=LIGAS_NAME[liga])

ax.axhline(0, color="gray", linestyle="--", linewidth=0.9, alpha=0.7,
           label="bias=0 (mercado perfecto)")
ax.set_ylabel("bias local (imp_prob_H - home_win)")
ax.set_title("Calibración del bias por temporada — ¿aprende el mercado?",
             fontweight="bold")
ax.legend(fontsize=8)
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "forecasting_bias_temporada.png"), dpi=130)
plt.close()
print("  guardado: img/forecasting_bias_temporada.png\n")


# analicé si el overround cambia por semana del año para detectar si las casas
# suben márgenes en fechas clave como navidad o finales de temporada

print("\n8. overround por semana del año")

fig, axes = plt.subplots(len(ligas), 1, figsize=(14, 4 * len(ligas)), sharex=False)
resumen_wk = []

for i, liga in enumerate(ligas):
    sub    = df[df["Div"] == liga].copy()
    wk_avg = sub.groupby("week_of_year")["overround"].mean().reset_index()
    wk_avg.columns = ["semana", "overround"]
    wk_avg["t"]    = range(len(wk_avg))

    X      = sm.add_constant(wk_avg["t"])
    model  = sm.OLS(wk_avg["overround"], X).fit()
    m_wk   = model.params["t"]
    yp_wk  = model.predict(X)
    col    = LIGA_COLORS[liga]

    semana_max = wk_avg.loc[wk_avg["overround"].idxmax(), "semana"]
    semana_min = wk_avg.loc[wk_avg["overround"].idxmin(), "semana"]

    resumen_wk.append({
        "Liga":        LIGAS_NAME[liga],
        "OR medio":    round(wk_avg["overround"].mean(), 5),
        "OR max sem":  int(semana_max),
        "OR min sem":  int(semana_min),
        "Rango":       round(wk_avg["overround"].max() - wk_avg["overround"].min(), 5),
    })

    ax = axes[i]
    ax.bar(wk_avg["semana"], (wk_avg["overround"] - 1) * 100,
           color=col, alpha=0.5, label="margen % por semana del año")
    ax.plot(wk_avg["semana"], (yp_wk - 1) * 100,
            color="red", linewidth=1.8, label=f"tendencia  R²={model.rsquared:.4f}")
    ax.axvline(semana_max, color="orange", linestyle="--", linewidth=1.2,
               label=f"máx semana {int(semana_max)}")
    ax.axvline(semana_min, color="blue", linestyle="--", linewidth=1.2,
               label=f"mín semana {int(semana_min)}")
    ax.set_xlabel("semana del año"); ax.set_ylabel("margen casa (%)")
    ax.set_title(f"{LIGAS_NAME[liga]}  máx: sem {int(semana_max)}  mín: sem {int(semana_min)}",
                 fontweight="bold")
    ax.legend(fontsize=7)
    print(f"  {LIGAS_NAME[liga]}: OR medio={(wk_avg['overround'].mean()-1)*100:.2f}%  máx sem={int(semana_max)}  mín sem={int(semana_min)}")

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "forecasting_overround_semana.png"), dpi=130)
plt.close()
print("  guardado: img/forecasting_overround_semana.png\n")
print_tabulate(pd.DataFrame(resumen_wk))


# simulé una estrategia de value bet acumulada por liga
# aposté siempre al visitante (donde el bias indica que la casa sobreestima al local)
# y calculé el bankroll acumulado a lo largo del tiempo asumiendo stake fijo de 1 unidad

print("\n9. value bet acumulado — apuesta sistemática al visitante")

fig, ax     = plt.subplots(figsize=(14, 6))
resumen_vb  = []

for liga in ligas:
    sub = df[df["Div"] == liga].copy().sort_values("Date").reset_index(drop=True)

    # calculé el retorno de apostar 1 unidad al visitante en cada partido
    # si gana el visitante: ganancia = odd_visitante - 1, si pierde: -1

    sub["retorno_away"] = np.where(
        sub["away_win"] == 1,
        sub["AvgA"] - 1,
        -1
    )
    sub["bankroll"] = sub["retorno_away"].cumsum()

    roi_total  = sub["retorno_away"].sum() / len(sub) * 100
    win_rate   = sub["away_win"].mean() * 100
    mejor_odd  = sub["AvgA"].mean()

    resumen_vb.append({
        "Liga":         LIGAS_NAME[liga],
        "Partidos":     len(sub),
        "Win rate %":   round(win_rate, 2),
        "Odd media":    round(mejor_odd, 3),
        "ROI %":        round(roi_total, 2),
        "P&L final":    round(sub["bankroll"].iloc[-1], 2),
    })

    ax.plot(sub["Date"], sub["bankroll"],
            color=LIGA_COLORS[liga], linewidth=1.8, label=f"{LIGAS_NAME[liga]}  ROI={roi_total:.1f}%")
    print(f"  {LIGAS_NAME[liga]}: win_rate={win_rate:.1f}%  odd_media={mejor_odd:.3f}  ROI={roi_total:.2f}%  P&L={sub['bankroll'].iloc[-1]:.2f}u")

ax.axhline(0, color="gray", linestyle="--", linewidth=0.9, alpha=0.7)
ax.set_xlabel("Fecha"); ax.set_ylabel("bankroll acumulado (unidades)")
ax.set_title("Value bet: bankroll acumulado apostando siempre al visitante por liga",
             fontweight="bold")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "forecasting_valuebet_visitante.png"), dpi=130)
plt.close()
print("  guardado: img/forecasting_valuebet_visitante.png\n")
print_tabulate(pd.DataFrame(resumen_vb))


# filtré partidos donde el visitante es underdog con cuota mayor a 4.0
# y analicé si la casa los subestima sistemáticamente comparando imp_prob vs resultado real

print("\n10. underdog visitante — ¿está subvaluado por la casa?")

fig, axes   = plt.subplots(len(ligas), 1, figsize=(14, 4 * len(ligas)), sharex=False)
resumen_ud  = []

for i, liga in enumerate(ligas):
    sub    = df[(df["Div"] == liga) & (df["underdog_away"] == 1)].copy()
    sub_all = df[df["Div"] == liga].copy()

    if len(sub) < 20:
        print(f"  {LIGAS_NAME[liga]}: pocos underdogs, omitido")
        continue

    imp_avg    = sub["imp_prob_A"].mean()
    win_real   = sub["away_win"].mean()
    bias_ud    = imp_avg - win_real
    n_partidos = len(sub)
    pct_total  = n_partidos / len(sub_all) * 100

    # calculé ROI de apostar siempre al underdog visitante en esta liga
    sub["retorno"] = np.where(sub["away_win"] == 1, sub["AvgA"] - 1, -1)
    roi_ud = sub["retorno"].sum() / n_partidos * 100
    bankroll_ud = sub.sort_values("Date")["retorno"].cumsum()

    resumen_ud.append({
        "Liga":          LIGAS_NAME[liga],
        "N underdogs":   n_partidos,
        "% del total":   f"{pct_total:.1f}%",
        "imp_prob avg":  round(imp_avg * 100, 2),
        "win rate real": f"{win_real*100:.1f}%",
        "Bias":          round(bias_ud, 5),
        "ROI %":         round(roi_ud, 2),
    })

    col = LIGA_COLORS[liga]
    ax  = axes[i]
    ax.plot(sub.sort_values("Date")["Date"], bankroll_ud.values,
            color=col, linewidth=1.8, label=f"bankroll underdog  ROI={roi_ud:.1f}%")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.set_ylabel("bankroll acumulado (u)"); ax.legend(fontsize=7)
    ax.set_title(
        f"{LIGAS_NAME[liga]}  underdogs: {n_partidos} ({pct_total:.1f}% total)  imp={imp_avg*100:.1f}%  real={win_real*100:.1f}%  bias={bias_ud:+.4f}",
        fontweight="bold")
    print(f"  {LIGAS_NAME[liga]}: n={n_partidos}  imp={imp_avg*100:.1f}%  real={win_real*100:.1f}%  bias={bias_ud:+.5f}  ROI={roi_ud:.2f}%")

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "forecasting_underdog.png"), dpi=130)
plt.close()
print("  guardado: img/forecasting_underdog.png\n")
print_tabulate(pd.DataFrame(resumen_ud))


# comparé métricas clave en semanas post fecha FIFA vs semanas normales
# los equipos llegan con jugadores cansados o lesionados después de internacionales
# eso debería aumentar varianza y reducir la ventaja del local

print("\n11. efecto semanas post fecha FIFA")

resumen_fifa = []

for liga in ligas:
    sub      = df[df["Div"] == liga].copy()
    normal   = sub[sub["post_fifa"] == 0]
    post     = sub[sub["post_fifa"] == 1]

    hw_norm  = normal["home_win"].mean() * 100
    hw_post  = post["home_win"].mean()   * 100
    o25_norm = normal["over25"].mean()   * 100
    o25_post = post["over25"].mean()     * 100
    or_norm  = (normal["overround"].mean() - 1) * 100
    or_post  = (post["overround"].mean()   - 1) * 100

    resumen_fifa.append({
        "Liga":           LIGAS_NAME[liga],
        "Local normal":   f"{hw_norm:.1f}%",
        "Local post-FIFA":f"{hw_post:.1f}%",
        "Δ local":        f"{hw_post-hw_norm:+.1f}pp",
        "O2.5 normal":    f"{o25_norm:.1f}%",
        "O2.5 post-FIFA": f"{o25_post:.1f}%",
        "Δ over2.5":      f"{o25_post-o25_norm:+.1f}pp",
        "OR normal":      f"{or_norm:.2f}%",
        "OR post-FIFA":   f"{or_post:.2f}%",
    })
    print(f"  {LIGAS_NAME[liga]}: local normal={hw_norm:.1f}% vs post-FIFA={hw_post:.1f}% (Δ{hw_post-hw_norm:+.1f}pp)  over2.5 normal={o25_norm:.1f}% vs post-FIFA={o25_post:.1f}% (Δ{o25_post-o25_norm:+.1f}pp)")

print_tabulate(pd.DataFrame(resumen_fifa))

# grafiqué la comparación de home_win y over2.5 entre semanas normales y post-FIFA

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
nombres  = [LIGAS_NAME[l] for l in ligas]
hw_norms = [float(r["Local normal"].replace("%",""))    for r in resumen_fifa]
hw_posts = [float(r["Local post-FIFA"].replace("%","")) for r in resumen_fifa]
o25_norms= [float(r["O2.5 normal"].replace("%",""))     for r in resumen_fifa]
o25_posts= [float(r["O2.5 post-FIFA"].replace("%",""))  for r in resumen_fifa]
x        = np.arange(len(nombres))
w        = 0.35

ax1.bar(x - w/2, hw_norms, w, label="normal",    color="#3498db", alpha=0.8)
ax1.bar(x + w/2, hw_posts, w, label="post-FIFA", color="#e74c3c", alpha=0.8)
ax1.set_xticks(x); ax1.set_xticklabels(nombres, rotation=15, fontsize=8)
ax1.set_ylabel("% victorias local"); ax1.legend()
ax1.set_title("% victorias local: normal vs post fecha FIFA", fontweight="bold")

ax2.bar(x - w/2, o25_norms, w, label="normal",    color="#2ecc71", alpha=0.8)
ax2.bar(x + w/2, o25_posts, w, label="post-FIFA", color="#f39c12", alpha=0.8)
ax2.set_xticks(x); ax2.set_xticklabels(nombres, rotation=15, fontsize=8)
ax2.set_ylabel("% over 2.5"); ax2.legend()
ax2.set_title("% over 2.5: normal vs post fecha FIFA", fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "forecasting_post_fifa.png"), dpi=130)
plt.close()
print("  guardado: img/forecasting_post_fifa.png")


# calculé el ROI de apostar under 2.5 en Ligue 1 específicamente en semanas
# post fecha FIFA, donde detectamos una caída de -5.2pp en over2.5
# usé la cuota implícita del under como 1 / (1 - imp_prob_over25) para estimar
# la cuota de mercado del under a partir de los datos disponibles

print("\n12. ROI under 2.5 en Ligue 1 post fecha FIFA")

ligue = df[(df["Div"] == "F1") & (df["post_fifa"] == 1)].copy()
ligue_norm = df[(df["Div"] == "F1") & (df["post_fifa"] == 0)].copy()

# aproximé la cuota del under usando la probabilidad implícita del over
# si imp_prob_over = 0.55, la cuota del under sería ~1 / 0.45 = 2.22
# apliqué un factor de overround para hacerlo más realista

ligue["imp_prob_over"] = ligue["imp_prob_H"] * 0 + ligue["over25"].expanding().mean()
ligue["cuota_under"]   = round(1 / (1 - ligue["over25"].expanding().mean() + 0.001), 4)

# usé una cuota fija estimada de 1.85 para under 2.5 en Ligue 1
# que es el valor típico de mercado cuando el over está entre 50-55%

CUOTA_UNDER_LIGUE = 1.85
ligue["under_real"]   = (ligue["total_goals"] <= 2).astype(int)
ligue["retorno_under"] = np.where(ligue["under_real"] == 1,
                                   CUOTA_UNDER_LIGUE - 1, -1)
ligue_norm["under_real"]    = (ligue_norm["total_goals"] <= 2).astype(int)
ligue_norm["retorno_under"] = np.where(ligue_norm["under_real"] == 1,
                                        CUOTA_UNDER_LIGUE - 1, -1)

ligue_sorted      = ligue.sort_values("Date").reset_index(drop=True)
ligue_norm_sorted = ligue_norm.sort_values("Date").reset_index(drop=True)

bankroll_post  = ligue_sorted["retorno_under"].cumsum()
bankroll_norm  = ligue_norm_sorted["retorno_under"].cumsum()

roi_post = ligue_sorted["retorno_under"].sum() / len(ligue_sorted) * 100
roi_norm = ligue_norm_sorted["retorno_under"].sum() / len(ligue_norm_sorted) * 100
wr_post  = ligue_sorted["under_real"].mean() * 100
wr_norm  = ligue_norm_sorted["under_real"].mean() * 100

print(f"  post-FIFA: n={len(ligue_sorted)}  win_rate={wr_post:.1f}%  ROI={roi_post:.2f}%  P&L={bankroll_post.iloc[-1]:.2f}u")
print(f"  normal:    n={len(ligue_norm_sorted)}  win_rate={wr_norm:.1f}%  ROI={roi_norm:.2f}%  P&L={bankroll_norm.iloc[-1]:.2f}u")
print(f"  diferencia win rate: {wr_post - wr_norm:+.1f}pp")
print(f"  diferencia ROI:      {roi_post - roi_norm:+.2f}pp")

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(ligue_sorted["Date"], bankroll_post,
        color="#9b59b6", linewidth=2,
        label=f"under post-FIFA  ROI={roi_post:.1f}%  WR={wr_post:.1f}%")
ax.plot(ligue_norm_sorted["Date"], bankroll_norm,
        color="gray", linewidth=1.5, alpha=0.7,
        label=f"under normal     ROI={roi_norm:.1f}%  WR={wr_norm:.1f}%")
ax.axhline(0, color="gray", linestyle="--", linewidth=0.9, alpha=0.6)
ax.set_xlabel("Fecha"); ax.set_ylabel("bankroll acumulado (unidades)")
ax.set_title("Ligue 1 — ROI under 2.5: semanas post-FIFA vs semanas normales\n(cuota estimada 1.85)",
             fontweight="bold")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "forecasting_under_ligue1_postfifa.png"), dpi=130)
plt.close()
print("  guardado: img/forecasting_under_ligue1_postfifa.png\n")


# comparé la probabilidad implícita que la casa pone a Arsenal de local
# contra su win rate rodante de las últimas 10 semanas para detectar value
# si el win rate supera consistentemente la imp_prob → la casa está subvaluando

print("\n13. Arsenal local — cuota implícita vs win rate rodante")

arsenal = df[df["HomeTeam"] == "Arsenal"].copy().sort_values("Date").reset_index(drop=True)
arsenal["rolling_wr"]   = arsenal["home_win"].rolling(10, min_periods=5).mean()
arsenal["gap_value"]    = arsenal["rolling_wr"] - arsenal["imp_prob_H"]
arsenal["value_bet"]    = (arsenal["gap_value"] > 0).astype(int)

n_value     = arsenal["value_bet"].sum()
pct_value   = n_value / len(arsenal) * 100
gap_medio   = arsenal["gap_value"].mean()
gap_reciente = arsenal["gap_value"].iloc[-10:].mean()

print(f"  partidos totales     : {len(arsenal)}")
print(f"  partidos con value   : {n_value} ({pct_value:.1f}%)")
print(f"  gap medio histórico  : {gap_medio:+.4f}")
print(f"  gap últimas 10 fechas: {gap_reciente:+.4f}")
print(f"  {'HAY VALUE en Arsenal local actualmente' if gap_reciente > 0 else 'NO hay value en Arsenal local actualmente'}")

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(arsenal["Date"], arsenal["rolling_wr"] * 100,
        color="#ef0107", linewidth=2, label="win rate rodante 10 partidos")
ax.plot(arsenal["Date"], arsenal["imp_prob_H"] * 100,
        color="#3498db", linewidth=1.8, linestyle="--",
        label="probabilidad implícita casa")
ax.fill_between(arsenal["Date"],
                arsenal["rolling_wr"] * 100,
                arsenal["imp_prob_H"] * 100,
                where=(arsenal["rolling_wr"] > arsenal["imp_prob_H"]),
                alpha=0.25, color="green", label="zona value (WR > imp_prob)")
ax.fill_between(arsenal["Date"],
                arsenal["rolling_wr"] * 100,
                arsenal["imp_prob_H"] * 100,
                where=(arsenal["rolling_wr"] <= arsenal["imp_prob_H"]),
                alpha=0.15, color="red", label="zona sin value")
ax.set_xlabel("Fecha"); ax.set_ylabel("%")
ax.set_title("Arsenal local — win rate rodante vs probabilidad implícita de la casa",
             fontweight="bold")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "forecasting_arsenal_value.png"), dpi=130)
plt.close()
print("  guardado: img/forecasting_arsenal_value.png\n")


# calculé el rolling win rate de los últimos 10 partidos para todos los equipos
# y lo comparé con la cuota implícita promedio reciente de la casa
# un gap positivo significa que el equipo gana más de lo que la casa estima → value

print("\n14. rolling win rate vs cuota implícita — todos los equipos")

resumen_roll = []

for equipo, liga, col in equipos:
    # local
    sub_h = df[df["HomeTeam"] == equipo].copy().sort_values("Date").reset_index(drop=True)
    sub_h["rolling_wr"] = sub_h["home_win"].rolling(10, min_periods=5).mean()
    sub_h["gap"]        = sub_h["rolling_wr"] - sub_h["imp_prob_H"]
    gap_rec_h   = sub_h["gap"].iloc[-10:].mean()
    wr_rec_h    = sub_h["rolling_wr"].iloc[-1]
    imp_rec_h   = sub_h["imp_prob_H"].iloc[-10:].mean()

    # visitante
    sub_a = df[df["AwayTeam"] == equipo].copy().sort_values("Date").reset_index(drop=True)
    sub_a["rolling_wr"] = sub_a["away_win"].rolling(10, min_periods=5).mean()
    sub_a["gap"]        = sub_a["rolling_wr"] - sub_a["imp_prob_A"]
    gap_rec_a   = sub_a["gap"].iloc[-10:].mean()
    wr_rec_a    = sub_a["rolling_wr"].iloc[-1]
    imp_rec_a   = sub_a["imp_prob_A"].iloc[-10:].mean()

    resumen_roll.append({
        "Equipo":         equipo,
        "WR local rec":   f"{wr_rec_h*100:.1f}%",
        "Imp local rec":  f"{imp_rec_h*100:.1f}%",
        "Gap local":      f"{gap_rec_h*100:+.1f}pp",
        "Value local":    "SI" if gap_rec_h > 0 else "no",
        "WR visita rec":  f"{wr_rec_a*100:.1f}%",
        "Imp visita rec": f"{imp_rec_a*100:.1f}%",
        "Gap visita":     f"{gap_rec_a*100:+.1f}pp",
        "Value visita":   "SI" if gap_rec_a > 0 else "no",
    })
    print(f"  {equipo}: local gap={gap_rec_h*100:+.1f}pp ({'VALUE' if gap_rec_h > 0 else 'sin value'})  visita gap={gap_rec_a*100:+.1f}pp ({'VALUE' if gap_rec_a > 0 else 'sin value'})")

print_tabulate(pd.DataFrame(resumen_roll))

# grafiqué el gap de todos los equipos en barras para ver quién tiene value hoy

fig, ax = plt.subplots(figsize=(13, 6))
nombres_eq  = [e[0] for e in equipos]
gaps_local  = [float(r["Gap local"].replace("pp",""))  for r in resumen_roll]
gaps_visita = [float(r["Gap visita"].replace("pp","")) for r in resumen_roll]
x = np.arange(len(nombres_eq))
w = 0.35

bars_h = ax.bar(x - w/2, gaps_local,  w, label="gap local",   alpha=0.85,
                color=["#2ecc71" if g > 0 else "#e74c3c" for g in gaps_local])
bars_a = ax.bar(x + w/2, gaps_visita, w, label="gap visitante", alpha=0.85,
                color=["#27ae60" if g > 0 else "#c0392b" for g in gaps_visita])
ax.axhline(0, color="gray", linestyle="--", linewidth=0.9)
ax.set_xticks(x); ax.set_xticklabels(nombres_eq, rotation=15, fontsize=9)
ax.set_ylabel("gap win rate - imp_prob (pp)")
ax.set_title("Gap value por equipo — verde=value positivo, rojo=sin value\n(últimas 10 fechas)",
             fontweight="bold")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "forecasting_rolling_gap_equipos.png"), dpi=130)
plt.close()
print("  guardado: img/forecasting_rolling_gap_equipos.png\n")


# consolidé todos los hallazgos en una tabla resumen por liga y tipo de apuesta
# para dar una recomendación clara basada en los datos del análisis completo

print("\n15. tabla resumen — mejor liga por tipo de apuesta")

resumen_final = []
for liga in ligas:
    sub = df[df["Div"] == liga].copy()

    over_media   = sub["over25"].mean() * 100
    hw_media     = sub["home_win"].mean() * 100
    aw_media     = sub["away_win"].mean() * 100
    bias_media   = (sub["imp_prob_H"] - sub["home_win"]).mean()
    or_media     = (sub["overround"].mean() - 1) * 100
    ud_sub       = sub[sub["underdog_away"] == 1]
    ud_bias      = (ud_sub["imp_prob_A"] - ud_sub["away_win"]).mean() if len(ud_sub) > 0 else 0
    post_sub     = sub[sub["post_fifa"] == 1]
    over_post    = post_sub["over25"].mean() * 100 if len(post_sub) > 0 else 0
    over_norm    = sub[sub["post_fifa"] == 0]["over25"].mean() * 100
    efecto_fifa  = over_post - over_norm

    resumen_final.append({
        "Liga":            LIGAS_NAME[liga],
        "% over2.5":       f"{over_media:.1f}%",
        "% local":         f"{hw_media:.1f}%",
        "% visitante":     f"{aw_media:.1f}%",
        "Bias local":      f"{bias_media*100:+.2f}pp",
        "Margen casa":     f"{or_media:.2f}%",
        "Bias underdog":   f"{ud_bias*100:+.2f}pp",
        "Efecto FIFA o25": f"{efecto_fifa:+.1f}pp",
        "Mejor apuesta":   (
            "under post-FIFA" if abs(efecto_fifa) > 3 and efecto_fifa < 0
            else "over2.5" if over_media > 58
            else "visitante" if bias_media > 0.03
            else "mercado eficiente"
        ),
    })

print_tabulate(pd.DataFrame(resumen_final))

fig, ax = plt.subplots(figsize=(13, 5))
nombres_l  = [LIGAS_NAME[l] for l in ligas]
over_vals  = [sub["over25"].mean() * 100 for l in ligas
              for sub in [df[df["Div"] == l]]]
hw_vals    = [sub["home_win"].mean() * 100 for l in ligas
              for sub in [df[df["Div"] == l]]]
aw_vals    = [sub["away_win"].mean() * 100 for l in ligas
              for sub in [df[df["Div"] == l]]]
x = np.arange(len(nombres_l))
w = 0.25

ax.bar(x - w,   over_vals, w, label="% over 2.5",   color="#3498db", alpha=0.85)
ax.bar(x,       hw_vals,   w, label="% local gana",  color="#e74c3c", alpha=0.85)
ax.bar(x + w,   aw_vals,   w, label="% visita gana", color="#2ecc71", alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(nombres_l, rotation=15, fontsize=9)
ax.set_ylabel("%"); ax.legend(fontsize=8)
ax.set_title("Resumen comparativo por liga — métricas base para decisión de apuesta",
             fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "forecasting_resumen_final.png"), dpi=130)
plt.close()
print("  guardado: img/forecasting_resumen_final.png")


# comparé el gap de las últimas 5 vs las últimas 10 fechas por equipo
# las casas tardan en ajustar cuotas cuando un equipo entra en racha caliente
# si el gap de las últimas 5 es mayor que el de las últimas 10, el value está creciendo

print("\n16. racha corta (5) vs racha larga (10) — detección de value creciente")

resumen_racha = []

for equipo, liga, col in equipos:
    sub_h = df[df["HomeTeam"] == equipo].copy().sort_values("Date").reset_index(drop=True)
    sub_a = df[df["AwayTeam"] == equipo].copy().sort_values("Date").reset_index(drop=True)

    sub_h["rwr10"] = sub_h["home_win"].rolling(10, min_periods=5).mean()
    sub_h["rwr5"]  = sub_h["home_win"].rolling(5,  min_periods=3).mean()
    sub_h["gap10"] = sub_h["rwr10"] - sub_h["imp_prob_H"]
    sub_h["gap5"]  = sub_h["rwr5"]  - sub_h["imp_prob_H"]

    sub_a["rwr10"] = sub_a["away_win"].rolling(10, min_periods=5).mean()
    sub_a["rwr5"]  = sub_a["away_win"].rolling(5,  min_periods=3).mean()
    sub_a["gap10"] = sub_a["rwr10"] - sub_a["imp_prob_A"]
    sub_a["gap5"]  = sub_a["rwr5"]  - sub_a["imp_prob_A"]

    g10_h = sub_h["gap10"].iloc[-10:].mean() * 100
    g5_h  = sub_h["gap5"].iloc[-5:].mean()   * 100
    g10_a = sub_a["gap10"].iloc[-10:].mean() * 100
    g5_a  = sub_a["gap5"].iloc[-5:].mean()   * 100

    # si gap5 > gap10 el value está creciendo — la racha es más fuerte que el histórico
    tendencia_h = "creciente" if g5_h > g10_h else "decreciente"
    tendencia_a = "creciente" if g5_a > g10_a else "decreciente"

    resumen_racha.append({
        "Equipo":       equipo,
        "Gap10 local":  f"{g10_h:+.1f}pp",
        "Gap5 local":   f"{g5_h:+.1f}pp",
        "Tendencia L":  tendencia_h,
        "Value L":      "SI CRECIENTE" if g5_h > 0 and tendencia_h == "creciente" else
                        "SI" if g5_h > 0 else "no",
        "Gap10 visita": f"{g10_a:+.1f}pp",
        "Gap5 visita":  f"{g5_a:+.1f}pp",
        "Tendencia V":  tendencia_a,
        "Value V":      "SI CRECIENTE" if g5_a > 0 and tendencia_a == "creciente" else
                        "SI" if g5_a > 0 else "no",
    })
    print(f"  {equipo}: local  gap10={g10_h:+.1f}pp gap5={g5_h:+.1f}pp ({tendencia_h})")
    print(f"  {' '*len(equipo)}  visita gap10={g10_a:+.1f}pp gap5={g5_a:+.1f}pp ({tendencia_a})")

print_tabulate(pd.DataFrame(resumen_racha))

# grafiqué gap5 vs gap10 para ver visualmente quién tiene value creciente

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
nombres_eq = [e[0] for e in equipos]
g10_locals  = [float(r["Gap10 local"].replace("pp",""))  for r in resumen_racha]
g5_locals   = [float(r["Gap5 local"].replace("pp",""))   for r in resumen_racha]
g10_visitas = [float(r["Gap10 visita"].replace("pp","")) for r in resumen_racha]
g5_visitas  = [float(r["Gap5 visita"].replace("pp",""))  for r in resumen_racha]
x = np.arange(len(nombres_eq))
w = 0.35

ax1.bar(x - w/2, g10_locals, w, label="gap 10 partidos", alpha=0.7,
        color=["#3498db" if g > 0 else "#e74c3c" for g in g10_locals])
ax1.bar(x + w/2, g5_locals,  w, label="gap 5 partidos",  alpha=0.85,
        color=["#27ae60" if g > 0 else "#c0392b" for g in g5_locals])
ax1.axhline(0, color="gray", linestyle="--", linewidth=0.9)
ax1.set_xticks(x); ax1.set_xticklabels(nombres_eq, rotation=15, fontsize=8)
ax1.set_ylabel("gap (pp)"); ax1.legend(fontsize=8)
ax1.set_title("Gap value LOCAL — últimas 5 vs 10 fechas", fontweight="bold")

ax2.bar(x - w/2, g10_visitas, w, label="gap 10 partidos", alpha=0.7,
        color=["#3498db" if g > 0 else "#e74c3c" for g in g10_visitas])
ax2.bar(x + w/2, g5_visitas,  w, label="gap 5 partidos",  alpha=0.85,
        color=["#27ae60" if g > 0 else "#c0392b" for g in g5_visitas])
ax2.axhline(0, color="gray", linestyle="--", linewidth=0.9)
ax2.set_xticks(x); ax2.set_xticklabels(nombres_eq, rotation=15, fontsize=8)
ax2.set_ylabel("gap (pp)"); ax2.legend(fontsize=8)
ax2.set_title("Gap value VISITANTE — últimas 5 vs 10 fechas", fontweight="bold")

plt.suptitle("Detección de value creciente: gap5 > gap10 → racha más fuerte que histórico",
             fontweight="bold", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "forecasting_racha_value.png"), dpi=130)
plt.close()
print("  guardado: img/forecasting_racha_value.png\n")


# analicé el over/under por equipo específico de local
# si un equipo mete sistemáticamente más goles de los que la cuota de over estima
# hay value en apostar over en sus partidos de local

print("\n17. over/under por equipo de local — ¿la casa estima bien los goles?")

resumen_ou_eq = []

for equipo, liga, col in equipos:
    sub = df[df["HomeTeam"] == equipo].copy().sort_values("Date").reset_index(drop=True)

    if len(sub) < 20:
        continue

    # calculé el over2.5 real del equipo de local vs la probabilidad implícita
    # que la casa le asigna (usando imp_prob de over como 1 - imp_prob_under)
    # como no tenemos cuota directa de over, uso el promedio de goles vs umbral

    over_real    = sub["over25"].mean() * 100
    avg_goles    = sub["total_goals"].mean()
    avg_goles_h  = sub["FTHG"].mean()
    avg_goles_a  = sub["FTAG"].mean()

    # rolling over rate últimas 10 fechas
    sub["rolling_over"] = sub["over25"].rolling(10, min_periods=5).mean()
    sub["rolling_goles"]= sub["total_goals"].rolling(10, min_periods=5).mean()

    over_rec    = sub["rolling_over"].iloc[-1] * 100
    goles_rec   = sub["rolling_goles"].iloc[-1]
    over_hist   = sub["over25"].mean() * 100
    gap_over    = over_rec - over_hist

    resumen_ou_eq.append({
        "Equipo":        equipo,
        "Liga":          LIGAS_NAME[liga],
        "Over2.5 hist":  f"{over_hist:.1f}%",
        "Over2.5 rec":   f"{over_rec:.1f}%",
        "Gap over":      f"{gap_over:+.1f}pp",
        "Avg goles":     round(avg_goles, 2),
        "Avg goles rec": round(goles_rec, 2),
        "Goles H avg":   round(avg_goles_h, 2),
        "Goles A avg":   round(avg_goles_a, 2),
        "Señal over":    "OVER" if gap_over > 5 else "UNDER" if gap_over < -5 else "neutro",
    })
    print(f"  {equipo}: over hist={over_hist:.1f}%  rec={over_rec:.1f}%  gap={gap_over:+.1f}pp  avg_goles={avg_goles:.2f}  señal={'OVER' if gap_over > 5 else 'UNDER' if gap_over < -5 else 'neutro'}")

print_tabulate(pd.DataFrame(resumen_ou_eq))

# grafiqué el over rate rodante por equipo para ver tendencia de goles

fig, axes = plt.subplots(len(equipos), 1, figsize=(14, 3.5 * len(equipos)), sharex=False)

for i, (equipo, liga, col) in enumerate(equipos):
    sub = df[df["HomeTeam"] == equipo].copy().sort_values("Date").reset_index(drop=True)
    if len(sub) < 20:
        continue

    sub["rolling_over"]  = sub["over25"].rolling(10, min_periods=5).mean() * 100
    sub["rolling_goles"] = sub["total_goals"].rolling(10, min_periods=5).mean()
    over_hist_line       = sub["over25"].mean() * 100

    ax = axes[i]
    ax.plot(sub["Date"], sub["rolling_over"],
            color=col, linewidth=2, label="over2.5 rodante 10 partidos")
    ax.axhline(over_hist_line, color="gray", linestyle="--", linewidth=1.2,
               label=f"media histórica {over_hist_line:.1f}%")
    ax.fill_between(sub["Date"], sub["rolling_over"], over_hist_line,
                    where=(sub["rolling_over"] > over_hist_line),
                    alpha=0.2, color="green", label="zona over")
    ax.fill_between(sub["Date"], sub["rolling_over"], over_hist_line,
                    where=(sub["rolling_over"] <= over_hist_line),
                    alpha=0.15, color="red", label="zona under")
    ax2_twin = ax.twinx()
    ax2_twin.plot(sub["Date"], sub["rolling_goles"],
                  color="black", linewidth=1.2, alpha=0.5, linestyle=":",
                  label="goles rodante")
    ax2_twin.set_ylabel("goles promedio", fontsize=7)
    ax.set_ylabel("% over 2.5"); ax.legend(fontsize=7); ax.set_ylim(0, 110)
    ax.set_title(f"{equipo} ({LIGAS_NAME[liga]}) — over2.5 rodante de local",
                 fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "forecasting_over_por_equipo.png"), dpi=130)
plt.close()
print("  guardado: img/forecasting_over_por_equipo.png")