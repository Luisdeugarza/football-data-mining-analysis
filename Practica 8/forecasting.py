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


# cargué el dataset y construí todas las variables necesarias para los 6 análisis
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

# calculé el bias como diferencia entre probabilidad implícita y resultado real
# bias positivo significa que la casa sobreestima la probabilidad del local

df["bias_H"] = round(df["imp_prob_H"] - df["home_win"], 4)
df["bias_A"] = round(df["imp_prob_A"] - df["away_win"], 4)
df["bias_D"] = round(df["imp_prob_D"] - df["draw"],     4)

df["Season_label"] = df["Season"].map(SEASON_MAP)


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

    X     = sm.add_constant(ts["t"])
    model = sm.OLS(ts["valor"], X).fit()

    m = model.params["t"]
    b = model.params["const"]

    t_last    = ts["t"].max()
    f_last    = ts["Fecha"].max()
    fut_t     = np.arange(t_last + 1, t_last + 1 + n_weeks)
    fut_dates = pd.date_range(start=f_last + pd.Timedelta(weeks=1),
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
    mae  = float(np.mean(np.abs(test["valor"].values - pred_test.values)))
    rmse = float(np.sqrt(np.mean((test["valor"].values - pred_test.values) ** 2)))
    return model, train, test, pred_test, mae, rmse


ligas = list(LIGAS_NAME.keys())


# analicé el % over 2.5 por liga como serie de tiempo semanal
# elegí esta métrica porque es la línea más transaccionada en el mercado europeo

print("over 2.5 por liga")

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
    ax.set_ylabel("% over 2.5")
    ax.set_title(
        f"{LIGAS_NAME[liga]}  tendencia: {m_f*100:+.4f}pp/sem  R²={model_full.rsquared:.4f}  p={model_full.pvalues['t']:.4f}",
        fontweight="bold")
    ax.legend(fontsize=7)
    ax.set_ylim(20, 90)
    print(f"  {LIGAS_NAME[liga]}: {m_f*100:+.5f}pp/sem  R²={model_full.rsquared:.4f}  p={model_full.pvalues['t']:.4f}  MAE={mae*100:.2f}pp  pred+16w={fut_pct.iloc[-1]:.1f}%")

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "forecasting_over25_por_liga.png"), dpi=130)
plt.close()
print("  guardado: img/forecasting_over25_por_liga.png\n")
print_tabulate(pd.DataFrame(resumen_o25))


# analicé el % de victorias local por liga para detectar si la home advantage
# tiene tendencia lineal — un descenso sostenido obligaría a reajustar cuotas

print("\nvictorias local por liga")

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
    ax.set_ylabel("% victoria local")
    ax.set_title(
        f"{LIGAS_NAME[liga]}  tendencia: {m_f*100:+.4f}pp/sem  R²={model_full.rsquared:.4f}  p={model_full.pvalues['t']:.4f}",
        fontweight="bold")
    ax.legend(fontsize=7)
    ax.set_ylim(20, 75)
    print(f"  {LIGAS_NAME[liga]}: {m_f*100:+.5f}pp/sem  R²={model_full.rsquared:.4f}  p={model_full.pvalues['t']:.4f}  MAE={mae*100:.2f}pp  pred+16w={fut_pct.iloc[-1]:.1f}%")

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "forecasting_homewin_por_liga.png"), dpi=130)
plt.close()
print("  guardado: img/forecasting_homewin_por_liga.png\n")
print_tabulate(pd.DataFrame(resumen_hw))


# analicé el movimiento de cuotas de apertura a cierre (odds_move_H)
# si la tendencia es consistente en una dirección, el smart money está corrigiendo
# un sesgo sistemático en la apertura de la casa

print("\nmovimiento de odds apertura a cierre por liga")

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
    ax.plot(ts_full["Fecha"], yp,
            color="red", linewidth=1.8,
            label=f"tendencia  R²={model_full.rsquared:.4f}  p={model_full.pvalues['t']:.4f}")
    ax.plot(fut_full["Fecha"], fut_full["pred"],
            color="orange", linewidth=2, linestyle="--",
            label=f"pred +16w  {fut_full['pred'].iloc[-1]:.4f}")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axvline(test["Fecha"].iloc[0], color="gray", linestyle=":", linewidth=1)
    ax.set_ylabel("odds_move_H (cierre - apertura)")
    ax.set_title(
        f"{LIGAS_NAME[liga]}  tendencia: {m_f:+.6f}/sem  R²={model_full.rsquared:.4f}  p={model_full.pvalues['t']:.4f}",
        fontweight="bold")
    ax.legend(fontsize=7)
    print(f"  {LIGAS_NAME[liga]}: media={ts_full['valor'].mean():+.5f}  tendencia={m_f:+.6f}/sem  R²={model_full.rsquared:.4f}  p={model_full.pvalues['t']:.4f}")

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "forecasting_odds_move.png"), dpi=130)
plt.close()
print("  guardado: img/forecasting_odds_move.png\n")
print_tabulate(pd.DataFrame(resumen_mov))


# analicé el overround semanal para ver si el margen de la casa sube o baja
# un margen creciente indica que la casa se vuelve menos competitiva con el tiempo

print("\noverround semanal por liga")

resumen_or = []
fig, axes  = plt.subplots(len(ligas), 1, figsize=(14, 4 * len(ligas)), sharex=False)

for i, liga in enumerate(ligas):
    sub = df[df["Div"] == liga].copy()
    model_full, ts_full, fut_full = forecast_serie(sub, "Date", "overround", n_weeks=16)
    model_tr, train, test, pred_test, mae, rmse = train_test_ols(ts_full)

    m_f         = model_full.params["t"]
    col         = LIGA_COLORS[liga]
    yp          = model_full.predict(sm.add_constant(ts_full["t"]))
    margen_medio = (ts_full["valor"].mean() - 1) * 100

    resumen_or.append({
        "Liga":           LIGAS_NAME[liga],
        "Overround avg":  round(ts_full["valor"].mean(), 5),
        "Margen casa %":  f"{margen_medio:.2f}%",
        "Tendencia/sem":  round(m_f, 6),
        "R²":             round(model_full.rsquared, 4),
        "p-value":        round(model_full.pvalues["t"], 4),
        "Pred +16w":      round(fut_full["pred"].iloc[-1], 5),
    })

    ax = axes[i]
    ax.plot(ts_full["Fecha"], (ts_full["valor"] - 1) * 100,
            color=col, alpha=0.45, linewidth=1, label="margen semanal %")
    ax.plot(ts_full["Fecha"], (yp - 1) * 100,
            color="red", linewidth=1.8,
            label=f"tendencia  R²={model_full.rsquared:.4f}  p={model_full.pvalues['t']:.4f}")
    ax.plot(fut_full["Fecha"], (fut_full["pred"] - 1) * 100,
            color="orange", linewidth=2, linestyle="--",
            label=f"pred +16w  {(fut_full['pred'].iloc[-1]-1)*100:.2f}%")
    ax.axvline(test["Fecha"].iloc[0], color="gray", linestyle=":", linewidth=1)
    ax.set_ylabel("margen casa (%)")
    ax.set_title(
        f"{LIGAS_NAME[liga]}  margen medio: {margen_medio:.2f}%  tendencia: {m_f*100:+.5f}pp/sem  R²={model_full.rsquared:.4f}",
        fontweight="bold")
    ax.legend(fontsize=7)
    print(f"  {LIGAS_NAME[liga]}: margen={margen_medio:.2f}%  tendencia={m_f*100:+.5f}pp/sem  R²={model_full.rsquared:.4f}  p={model_full.pvalues['t']:.4f}")

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "forecasting_overround.png"), dpi=130)
plt.close()
print("  guardado: img/forecasting_overround.png\n")
print_tabulate(pd.DataFrame(resumen_or))


# bajé al nivel de equipo específico para mostrar que el análisis es granular
# usé períodos de 4 semanas porque cada equipo juega pocos partidos de local por mes (al menos en liga ya que juegan otras compes)

print("\nvictorias local por equipo específico")

equipos = [
    ("Man City",     "E0",  "#6caddf"),
    ("Arsenal",      "E0",  "#ef0107"),
    ("Real Madrid",  "SP1", "#ffd700"),
    ("Barcelona",    "SP1", "#a50044"),
    ("Bayern Munich","D1",  "#dc052d"),
    ("Juventus",     "I1",  "#000000"),
    ("Paris SG",     "F1",  "#004170"),
]

resumen_eq = []
fig, axes  = plt.subplots(len(equipos), 1, figsize=(14, 4 * len(equipos)), sharex=False)

for i, (equipo, liga, col) in enumerate(equipos):
    sub = df[df["HomeTeam"] == equipo].copy()

    if len(sub) < 20:
        print(f"  {equipo}: pocos datos ({len(sub)} partidos), omitido")
        continue

    model_full, ts_full, fut_full = forecast_serie(sub, "Date", "home_win",
                                                    n_weeks=16, freq="4W")
    model_tr, train, test, pred_test, mae, rmse = train_test_ols(ts_full)

    m_f = model_full.params["t"]
    yp  = model_full.predict(sm.add_constant(ts_full["t"]))

    resumen_eq.append({
        "Equipo":        equipo,
        "Liga":          LIGAS_NAME[liga],
        "Partidos":      len(sub),
        "Media %local":  f"{ts_full['valor'].mean()*100:.1f}%",
        "Tendencia/per": round(m_f * 100, 4),
        "R²":            round(model_full.rsquared, 4),
        "p-value":       round(model_full.pvalues["t"], 4),
        "Pred +16w":     f"{fut_full['pred'].iloc[-1]*100:.1f}%",
    })

    ax = axes[i]
    ax.plot(ts_full["Fecha"], ts_full["valor"] * 100,
            color=col, alpha=0.55, linewidth=1.2, label="observado (4W)")
    ax.plot(ts_full["Fecha"], yp * 100,
            color="red", linewidth=1.8,
            label=f"tendencia  R²={model_full.rsquared:.4f}  p={model_full.pvalues['t']:.4f}")
    ax.scatter(test["Fecha"], test["valor"] * 100,
               color="black", s=14, zorder=5, label=f"test  MAE={mae*100:.1f}pp")
    ax.plot(fut_full["Fecha"], fut_full["pred"] * 100,
            color="orange", linewidth=2, linestyle="--",
            label=f"pred  {fut_full['pred'].iloc[-1]*100:.1f}%")
    ax.axvline(test["Fecha"].iloc[0], color="gray", linestyle=":", linewidth=1)
    ax.set_ylabel("% victorias local")
    ax.set_title(
        f"{equipo} ({LIGAS_NAME[liga]})  tendencia: {m_f*100:+.4f}pp/per  R²={model_full.rsquared:.4f}",
        fontweight="bold")
    ax.legend(fontsize=7)
    ax.set_ylim(-10, 120)
    print(f"  {equipo}: media={ts_full['valor'].mean()*100:.1f}%  tendencia={m_f*100:+.4f}pp/per  R²={model_full.rsquared:.4f}  p={model_full.pvalues['t']:.4f}")

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "forecasting_equipos.png"), dpi=130)
plt.close()
print("  guardado: img/forecasting_equipos.png\n")
print_tabulate(pd.DataFrame(resumen_eq))


# calculé el bias semanal entre probabilidad implícita y resultado real
# un bias con tendencia lineal significativa indica una ineficiencia estructural de las casas de apuestas
# detectable — la casa está sistemáticamente equivocada en una dirección

print("\nbias probabilidad implícita vs resultado real por liga")

resumen_bias = []
fig, axes    = plt.subplots(len(ligas), 1, figsize=(14, 4 * len(ligas)), sharex=False)

for i, liga in enumerate(ligas):
    sub = df[df["Div"] == liga].copy()
    model_full, ts_full, fut_full = forecast_serie(sub, "Date", "bias_H", n_weeks=16)
    model_tr, train, test, pred_test, mae, rmse = train_test_ols(ts_full)

    m_f   = model_full.params["t"]
    col   = LIGA_COLORS[liga]
    yp    = model_full.predict(sm.add_constant(ts_full["t"]))
    media = ts_full["valor"].mean()

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
    ax.plot(ts_full["Fecha"], yp,
            color="red", linewidth=1.8,
            label=f"tendencia  R²={model_full.rsquared:.4f}  p={model_full.pvalues['t']:.4f}")
    ax.plot(fut_full["Fecha"], fut_full["pred"],
            color="orange", linewidth=2, linestyle="--",
            label=f"pred +16w  {fut_full['pred'].iloc[-1]:.4f}")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.9, alpha=0.7,
               label="bias=0 (mercado perfecto)")
    ax.axvline(test["Fecha"].iloc[0], color="gray", linestyle=":", linewidth=1)
    ax.set_ylabel("bias (imp_prob_H - home_win)")
    ax.set_title(
        f"{LIGAS_NAME[liga]}  bias medio: {media:+.4f} ({direccion})  R²={model_full.rsquared:.4f}  p={model_full.pvalues['t']:.4f}",
        fontweight="bold")
    ax.legend(fontsize=7)
    print(f"  {LIGAS_NAME[liga]}: bias={media:+.5f} ({direccion})  tendencia={m_f:+.6f}/sem  R²={model_full.rsquared:.4f}  p={model_full.pvalues['t']:.4f}")

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "forecasting_bias.png"), dpi=130)
plt.close()
print("  guardado: img/forecasting_bias.png\n")
print_tabulate(pd.DataFrame(resumen_bias))