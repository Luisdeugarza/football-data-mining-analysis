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

os.makedirs("img", exist_ok=True)

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


# cargué el dataset limpio y construí las variables de goles que necesito para la serie de tiempo

df = pd.read_csv("../Practica 1/data/clean/football_clean.csv", parse_dates=["Date"])

df["total_goals"] = df["FTHG"] + df["FTAG"]
df["ht_goals"]    = df["HTHG"] + df["HTAG"]
df["home_win"]    = (df["FTR"] == "H").astype(int)
df["over25"]      = (df["total_goals"] > 2).astype(int)
df["btts"]        = ((df["FTHG"] > 0) & (df["FTAG"] > 0)).astype(int)


# agrupé los partidos por semana y calculé el promedio de goles por jornada
# usé resample('W') sobre el índice de fecha para tener una observación por semana

df_ts = (
    df.set_index("Date")["total_goals"]
    .resample("W")
    .mean()
    .dropna()
    .reset_index()
)
df_ts.columns = ["Fecha", "avg_goals"]
df_ts["t"] = range(len(df_ts))

print("=" * 60)
print("  SERIE DE TIEMPO — promedio semanal de goles por partido")
print("=" * 60)
print(f"\n  semanas totales : {len(df_ts)}")
print(f"  desde           : {df_ts['Fecha'].min().date()}")
print(f"  hasta           : {df_ts['Fecha'].max().date()}")
print(f"  media global    : {df_ts['avg_goals'].mean():.4f}")
print(f"  desv. estándar  : {df_ts['avg_goals'].std():.4f}")
print()
print_tabulate(df_ts.head(10))


# ajusté la regresión lineal usando OLS de statsmodels con t como variable independiente
# convertí la fecha a índice numérico porque OLS no acepta datetime directamente

X = sm.add_constant(df_ts["t"])
model = sm.OLS(df_ts["avg_goals"], X).fit()

print("\n" + "=" * 60)
print("  RESULTADOS DE LA REGRESIÓN LINEAL")
print("=" * 60)
print(model.summary())

b     = model.params["const"]
m     = model.params["t"]
r2    = model.rsquared
r2adj = model.rsquared_adj
pval  = model.pvalues["t"]

conf  = model.conf_int()
lo    = conf.loc["const", 0]
hi    = conf.loc["const", 1]

print(f"\n  pendiente (m)  : {m:.6f} goles por semana")
print(f"  intercepto (b) : {b:.4f}")
print(f"  R²             : {r2:.4f}")
print(f"  R² ajustado    : {r2adj:.4f}")
print(f"  p-value (t)    : {pval:.4f}")


# generé la gráfica de la serie de tiempo con su línea de tendencia y banda de confianza

y_pred  = model.predict(X)
y_lo    = model.get_prediction(X).conf_int()[:, 0]
y_hi    = model.get_prediction(X).conf_int()[:, 1]

fig, ax = plt.subplots(figsize=(14, 5))
ax.scatter(df_ts["Fecha"], df_ts["avg_goals"],
           alpha=0.4, s=12, color="#3498db", label="avg goles por semana")
ax.plot(df_ts["Fecha"], y_pred,
        color="red", linewidth=2,
        label=f"tendencia lineal  R²={r2:.4f}  p={pval:.4f}")
ax.fill_between(df_ts["Fecha"], y_lo, y_hi, alpha=0.15, color="red",
                label="intervalo de confianza 95%")
ax.set_xlabel("Fecha")
ax.set_ylabel("avg goles por partido")
ax.set_title("Serie temporal: promedio semanal de goles con regresión lineal",
             fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig("img/forecasting_serie_goles.png", dpi=130)
plt.close()
print("\n  gráfica guardada: img/forecasting_serie_goles.png")


# predije las próximas 12 semanas usando la ecuación del modelo: ŷ = m*t + b
# extendí el índice t más allá del último punto observado

n_weeks   = 12
t_last    = df_ts["t"].max()
fecha_last = df_ts["Fecha"].max()

future_t     = np.arange(t_last + 1, t_last + 1 + n_weeks)
future_dates = pd.date_range(start=fecha_last + pd.Timedelta(weeks=1),
                             periods=n_weeks, freq="W")
future_pred  = m * future_t + b

df_future = pd.DataFrame({
    "Semana": future_dates.date,
    "t":      future_t,
    "avg_goals_predicho": np.round(future_pred, 4),
})

print("\n" + "=" * 60)
print("  PREDICCIÓN — próximas 12 semanas")
print("=" * 60)
print_tabulate(df_future)


# grafiqué la serie histórica junto con el tramo predicho para visualizar el forecast

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df_ts["Fecha"], df_ts["avg_goals"],
        color="#3498db", linewidth=1, alpha=0.7, label="histórico")
ax.plot(df_ts["Fecha"], y_pred,
        color="red", linewidth=2, label=f"tendencia  R²={r2:.4f}")
ax.plot(future_dates, future_pred,
        color="orange", linewidth=2, linestyle="--", label="predicción (12 semanas)")
ax.axvline(fecha_last, color="gray", linestyle=":", linewidth=1.2, label="hoy")
ax.set_xlabel("Fecha")
ax.set_ylabel("avg goles por partido")
ax.set_title("Forecasting: tendencia histórica + predicción a 12 semanas",
             fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig("img/forecasting_prediccion_12w.png", dpi=130)
plt.close()
print("  gráfica guardada: img/forecasting_prediccion_12w.png")