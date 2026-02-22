# Práctica 1 — Adquisición y Limpieza de Datos

Descarga automatizada y limpieza del dataset de resultados y cuotas de fútbol europeo desde [football-data.co.uk](https://www.football-data.co.uk).

## Objetivo

Construir un dataset limpio, consistente y listo para análisis a partir de fuentes crudas distribuidas en múltiples archivos CSV por liga y temporada.

## Estructura

```
Practica 1/
├── data/
│   ├── raw/football_raw.csv
│   └── clean/football_clean.csv
└── data_adquisition_cleaning.py
```

## Dataset

| Parámetro | Valor |
|-----------|-------|
| Fuente | football-data.co.uk |
| Ligas | E0, SP1, D1, I1, F1 |
| Temporadas | 2019/20 — 2025/26 |
| Patrón URL | `https://www.football-data.co.uk/mmz4281/{season}/{league}.csv` |
| Filas (raw) | ~12,000 partidos |
| Filas (clean) | 11,851 partidos |
| Columnas (clean) | 29 variables |

## Variables del dataset limpio

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `Div` | str | Liga (E0, SP1, D1, I1, F1) |
| `Season` | int | Temporada codificada (ej. 1920 = 2019/20) |
| `Date` | datetime | Fecha del partido |
| `HomeTeam` | str | Equipo local |
| `AwayTeam` | str | Equipo visitante |
| `FTHG` | int | Goles local al final |
| `FTAG` | int | Goles visitante al final |
| `FTR` | str | Resultado final (H/D/A) |
| `HTHG` | int | Goles local al medio tiempo |
| `HTAG` | int | Goles visitante al medio tiempo |
| `HTR` | str | Resultado medio tiempo (H/D/A) |
| `B365H/D/A` | float | Cuotas apertura Bet365 |
| `MaxH/D/A` | float | Cuota máxima apertura |
| `AvgH/D/A` | float | Cuota promedio apertura |
| `B365CH/D/A` | float | Cuotas cierre Bet365 |
| `MaxCH/D/A` | float | Cuota máxima cierre |
| `AvgCH/D/A` | float | Cuota promedio cierre |

## Proceso de limpieza

### 1. Descarga

```python
URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
```

Se itera sobre las 5 ligas × 7 temporadas para descargar y concatenar en un solo DataFrame.

### 2. Decisiones de diseño

| Decisión | Razón |
|----------|-------|
| Eliminar columnas Pinnacle (PSH/PSD/PSA) | ~250 nulos desde julio 2025, fuente poco confiable |
| No usar `Latest_Results.csv` | Solo contiene los últimos 40 partidos — estadísticamente insuficiente |
| Mantener fechas como `datetime` | Permite agrupaciones por mes, año y series de tiempo en prácticas futuras |
| Imputar nulos residuales con mediana | 1-5 nulos por columna en odds — mediana es robusta a outliers |
| Eliminar filas con HTHG/HTAG/HTR nulos | Solo 1 fila afectada, no imputar datos de medio tiempo |

### 3. Transformaciones aplicadas

```python
# parseo de fechas
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

# tipos correctos
df[["FTHG","FTAG","HTHG","HTAG"]] = df[["FTHG","FTAG","HTHG","HTAG"]].astype("Int64")

# estandarización de resultados
df["FTR"] = df["FTR"].str.strip().str.upper()   # H, D, A

# deduplicación
df = df.drop_duplicates(subset=["Div","Date","HomeTeam","AwayTeam"])

# imputación de odds
for col in odds_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(df[col].median())
```

## Cumplimiento de requisitos

| Requisito | Valor |
|-----------|-------|
| Filas | 11,851 ✓ (mínimo 5,000) |
| Variables numéricas | FTHG, FTAG, HTHG, HTAG, todas las odds ✓ (mínimo 2) |
| Variables alfanuméricas | HomeTeam, AwayTeam, FTR, HTR, Div ✓ (mínimo 1) |
| Variable fecha | Date ✓ (mínimo 1) |
| Nulos en dataset final | 0 en todas las columnas ✓ |
