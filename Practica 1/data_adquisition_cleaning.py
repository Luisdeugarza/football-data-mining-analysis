import requests
import io
import pandas as pd
from tabulate import tabulate

def get_csv_from_url(url: str, encoding: str = 'latin1') -> pd.DataFrame:
    s = requests.get(url).content
    return pd.read_csv(io.StringIO(s.decode(encoding)))

def print_tabulate(df: pd.DataFrame):
    print(tabulate(df, headers=df.columns, tablefmt='orgtbl'))

def descargar_temporada(liga: str, temporada: str) -> pd.DataFrame:
    url = f"https://www.football-data.co.uk/mmz4281/{temporada}/{liga}.csv"
    try:
        df = get_csv_from_url(url)
        df["Season"] = temporada
        if "Div" not in df.columns:
            df["Div"] = liga
        print(f"{liga} {temporada}: {len(df)} filas")
        return df
    except Exception as e:
        print(f"error {liga} {temporada}: {e}")
        return pd.DataFrame()

def filtrar_columnas(df: pd.DataFrame, columnas: list) -> pd.DataFrame:
    disponibles = [c for c in columnas if c in df.columns]
    return df[disponibles]

def limpiar_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(how="all")
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam"])
    df = df.dropna(subset=["HTHG", "HTAG", "HTR"]) #esto es por un solo partido que no tiene dato de tiempo

    for col in ["FTR", "HTR"]:
        if col in df.columns:
            df[col] = df[col].str.strip().str.upper()
            df[col] = df[col].where(df[col].isin(["H", "D", "A"]), other=pd.NA)

    for col in ["FTHG", "FTAG", "HTHG", "HTAG"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    for col in odds_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # reemplazo de nulos de odds con la mediana de cada columna
    for col in odds_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    df = df.drop_duplicates(subset=["Div", "Date", "HomeTeam", "AwayTeam"])
    df = df.sort_values(["Date", "Div"]).reset_index(drop=True)
    return df

# top 5 ligas europeas
ligas = ["E0", "SP1", "D1", "I1", "F1"]
temporadas = ["1920", "2021", "2122", "2223", "2324", "2425", "2526"]

columnas_resultado = ["Div", "Season", "Date", "HomeTeam", "AwayTeam",
                      "FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR"]

# se eliminó Pinnacle por irregularidades dichos por el creador de la api y nulos excesivos +250
odds_cols = ["B365H", "B365D", "B365A",
             "MaxH", "MaxD", "MaxA",
             "AvgH", "AvgD", "AvgA",
             "B365CH", "B365CD", "B365CA",
             "MaxCH", "MaxCD", "MaxCA",
             "AvgCH", "AvgCD", "AvgCA"]

ldfs = []
for temporada in temporadas:
    for liga in ligas:
        df = descargar_temporada(liga, temporada)
        if not df.empty:
            ldfs.append(filtrar_columnas(df, columnas_resultado + odds_cols))

combined = pd.concat(ldfs, ignore_index=True)
combined.to_csv("data/raw/football_raw.csv", index=False)
print(f"\nraw: {combined.shape}")

df_clean = limpiar_dataset(combined)
print(df_clean.isnull().sum())
df_clean.to_csv("data/clean/football_clean.csv", index=False)

print(f"clean: {df_clean.shape}")
print(f"fechas: {df_clean['Date'].min().date()} - {df_clean['Date'].max().date()}")
print_tabulate(df_clean.head(10))