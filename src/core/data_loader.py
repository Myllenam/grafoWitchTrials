import pandas as pd

def carregar_dados(caminho_csv: str) -> pd.DataFrame:
    df = pd.read_csv(caminho_csv, sep=',')
    df.columns = df.columns.str.strip()
    return df
