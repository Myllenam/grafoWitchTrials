import pandas as pd

def preprocessar(df: pd.DataFrame) -> pd.DataFrame:
    df = df[(df['gadm_adm0'].notna()) & (df['gadm_adm0'] != '')]

    df['is_desconhecido'] = df[['city', 'gadm_adm1', 'gadm_adm0']].isnull().any(axis=1)

    df['tried'] = pd.to_numeric(df['tried'], errors='coerce').fillna(0).astype(int)
    df['deaths'] = pd.to_numeric(df['deaths'], errors='coerce').fillna(0).astype(int)

    df['localizacao'] = df.apply(
        lambda x: f"{x['city']}, {x['gadm_adm1']}" if pd.notna(x['gadm_adm1']) and x['gadm_adm1'] != '' else x['city'],
        axis=1
    ) + ", " + df['gadm_adm0']

    return df
