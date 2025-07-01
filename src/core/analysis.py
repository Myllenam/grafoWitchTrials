import pandas as pd
import networkx as nx
from core.visualization import (
    top_paises,
    top_localizacoes,
    julgamentos_por_decada,
    distribuir_geograficamente
)

def total_julgamentos(df: pd.DataFrame) -> int:
    return df['tried'].sum()

def total_mortes(df: pd.DataFrame) -> int:
    return df['deaths'].sum()

def total_registros_desconhecidos(df: pd.DataFrame) -> int:
    return df['is_desconhecido'].sum()

def taxa_mortalidade(df: pd.DataFrame) -> float:
    try:
        return df['deaths'].sum() / df['tried'].sum()
    except ZeroDivisionError:
        return 0.0

def mostrar_taxa_mortalidade(df: pd.DataFrame):
    total_j = total_julgamentos(df)
    total_m = total_mortes(df)
    taxa = taxa_mortalidade(df)
    print("\nTaxa de Mortalidade Geral:")
    print(f"Total de Julgamentos: {total_j}")
    print(f"Total de Mortes: {total_m}")
    print(f"Taxa de Mortalidade Geral (Mortes/Julgamentos): {taxa:.4f}")

def calcular_taxas_mortalidade_geral(df: pd.DataFrame):
    if df.empty:
        print("DataFrame não carregado ou está vazio para calcular taxas de mortalidade.")
        return None, None, None

    total_j = total_julgamentos(df)
    total_m = total_mortes(df)
    taxa = taxa_mortalidade(df)

    print(f"\nTaxa de Mortalidade Geral:")
    print(f"Total de Julgamentos: {total_j}")
    print(f"Total de Mortes: {total_m}")
    print(f"Taxa de Mortalidade Geral (Mortes/Julgamentos): {taxa:.4f}")

    return total_j, total_m, taxa

def executar_analise_completa(df: pd.DataFrame, G: nx.DiGraph):
    print("\n=== ANÁLISE DE JULGAMENTOS HISTÓRICOS INICIADA ===\n")
    resultados = {}

    print("1. Regiões com maior número de julgamentos e mortes (por País):")
    resultados['top_paises'] = top_paises(df, G)

    print("\n1b. Regiões com maior número de julgamentos e mortes (por Localização detalhada):")
    resultados['top_localizacoes'] = top_localizacoes(df, G)

    print("\n2. Padrões temporais (ondas de perseguição):")
    resultados['analise_temporal'] = julgamentos_por_decada(df, G)

    print("\n3. Distribuição geográfica (correlação entre proximidade geográfica e número de julgamentos - visual):")
    resultados['distribuicao_geografica'] = distribuir_geograficamente(df, G)

    print("\n4. Taxas de mortalidade baseada na quantidade de julgamentos:")
    resultados['taxas_mortalidade_geral'] = calcular_taxas_mortalidade_geral(df)

    print("\n=== ANÁLISE CONCLUÍDA ===")
    return resultados
