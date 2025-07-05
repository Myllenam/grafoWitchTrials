import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import os

def visualizar_grafo(G: nx.DiGraph, title="Visualização do Grafo"):
    if not G.nodes:
        print("Grafo está vazio. Nada a visualizar.")
        return

    pos = nx.spring_layout(G, seed=42)
    node_colors = [
        'skyblue' if G.nodes[n].get('tipo') == 'local'
        else 'lightgreen' if G.nodes[n].get('tipo') == 'decada'
        else 'gray'
        for n in G.nodes()
    ]
    edge_weights = np.array([data.get('weight', 1) for _, _, data in G.edges(data=True)])
    edge_widths = 0.5 + 5 * (edge_weights / edge_weights.max()) if len(edge_weights) > 0 and edge_weights.max() > 0 else 0.5

    plt.figure(figsize=(18, 12))
    nx.draw_networkx(
        G, pos,
        node_color=node_colors,
        edge_color='gray',
        width=edge_widths,
        alpha=0.7,
        with_labels=True,
        font_size=7,
        node_size=1200,
        arrows=True,
        arrowsize=10
    )
    plt.title(title, size=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def top_paises(df: pd.DataFrame, G: nx.DiGraph, n=10):
    paises = defaultdict(lambda: {'Julgamentos': 0, 'Mortes': 0})
    for no in G.nodes():
        if G.nodes[no].get('tipo') == 'local' and 'pais' in G.nodes[no]:
            pais = G.nodes[no]['pais']
            julgamentos = sum(data.get('weight', 0) for _, _, data in G.out_edges(no, data=True))
            mortes = sum(data.get('mortes', 0) for _, _, data in G.out_edges(no, data=True))
            paises[pais]['Julgamentos'] += julgamentos
            paises[pais]['Mortes'] += mortes

    df_out = pd.DataFrame([{
        'País': pais,
        'Julgamentos': dados['Julgamentos'],
        'Mortes': dados['Mortes'],
        'Taxa Mortalidade': dados['Mortes'] / dados['Julgamentos'] if dados['Julgamentos'] > 0 else 0
    } for pais, dados in paises.items() if dados['Julgamentos'] > 0]).sort_values('Julgamentos', ascending=False).head(n)

    if df_out.empty:
        print("Nenhum país com julgamentos encontrado.")
        return df_out

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    df_out.plot.barh(x='País', y='Julgamentos', ax=ax[0], color='darkred', legend=False)
    ax[0].set_title('Top Países por Julgamentos')
    df_out.plot.barh(x='País', y='Taxa Mortalidade', ax=ax[1], color='darkblue', legend=False)
    ax[1].set_title('Taxa de Mortalidade por País')
    plt.tight_layout()
    plt.show()

    visualizar_grafo(G, "Grafo após Análise de Top Países")
    return df_out

def top_localizacoes(df: pd.DataFrame, G: nx.DiGraph, n=10):
    locais_data = defaultdict(lambda: {'Julgamentos': 0, 'Mortes': 0})
    for no in G.nodes():
        if G.nodes[no].get('tipo') == 'local':
            julgamentos = sum(data.get('weight', 0) for _, _, data in G.out_edges(no, data=True))
            mortes = sum(data.get('mortes', 0) for _, _, data in G.out_edges(no, data=True))
            locais_data[no]['Julgamentos'] += julgamentos
            locais_data[no]['Mortes'] += mortes

    df_out = pd.DataFrame([{
        'Localização': loc,
        'Julgamentos': dados['Julgamentos'],
        'Mortes': dados['Mortes'],
        'Taxa Mortalidade': dados['Mortes'] / dados['Julgamentos'] if dados['Julgamentos'] > 0 else 0
    } for loc, dados in locais_data.items() if dados['Julgamentos'] > 0]).sort_values('Julgamentos', ascending=False).head(n)

    if df_out.empty:
        print("Nenhuma localização com julgamentos encontrada.")
        return df_out

    fig, ax = plt.subplots(1, 2, figsize=(18, 7))
    df_out.plot.barh(x='Localização', y='Julgamentos', ax=ax[0], color='darkgreen', legend=False)
    ax[0].set_title('Top Localizações por Julgamentos')
    df_out.plot.barh(x='Localização', y='Taxa Mortalidade', ax=ax[1], color='darkorange', legend=False)
    ax[1].set_title('Taxa de Mortalidade nas Top Localizações')
    ax[0].invert_yaxis()
    ax[1].invert_yaxis()
    plt.tight_layout()
    plt.show()

    visualizar_grafo(G, "Grafo após Análise de Top Localizações")
    return df_out

def julgamentos_por_decada(df: pd.DataFrame, G: nx.DiGraph):
    decadas = sorted([n for n in G.nodes()
                      if G.nodes[n].get('tipo') == 'decada'
                      and str(n).isdigit()],
                     key=lambda x: int(x))

    ts_data = []
    for decada in decadas:
        julgamentos = sum(data.get('weight', 0) for _, _, data in G.in_edges(decada, data=True))
        mortes = sum(data.get('mortes', 0) for _, _, data in G.in_edges(decada, data=True))
        if julgamentos > 0:
            ts_data.append({
                'Década': int(decada),
                'Julgamentos': julgamentos,
                'Mortes': mortes,
                'Taxa Mortalidade': mortes / julgamentos
            })

    ts_df = pd.DataFrame(ts_data).set_index('Década').sort_index()
    if ts_df.empty:
        print("Nenhuma década com julgamentos encontrada.")
        return ts_df

    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    ts_df['Julgamentos'].plot(ax=ax[0], marker='o', color='darkred', title='Julgamentos por Década')
    ts_df['Taxa Mortalidade'].plot(ax=ax[1], marker='o', color='darkblue', title='Taxa de Mortalidade por Década')
    ax[0].grid(True, linestyle='--', alpha=0.6)
    ax[1].grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    visualizar_grafo(G, "Grafo após Análise Temporal")
    return ts_df

def distribuir_geograficamente(df: pd.DataFrame, G: nx.DiGraph):
    locais = [n for n in G.nodes()
              if G.nodes[n].get('tipo') == 'local'
              and G.nodes[n].get('lat') is not None
              and G.nodes[n].get('lon') is not None]

    if not locais:
        print("Nenhum dado geográfico completo disponível.")
        return pd.DataFrame()

    geo_data = [{
        'Local': local,
        'Julgamentos': sum(data.get('weight', 0) for _, _, data in G.out_edges(local, data=True)),
        'Mortes': sum(data.get('mortes', 0) for _, _, data in G.out_edges(local, data=True)),
        'Latitude': G.nodes[local]['lat'],
        'Longitude': G.nodes[local]['lon'],
        'País': G.nodes[local]['pais']
    } for local in locais]

    geo_df = pd.DataFrame(geo_data)
    plt.figure(figsize=(14, 7))
    scatter = plt.scatter(geo_df['Longitude'], geo_df['Latitude'],
                          s=geo_df['Julgamentos']/5 + 5,
                          c=geo_df['Julgamentos'], cmap='YlOrRd',
                          alpha=0.7, edgecolors='w', linewidth=0.5)

    plt.colorbar(scatter, label='Número de Julgamentos')
    plt.title('Distribuição Geográfica dos Julgamentos')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    visualizar_grafo(G, "Grafo após Análise de Distribuição Geográfica")
    return geo_df

def visualizar_subgrafo(G, nodes, titulo="Subgrafo"):
    """
    Visualiza e salva um subgrafo construído a partir de um conjunto de nós.
    Filtra também as arestas entre esses nós.
    """
    # Cria subgrafo real (com dados copiados)
    subG = nx.Graph()
    subG.add_nodes_from((n, G.nodes[n]) for n in nodes)

    for u, v in G.edges():
        if u in nodes and v in nodes:
            subG.add_edge(u, v, **G.edges[u, v])

    if subG.number_of_nodes() == 0 or subG.number_of_edges() == 0:
        print(f"[AVISO] O subgrafo '{titulo}' não possui dados suficientes para visualização.")
        return

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(subG, seed=42)

    graus = dict(subG.degree())
    cores = ['skyblue' if d.get('tipo') == 'local' else 'lightgreen' for _, d in subG.nodes(data=True)]

    nx.draw(subG, pos,
            with_labels=True,
            node_color=cores,
            node_size=[graus[n]*100 for n in subG.nodes()],
            font_size=6,
            edge_color='gray',
            alpha=0.8)

    plt.title(titulo)
    plt.tight_layout()

    os.makedirs("outputs", exist_ok=True)
    nome_arquivo = f"outputs/{titulo.lower().replace(' ', '_').replace(':', '')}.png"
    plt.savefig(nome_arquivo)
    print(f"[✔] Imagem salva: {nome_arquivo}")
    plt.show()
