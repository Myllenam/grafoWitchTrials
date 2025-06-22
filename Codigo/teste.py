import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np


df = pd.read_csv('trials.csv', delimiter=';')
G = nx.Graph()


for _, row in df.iterrows():
    local = row['city'] or row['gadm.adm2'] or row['gadm.adm1'] or row['gadm.adm0']
    decada = row['decade']
    julgamentos = int(row['tried']) if pd.notna(row['tried']) else 0
    mortes = int(row['deaths']) if pd.notna(row['deaths']) else 0

    if pd.notna(local) and pd.notna(decada):
        G.add_node(local, tipo='local')
        G.add_node(decada, tipo='decada')

        if G.has_edge(local, decada):
            G[local][decada]['julgamentos'] += julgamentos
            G[local][decada]['mortes'] += mortes
        else:
            G.add_edge(local, decada, julgamentos=julgamentos, mortes=mortes)


def analise_regioes(G):
    regioes_julgamentos = defaultdict(int)
    regioes_mortes = defaultdict(int)

    for u, v, data in G.edges(data=True):
        if G.nodes[u]['tipo'] == 'local':
            local = u
        elif G.nodes[v]['tipo'] == 'local':
            local = v
        else:
            continue

        regioes_julgamentos[local] += data.get('julgamentos', 0)
        regioes_mortes[local] += data.get('mortes', 0)

    top_julgamentos = sorted(regioes_julgamentos.items(), key=lambda x: x[1], reverse=True)[:10]
    top_mortes = sorted(regioes_mortes.items(), key=lambda x: x[1], reverse=True)[:10]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    labels_julg = [str(x[0]) if pd.notna(x[0]) else 'Desconhecido' for x in top_julgamentos]
    labels_mortes = [str(x[0]) if pd.notna(x[0]) else 'Desconhecido' for x in top_mortes]

    ax1.barh(labels_julg, [x[1] for x in top_julgamentos], color='skyblue')
    ax1.set_title('Top 10 Regiões por Número de Julgamentos')
    ax1.set_xlabel('Total de Julgamentos')

    ax2.barh(labels_mortes, [x[1] for x in top_mortes], color='salmon')
    ax2.set_title('Top 10 Regiões por Número de Mortes')
    ax2.set_xlabel('Total de Mortes')

    plt.tight_layout()
    plt.show()

    return top_julgamentos, top_mortes


def correlacao_geografica(G, df):
    geo_graph = nx.Graph()

    locais = [n for n in G.nodes() if G.nodes[n]['tipo'] == 'local']
    geo_graph.add_nodes_from(locais)

    for i, loc1 in enumerate(locais):
        for j, loc2 in enumerate(locais):
            if i < j:
                loc1_parts = loc1.split(',')
                loc2_parts = loc2.split(',')

                if len(loc1_parts) > 1 and len(loc2_parts) > 1 and loc1_parts[-1].strip() == loc2_parts[-1].strip():
                    geo_graph.add_edge(loc1, loc2, weight=1)

    correlacoes = []
    for u, v in geo_graph.edges():
        julgamentos_u = sum(data['julgamentos'] for _, _, data in G.edges(u, data=True) if 'julgamentos' in data)
        julgamentos_v = sum(data['julgamentos'] for _, _, data in G.edges(v, data=True) if 'julgamentos' in data)
        correlacoes.append((julgamentos_u, julgamentos_v))

    if correlacoes:
        x, y = zip(*correlacoes)
        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, alpha=0.5)
        plt.xlabel('Julgamentos na Região A')
        plt.ylabel('Julgamentos na Região B Próxima')
        plt.title('Correlação entre Proximidade Geográfica e Número de Julgamentos')

        corr_coef = np.corrcoef(x, y)[0, 1]
        plt.annotate(f'Coeficiente de Correlação: {corr_coef:.2f}',
                     xy=(0.5, 0.95), xycoords='axes fraction',
                     ha='center', va='center', bbox=dict(boxstyle="round", fc="w"))

        plt.show()

    return geo_graph, correlacoes


def padroes_temporais(G):
    decadas = sorted([n for n in G.nodes() if G.nodes[n]['tipo'] == 'decada'])
    julgamentos_por_decada = []
    mortes_por_decada = []

    for decada in decadas:
        total_julg = sum(data['julgamentos'] for _, _, data in G.edges(decada, data=True) if 'julgamentos' in data)
        total_mortes = sum(data['mortes'] for _, _, data in G.edges(decada, data=True) if 'mortes' in data)
        julgamentos_por_decada.append(total_julg)
        mortes_por_decada.append(total_mortes)

    plt.figure(figsize=(12, 6))

    plt.plot(decadas, julgamentos_por_decada, marker='o', label='Julgamentos')
    plt.plot(decadas, mortes_por_decada, marker='o', label='Mortes')

    plt.xlabel('Década')
    plt.ylabel('Contagem')
    plt.title('Padrão Temporal de Julgamentos e Mortes por Bruxaria')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    for i in range(1, len(decadas) - 1):
        if julgamentos_por_decada[i] > julgamentos_por_decada[i - 1] and julgamentos_por_decada[i] > julgamentos_por_decada[i + 1]:
            plt.annotate('Pico', xy=(decadas[i], julgamentos_por_decada[i]),
                         xytext=(0, 10), textcoords='offset points',
                         ha='center', va='bottom', color='red')

    plt.tight_layout()
    plt.show()

    return decadas, julgamentos_por_decada, mortes_por_decada


def regioes_letais_conectadas(G, top_mortes):
    regioes_letais = [x[0] for x in top_mortes]
    subgraph = G.subgraph(regioes_letais).copy()

    for i, loc1 in enumerate(regioes_letais):
        for j, loc2 in enumerate(regioes_letais):
            if i < j and any(x in loc1 for x in ['Switzerland', 'Germany', 'Bayern']) and any(x in loc2 for x in ['Switzerland', 'Germany', 'Bayern']):
                subgraph.add_edge(loc1, loc2, tipo='proximidade')

    fig, ax = plt.subplots(figsize=(12, 8))
    pos = nx.circular_layout(subgraph)

    node_colors = []
    node_sizes = []
    mortes_dict = dict(top_mortes)
    for node in subgraph.nodes():
        node_colors.append(mortes_dict[node])
        node_sizes.append(mortes_dict[node] * 10)

    norm = plt.Normalize(min(node_colors), max(node_colors))
    cmap = plt.colormaps['Reds']

    nodes = nx.draw_networkx_nodes(subgraph, pos, node_size=node_sizes,
                                   node_color=node_colors, cmap=cmap, alpha=0.8, ax=ax)
    nx.draw_networkx_labels(subgraph, pos, font_size=10, font_weight='bold', ax=ax)

    nx.draw_networkx_edges(subgraph, pos, edgelist=[(u, v) for u, v, d in subgraph.edges(data=True) if d.get('tipo') == 'proximidade'],
                           edge_color='gray', style='dashed', alpha=0.5, ax=ax)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Número de Mortes')

    plt.title('Conexões entre as Regiões Mais Letais\n(Tamanho e cor representam número de mortes)')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    return subgraph


# Análises
print("1. Análise das regiões com mais julgamentos e mortes:")
top_julg, top_mortes = analise_regioes(G)

print("\n2. Correlação entre proximidade geográfica e número de julgamentos:")
geo_graph, correlacoes = correlacao_geografica(G, df)

print("\n3. Padrões temporais de perseguição:")
decadas, julgamentos, mortes = padroes_temporais(G)

print("\n4. Conexões entre regiões mais letais:")
subgraph_letais = regioes_letais_conectadas(G, top_mortes[:5])
