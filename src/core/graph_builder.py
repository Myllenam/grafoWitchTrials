import networkx as nx
import pandas as pd

def construir_grafo(df: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()

    for _, row in df.iterrows():
        if pd.isna(row['gadm_adm0']) or row['gadm_adm0'] == '':
            continue

        origem = row['localizacao']
        decada = row['decade']
        julgamentos = row['tried']
        mortes = row['deaths']

        # nó da localização
        G.add_node(origem, tipo='local',
                   lat=row['lat'] if pd.notna(row['lat']) else None,
                   lon=row['lon'] if pd.notna(row['lon']) else None,
                   pais=row['gadm_adm0'])

        # nó da década
        G.add_node(decada, tipo='decada')

        # aresta origem → década
        if G.has_edge(origem, decada):
            G.edges[(origem, decada)]['weight'] += julgamentos
            G.edges[(origem, decada)]['mortes'] += mortes
        else:
            G.add_edge(origem, decada,
                       weight=julgamentos,
                       mortes=mortes)

    print(f"\nGrafo construído com:")
    print(f"- {len([n for n in G.nodes() if G.nodes.get(n, {}).get('tipo') == 'local'])} locais")
    print(f"- {len([n for n in G.nodes() if G.nodes.get(n, {}).get('tipo') == 'decada'])} décadas")
    print(f"- {len(G.edges())} conexões")
    return G
