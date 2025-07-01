import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
import matplotlib.pyplot as plt

def analisar_grafo(G: nx.DiGraph):
    print("\n=== ANÁLISE ESTRUTURAL DA REDE ===")
    print(f"Número de nós: {G.number_of_nodes()}")
    print(f"Número de arestas: {G.number_of_edges()}")

    graus = dict(G.degree())
    grau_medio = sum(graus.values()) / len(graus) if graus else 0
    print(f"Grau médio: {grau_medio:.2f}")

    componentes = list(nx.weakly_connected_components(G))
    print(f"Componentes fracos conectados: {len(componentes)}")

    if componentes:
        maior_componente = max(componentes, key=len)
        print(f"Tamanho do maior componente: {len(maior_componente)}")

    centralidade = nx.degree_centrality(G)
    mais_central = max(centralidade, key=centralidade.get) if centralidade else None
    if mais_central:
        print(f"Nó mais central (grau): {mais_central} - Centralidade: {centralidade[mais_central]:.4f}")

def mostrar_nos_com_maior_grau(G: nx.DiGraph, top_n=5):
    graus = dict(G.degree())
    maiores = sorted(graus.items(), key=lambda x: x[1], reverse=True)[:top_n]
    print(f"\nTop {top_n} nós com maior grau:")
    for no, grau in maiores:
        print(f"{no}: grau {grau}")

def mostrar_nos_sem_arestas(G: nx.DiGraph):
    isolados = [n for n in G.nodes if G.degree(n) == 0]
    if isolados:
        print(f"\nNós sem arestas ({len(isolados)}):")
        for no in isolados:
            print(f"- {no}")
    else:
        print("\nNão há nós isolados.")

def analisar_centralidades(G: nx.DiGraph):
    print("\nTop 5 nós por Centralidade de Grau:")
    grau = nx.degree_centrality(G)
    for no, val in sorted(grau.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"{no}: {val:.4f}")

    print("\nTop 5 nós por Centralidade de Proximidade:")
    try:
        prox = nx.closeness_centrality(G)
        for no, val in sorted(prox.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"{no}: {val:.4f}")
    except Exception:
        print("Erro ao calcular centralidade de proximidade")

    print("\nTop 5 nós por Centralidade de Intermediação:")
    try:
        btw = nx.betweenness_centrality(G)
        for no, val in sorted(btw.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"{no}: {val:.4f}")
    except Exception:
        print("Erro ao calcular centralidade de intermediação")

def densidade_rede(G: nx.DiGraph):
    densidade = nx.density(G)
    print(f"\nDensidade da rede: {densidade:.4f}")


def detectar_comunidades(G: nx.DiGraph):
    print("\n=== Detecção de Comunidades ===")

    # Subgrafo apenas com nós de tipo 'local'
    locais = [n for n, d in G.nodes(data=True) if d.get('tipo') == 'local']
    subgrafo = G.subgraph(locais).to_undirected()

    if subgrafo.number_of_nodes() == 0:
        print("Nenhum nó 'local' disponível para análise de comunidades.")
        return

    comunidades = list(greedy_modularity_communities(subgrafo))
    print(f"Total de comunidades detectadas: {len(comunidades)}")

    # Mapeia cada nó à comunidade que pertence
    node_colors = {}
    for i, comunidade in enumerate(comunidades):
        for no in comunidade:
            node_colors[no] = i

    # Visualização
    pos = nx.spring_layout(subgrafo, seed=42)
    plt.figure(figsize=(14, 10))
    nx.draw(
        subgrafo,
        pos,
        node_color=[node_colors.get(n, 0) for n in subgrafo.nodes()],
        with_labels=True,
        node_size=500,
        font_size=8,
        cmap=plt.cm.get_cmap("tab20"),
        edge_color="lightgray"
    )
    plt.title("Comunidades Regionais Detectadas")
    plt.axis("off")
    plt.tight_layout()
    plt.show()