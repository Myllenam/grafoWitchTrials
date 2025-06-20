import csv
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict


with open('trials.csv', mode='r', encoding='utf-8') as arquivo:
    leitor = csv.DictReader(arquivo, delimiter=';')
    dados = [linha for linha in leitor]


grafo = nx.Graph()


locais_por_decada = defaultdict(set)
fontes_por_local = defaultdict(set)


for linha in dados:
    cidade = linha.get('city', '').strip()
    regiao = linha.get('gadm.adm2', '').strip()
    estado = linha.get('gadm.adm1', '').strip()
    pais = linha.get('gadm.adm0', '').strip()
    decada = linha.get('decade', '').strip()
    fonte = linha.get('record.source', '').strip()
    julgamentos = linha.get('tried', '0').strip()
    mortes = linha.get('deaths', '0').strip()
    
    
    local = cidade or regiao or estado or pais
    
    if not local or not decada:
        continue

    grafo.add_node(local, tipo='local', nivel=('cidade' if cidade else 'regiao' if regiao else 'estado' if estado else 'pais'))
    grafo.add_node(decada, tipo='decada')
    
    
    if grafo.has_edge(local, decada):
       
        grafo[local][decada]['julgamentos'] += int(julgamentos) if julgamentos.isdigit() else 0
        grafo[local][decada]['mortes'] += int(mortes) if mortes.isdigit() else 0
    else:
        grafo.add_edge(local, decada, 
                      julgamentos=int(julgamentos) if julgamentos.isdigit() else 0,
                      mortes=int(mortes) if mortes.isdigit() else 0)
    
   
    locais_por_decada[decada].add(local)
    
   
    if fonte:
        grafo.add_node(fonte, tipo='fonte')
        grafo.add_edge(fonte, local, tipo='documenta')
        fontes_por_local[fonte].add(local)


for decada, locais in locais_por_decada.items():
    locais = list(locais)
    for i in range(len(locais)):
        for j in range(i+1, len(locais)):
            if not grafo.has_edge(locais[i], locais[j]):
                grafo.add_edge(locais[i], locais[j], tipo='mesma_decada', decada=decada)


plt.figure(figsize=(20, 15))


pos = {}
node_types = {'local': (0, 0), 'decada': (1, 0), 'fonte': (0.5, 1)}


for i, node in enumerate(grafo.nodes()):
    node_type = grafo.nodes[node]['tipo']
    offset = (i % 20) * 0.05
    
    if node_type == 'local':
        pos[node] = (node_types['local'][0] + offset, 
                    node_types['local'][1] - i*0.005)
    elif node_type == 'decada':
        pos[node] = (node_types['decada'][0] + offset, 
                    node_types['decada'][1] - i*0.005)
    else:  # fonte
        pos[node] = (node_types['fonte'][0] + offset, 
                    node_types['fonte'][1] + i*0.005)


node_colors = []
node_sizes = []
for node in grafo.nodes():
    if grafo.nodes[node]['tipo'] == 'local':
        node_colors.append('skyblue')
        node_sizes.append(300)
    elif grafo.nodes[node]['tipo'] == 'decada':
        node_colors.append('lightgreen')
        node_sizes.append(500)
    else:
        node_colors.append('salmon')
        node_sizes.append(200)


nx.draw_networkx_nodes(grafo, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8)


edge_colors = []
edge_widths = []
for u, v, data in grafo.edges(data=True):
    if grafo.nodes[u]['tipo'] == 'fonte' or grafo.nodes[v]['tipo'] == 'fonte':
        edge_colors.append('gray')
        edge_widths.append(1)
    elif 'mesma_decada' in data.get('tipo', ''):
        edge_colors.append('orange')
        edge_widths.append(0.5)
    else:  
        edge_colors.append('purple')
        edge_widths.append(0.3 + data.get('julgamentos', 0)/50)

nx.draw_networkx_edges(grafo, pos, edge_color=edge_colors, width=edge_widths, alpha=0.5)

# Labels 
labels = {}
for node in grafo.nodes():
    if grafo.nodes[node]['tipo'] != 'local' or grafo.degree(node) > 2:
        labels[node] = node
        
nx.draw_networkx_labels(grafo, pos, labels, font_size=8, font_weight='bold')

# Legenda
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', label='Local (cidade/região/país)',
              markerfacecolor='skyblue', markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='Década',
              markerfacecolor='lightgreen', markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='Fonte histórica',
              markerfacecolor='salmon', markersize=10),
    plt.Line2D([0], [0], color='purple', lw=2, label='Local-Década (julgamentos)'),
    plt.Line2D([0], [0], color='orange', lw=2, label='Locais na mesma década'),
    plt.Line2D([0], [0], color='gray', lw=2, label='Fonte documental')
]
plt.legend(handles=legend_elements, loc='upper right')

plt.title("Rede Completa de Julgamentos de Bruxaria", fontsize=16)
plt.axis('off')
plt.tight_layout()
plt.show()


print("### Análise da Rede Completa ###")
print(f"Total de nós: {len(grafo.nodes())}")
print(f"Total de arestas: {len(grafo.edges())}")


print("\nLocais mais conectados:")
locais = [n for n in grafo.nodes() if grafo.nodes[n]['tipo'] == 'local']
graus_locais = sorted([(n, grafo.degree(n)) for n in locais], key=lambda x: x[1], reverse=True)[:10]
for local, grau in graus_locais:
    print(f"{local}: {grau} conexões")


print("\nDécadas com mais julgamentos:")
decadas = [n for n in grafo.nodes() if grafo.nodes[n]['tipo'] == 'decada']
julgamentos_por_decada = []
for d in decadas:
    total = sum(data['julgamentos'] for _, _, data in grafo.edges(d, data=True) if 'julgamentos' in data)
    julgamentos_por_decada.append((d, total))
    
for decada, total in sorted(julgamentos_por_decada, key=lambda x: x[1], reverse=True)[:5]:
    print(f"{decada}: {total} julgamentos")