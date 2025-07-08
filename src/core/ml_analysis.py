import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, silhouette_score
import matplotlib.pyplot as plt
import networkx as nx
import os
import time

def analise_classificacao(df):
    """
    Realiza análise de classificação para prever mortes baseado em características dos julgamentos.
    """
    print("\n[🔍] Executando análise de classificação...")

  
    df = df.copy()
    df['morreu'] = (df['deaths'] > 0).astype(int)
    df = df[df['tried'] > 0].dropna(subset=['decade', 'gadm_adm0', 'gadm_adm1', 'city'])

 
    categorias = df[['decade', 'gadm_adm0', 'gadm_adm1', 'city']].fillna("Desconhecido")
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_encoded = encoder.fit_transform(categorias)

 
    X = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out())
    X['tried'] = df['tried'].values
    y = df['morreu']


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    models = {
        "Regressão Logística": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(class_weight='balanced', n_estimators=100)
    }

    results = {}
    for name, model in models.items():
        print(f"  - Treinando modelo: {name}...")
        start_time = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        results[name] = {
            'report': report,
            'features_importance': model.feature_importances_ if hasattr(model, 'feature_importances_') else None,
            'time_elapsed': time.time() - start_time
        }

        print(f"    ✓ Concluído em {results[name]['time_elapsed']:.2f} segundos")
        print(f"📊 {name} - Relatório de Classificação:")
        print(classification_report(y_test, y_pred))

    return results

def analise_clusterizacao(df):
    """
    Realiza análise de clusterização de localidades baseada em atividade e taxa de mortalidade.
    """
    print("\n[🔍] Executando análise de clusterização de localidades...")

    print("  - Agregando dados por cidade...")
    agrupado = df.groupby('city').agg({
        'tried': 'sum',
        'deaths': 'sum'
    }).query('tried > 0').copy()

    agrupado['taxa_morte'] = agrupado['deaths'] / agrupado['tried']

  
    print("  - Normalizando dados...")
    scaler = StandardScaler()
    X_cluster = scaler.fit_transform(agrupado[['tried', 'taxa_morte']])

 
    print("  - Determinando número ótimo de clusters...")
    silhouette_scores = []
    for k in range(2, 8):
        print(f"    - Testando k={k} clusters...", end=' ')
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_cluster)
        score = silhouette_score(X_cluster, labels)
        silhouette_scores.append(score)
        print(f"Silhouette: {score:.3f}")

    optimal_k = np.argmax(silhouette_scores) + 2  # +2 porque começamos de k=2
    print(f"  - Número ótimo de clusters determinado: {optimal_k}")


    print(f"  - Aplicando K-Means com k={optimal_k}...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    agrupado['cluster'] = kmeans.fit_predict(X_cluster)
    

    cluster_labels = {
        0: "Baixa atividade - Baixa mortalidade",
        1: "Média atividade - Baixa mortalidade",
        2: "Alta atividade - Média mortalidade",
        3: "Média atividade - Alta mortalidade",
        4: "Alta atividade - Alta mortalidade"
    }
    agrupado['cluster_label'] = agrupado['cluster'].map(cluster_labels)


    print("  - Gerando visualização...")
    plt.figure(figsize=(14, 8))
    scatter = plt.scatter(
        agrupado['tried'], 
        agrupado['taxa_morte'], 
        c=agrupado['cluster'], 
        cmap='viridis',
        alpha=0.7,
        edgecolors='w',
        s=100
    )
    
    plt.legend(
        handles=scatter.legend_elements()[0],
        labels=[cluster_labels.get(i, f"Cluster {i}") for i in range(optimal_k)],
        title="Perfil dos Clusters",
        bbox_to_anchor=(1.05, 1),
        loc='upper left'
    )
    
    plt.xlabel("Total de Julgamentos (log)", fontsize=12)
    plt.ylabel("Taxa de Mortalidade", fontsize=12)
    plt.title("Clusterização de Cidades por Perfil de Julgamento", fontsize=14, pad=20)
    plt.xscale('log')
    plt.grid(True, linestyle='--', alpha=0.3)


    top_cities = agrupado.nlargest(5, 'tried')
    for idx, row in top_cities.iterrows():
        plt.annotate(
            idx, 
            (row['tried'], row['taxa_morte']), 
            textcoords="offset points",
            xytext=(0,5), 
            ha='center',
            fontsize=9
        )


    os.makedirs("outputs", exist_ok=True)
    caminho_img = "outputs/clusterizacao_localidades.png"
    plt.savefig(caminho_img, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[✔] Imagem salva: {caminho_img}")
    
    return agrupado

def visualizar_subgrafos_temporais(G, df, top_decades=5):
    """
    Visualiza subgrafos temporais para as décadas com mais atividade.
    
    Args:
        G: Grafo networkx completo
        df: DataFrame com os dados originais
        top_decades: Número de décadas principais para visualizar
    """
    print("\n[🕰️] Visualizando subgrafos temporais...")
    
    decada_counts = df['decade'].value_counts()
    top_decadas = decada_counts.head(top_decades).index.tolist()
    
    os.makedirs("outputs/subgrafos_temporais", exist_ok=True)
    
    print(f"  - Processando {len(top_decadas)} décadas principais...")
    start_time = time.time()
    
    for i, decada in enumerate(top_decadas, 1):
        print(f"    [{i}/{len(top_decadas)}] Processando década {decada}...", end=' ')
        
        nos_decada = [n for n in G.nodes if n == decada or 
                     (G.nodes[n].get('tipo') == 'local' and 
                      any(data.get('decade') == decada for _, _, data in G.edges(n, data=True)))]
        
        if not nos_decada:
            print("Nenhum nó encontrado.")
            continue
            
        subgrafo = G.subgraph(nos_decada)
        
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(subgrafo, seed=42, k=0.3)
        

        node_colors = []
        node_sizes = []
        for node in subgrafo.nodes():
            if node == decada:
                node_colors.append('red')
                node_sizes.append(800)
            elif subgrafo.nodes[node].get('tipo') == 'local':
                node_colors.append('skyblue')
                node_sizes.append(300 + 50 * len(list(subgrafo.neighbors(node))))
            else:
                node_colors.append('lightgreen')
                node_sizes.append(500)
        
        
        nx.draw_networkx_nodes(subgrafo, pos, node_color=node_colors, node_size=node_sizes)
        nx.draw_networkx_edges(subgrafo, pos, alpha=0.3, edge_color='gray')
        nx.draw_networkx_labels(subgrafo, pos, font_size=8)

        plt.title(f"Rede de julgamentos na década de {decada}\n"
                 f"Total: {decada_counts[decada]} julgamentos", fontsize=14)
        plt.axis('off')
        

        caminho_img = f"outputs/subgrafos_temporais/subgrafo_{decada}.png"
        plt.savefig(caminho_img, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Concluído → {caminho_img}")
    
    total_time = time.time() - start_time
    print(f"[✔] Processo concluído em {total_time:.2f} segundos")
    print(f"[✔] Subgrafos temporais salvos em outputs/subgrafos_temporais/")
