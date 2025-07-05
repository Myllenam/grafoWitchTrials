import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import networkx as nx
import os

def analise_classificacao(df):
    print("\n[🔍] Executando análise de classificação...")

    df = df.copy()
    df['morreu'] = (df['deaths'] > 0).astype(int)
    df = df[df['tried'] > 0]

    categorias = df[['decade', 'gadm_adm0', 'gadm_adm1', 'city']].fillna("Desconhecido")
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_encoded = encoder.fit_transform(categorias)

    X_encoded_df = pd.DataFrame(X_encoded)
    X_encoded_df.columns = X_encoded_df.columns.astype(str)  # <- converte para string

    X = pd.concat([X_encoded_df, df[['tried']].reset_index(drop=True)], axis=1)

    y = df['morreu']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)

    logreg = LogisticRegression(max_iter=500)
    rf = RandomForestClassifier()

    logreg.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    logreg_report = classification_report(y_test, logreg.predict(X_test))
    rf_report = classification_report(y_test, rf.predict(X_test))

    print("\n📊 Relatório - Regressão Logística:")
    print(logreg_report)

    print("\n📊 Relatório - Random Forest:")
    print(rf_report)

def analise_clusterizacao(df):
    print("\n[🔍] Executando análise de clusterização de localidades...")

    agrupado = df.groupby('city').agg({
        'tried': 'sum',
        'deaths': 'sum'
    }).query('tried > 0').copy()

    agrupado['taxa_morte'] = agrupado['deaths'] / agrupado['tried']

    scaler = StandardScaler()
    X_cluster = scaler.fit_transform(agrupado)

    kmeans = KMeans(n_clusters=4, random_state=42)
    agrupado['cluster'] = kmeans.fit_predict(X_cluster)

    # Criar pasta se não existir
    os.makedirs("outputs", exist_ok=True)

    # Gerar gráfico
    plt.figure(figsize=(10, 6))
    plt.scatter(agrupado['tried'], agrupado['taxa_morte'], c=agrupado['cluster'], cmap='viridis')
    plt.xlabel("Total de Julgamentos")
    plt.ylabel("Taxa de Mortalidade")
    plt.title("Clusterização de Cidades por Perfil de Julgamento")
    plt.grid(True)
    plt.tight_layout()

    caminho_img = "outputs/clusterizacao_localidades.png"
    plt.savefig(caminho_img)
    print(f"[✔] Imagem salva: {caminho_img}")

def visualizar_subgrafos_temporais(G, df):
    print("\n[🔍] Gerando visualizações temporais por década...")

    decadas = df['decade'].dropna().unique()
    decadas.sort()

    os.makedirs("outputs/temporal", exist_ok=True)

    for decada in decadas:
        nos_relacionados = [u for u, v in G.edges() if v == decada or u == decada] + [decada]
        subgrafo = G.subgraph(nos_relacionados)

        plt.figure(figsize=(10, 6))
        pos = nx.spring_layout(subgrafo, seed=42)
        nx.draw(subgrafo, pos, with_labels=True, node_color='lightcoral', edge_color='gray', node_size=300, font_size=8)
        plt.title(f"Subgrafo - Julgamentos na Década de {decada}")
        plt.tight_layout()

        caminho = f"outputs/temporal/grafo_decada_{decada}.png"
        plt.savefig(caminho)
        plt.close()
        print(f"[✔] Subgrafo salvo: {caminho}")
