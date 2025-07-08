# main.py
from core.data_loader import carregar_dados
from core.preprocessing import preprocessar
from core.graph_builder import construir_grafo
from core.analysis import (total_julgamentos, total_mortes,
                           total_registros_desconhecidos, mostrar_taxa_mortalidade,
                           executar_analise_completa)
from core.graph_analysis import (analisar_grafo, mostrar_nos_com_maior_grau,
                                  mostrar_nos_sem_arestas, densidade_rede,
                                  analisar_centralidades, detectar_comunidades)
from core.visualization import visualizar_subgrafo
import networkx as nx
from core.ml_analysis import analise_classificacao, analise_clusterizacao, visualizar_subgrafos_temporais

def main():
    print("Configurando análise de julgamentos históricos...")
    caminho_csv = 'trials.csv'

    try:
        df = carregar_dados(caminho_csv)
        print("Dados carregados com sucesso!")
        print(f"Total de registros: {len(df)}")

        df = preprocessar(df)

        print(f"Total de julgamentos: {total_julgamentos(df)}")
        print(f"Total de mortes: {total_mortes(df)}")
        print(f"Registros com dados faltantes: {total_registros_desconhecidos(df)}")

        G = construir_grafo(df)

        print(f"\nGrafo construído com:")
        print(f"- {len([n for n, d in G.nodes(data=True) if d.get('tipo') == 'local'])} locais")
        print(f"- {len([n for n, d in G.nodes(data=True) if d.get('tipo') == 'decada'])} décadas")
        print(f"- {len(G.edges())} conexões")

      # Visualização 1: Subgrafo da década com mais julgamentos
        decada_top = df['decade'].value_counts().idxmax()
        nos_decada = [u for u, v in G.edges() if v == decada_top] + [decada_top]
        visualizar_subgrafo(G, nos_decada, f"Subgrafo da década com mais julgamentos ({decada_top})")

        # Visualização 2: Subgrafo da década com mais mortes
        decada_top_mortes = df.groupby('decade')['deaths'].sum().idxmax()
        nos_decada_mortes = [u for u, v in G.edges() if v == decada_top_mortes] + [decada_top_mortes]
        visualizar_subgrafo(G, nos_decada_mortes, f"Subgrafo da década com mais mortes ({decada_top_mortes})")
        
         # Visualização 3: Subgrafo da localidade com maior taxa de mortalidade
        df_valid = df[df['tried'] > 0].copy()
        df_valid['taxa_mortalidade'] = df_valid['deaths'] / df_valid['tried']
        local_top_taxa = df_valid.sort_values('taxa_mortalidade', ascending=False).iloc[0]['localizacao']
        nos_local_taxa = [local_top_taxa]
        if local_top_taxa in G:
            nos_local_taxa += list(G.neighbors(local_top_taxa))
            visualizar_subgrafo(G, nos_local_taxa, f"Localidade com maior taxa de mortalidade ({local_top_taxa})")
        else:
            print(f"[AVISO] Localização '{local_top_taxa}' não encontrada no grafo.")


       
        resultados = executar_analise_completa(df, G)
        analisar_grafo(G)
        mostrar_nos_com_maior_grau(G)
        mostrar_nos_sem_arestas(G)
        mostrar_taxa_mortalidade(df)
        densidade_rede(G)
        analisar_centralidades(G)
        detectar_comunidades(G)

        # Exibição dos resultados
        print("\n--- Resumo Detalhado dos Resultados ---")

        if not resultados['top_paises'].empty:
            print("\n**1. Regiões (Países) com Mais Julgamentos e Mortes:**")
            print(resultados['top_paises'].head())

        if not resultados['top_localizacoes'].empty:
            print("\n**1b. Regiões (Localizações Detalhadas) com Mais Julgamentos e Mortes:**")
            print(resultados['top_localizacoes'].head())

        if not resultados['analise_temporal'].empty:
            print("\n**2. Padrões Temporais (Julgamentos e Taxa de Mortalidade por Década):**")
            print("Observe os gráficos gerados para identificar picos e vales, que podem indicar ondas de perseguição.")
            print("As décadas com mais julgamentos e as maiores taxas de mortalidade são:")
            print(resultados['analise_temporal'].sort_values('Julgamentos', ascending=False).head(5))
            print(resultados['analise_temporal'].sort_values('Taxa Mortalidade', ascending=False).head(5))

        if not resultados['distribuicao_geografica'].empty:
            print("\n**3. Correlação entre Proximidade Geográfica e Julgamentos:**")
            print("O gráfico de dispersão com o tamanho dos pontos proporcionais aos julgamentos\npermite uma inspeção visual de clusters geográficos.")
            print("Top 5 Localizações por Julgamentos (Geográfico):")
            print(resultados['distribuicao_geografica'].sort_values('Julgamentos', ascending=False).head(5))

        if resultados['taxas_mortalidade_geral'] is not None:
            print("\n**4. Taxas de mortalidade baseada na quantidade de julgamentos:**")
            mostrar_taxa_mortalidade(df)
        analise_classificacao(df)
        analise_clusterizacao(df)
        visualizar_subgrafos_temporais(G, df)
        
    except FileNotFoundError:
        print("\nERRO: O arquivo 'trials.csv' não foi encontrado. Por favor, verifique o caminho do arquivo.")
    except Exception as e:
        print(f"\nERRO inesperado durante a execução principal: {str(e)}")
    finally:
        print("\nProcesso de análise concluído.")


if __name__ == "__main__":
    main()
