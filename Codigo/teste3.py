import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import pdist, squareform

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 8)
pd.set_option('display.max_columns', 50)

class WitchTrialsAnalysis:
    def __init__(self, arquivo_dados):
        """
        Inicializa a análise carregando os dados do arquivo CSV.
        Realiza pré-processamento e constrói o grafo.
        """
        try:
            self.df = pd.read_csv(arquivo_dados, delimiter=';')
            print("Dados carregados com sucesso!")
            print(f"Total de registros: {len(self.df)}")
            
            self._preprocessar_dados()
            self._construir_grafo()
            
        except Exception as e:
            print(f"\nERRO durante inicialização: {str(e)}")
            self.G = nx.DiGraph() # Garante que G seja inicializado mesmo com erro

    def _preprocessar_dados(self):
        """
        Prepara os dados para análise:
        - Filtra registros com país desconhecido.
        - Preenche valores faltantes para 'city' e 'decade'.
        - Converte 'tried' e 'deaths' para numérico, preenchendo NaNs com 0.
        - Cria um identificador de localização consolidado (cidade, estado, país).
        """
        # Filtra registros onde o país é desconhecido
        self.df = self.df.loc[(self.df['gadm.adm0'].notna()) & (self.df['gadm.adm0'] != '')]
        
        # Preenche valores faltantes
        self.df['city'] = self.df['city'].fillna('Desconhecido')
        self.df['decade'] = self.df['decade'].fillna('ND')
        
        # Converte números, tratando erros e preenchendo NaNs
        self.df['tried'] = pd.to_numeric(self.df['tried'], errors='coerce').fillna(0).astype(int)
        self.df['deaths'] = pd.to_numeric(self.df['deaths'], errors='coerce').fillna(0).astype(int)
        
        # Cria identificador de localização que inclui cidade, estado e país
        self.df['localizacao'] = self.df.apply(
            lambda x: f"{x['city']}, {x['gadm.adm1']}" if pd.notna(x['gadm.adm1']) and x['gadm.adm1'] != '' else x['city'], axis=1)
        self.df['localizacao'] = self.df['localizacao'] + ", " + self.df['gadm.adm0']

    def _construir_grafo(self):
        """
        Constrói o grafo NetworkX a partir dos dados pré-processados.
        - Nós de tipo 'local' representam localizações geográficas (cidade, estado, país)
          com atributos de latitude, longitude e país.
        - Nós de tipo 'decada' representam as décadas.
        - Arestas conectam 'local' a 'decada', com pesos ('weight' para julgamentos
          e 'mortes' para o número de mortos).
        """
        self.G = nx.DiGraph()
        
        # Adiciona nós e arestas com base nos dados do DataFrame
        for _, row in self.df.iterrows():
            if pd.isna(row['gadm.adm0']) or row['gadm.adm0'] == '':
                continue # Pula registros sem país definido
                
            origem = row['localizacao']
            decada = row['decade']
            
            julgamentos = row['tried']
            mortes = row['deaths']
            
            # Adiciona nós de localização com seus atributos geográficos e país
            self.G.add_node(origem, tipo='local', 
                            lat=row['lat'] if pd.notna(row['lat']) else None,
                            lon=row['lon'] if pd.notna(row['lon']) else None,
                            pais=row['gadm.adm0'])
            
            # Adiciona nós de década
            self.G.add_node(decada, tipo='decada')
            
            # Adiciona arestas entre localização e década, acumulando julgamentos e mortes
            if self.G.has_edge(origem, decada):
                self.G.edges[(origem, decada)]['weight'] += julgamentos
                self.G.edges[(origem, decada)]['mortes'] += mortes
            else:
                self.G.add_edge(origem, decada, 
                                weight=julgamentos,
                                mortes=mortes)
                
        print(f"\nGrafo construído com:")
        print(f"- {len([n for n in self.G.nodes() if self.G.nodes.get(n, {}).get('tipo') == 'local'])} locais")
        print(f"- {len([n for n in self.G.nodes() if self.G.nodes.get(n, {}).get('tipo') == 'decada'])} décadas")
        print(f"- {len(self.G.edges())} conexões")

    def _visualizar_grafo(self, title="Visualização da Rede de Julgamentos"):
        """
        Visualiza o grafo NetworkX.
        Nós de localização são azuis, nós de década são verdes.
        A espessura das arestas pode representar o peso (julgamentos).
        """
        if not self.G.nodes:
            print(f"Não é possível visualizar o grafo '{title}': o grafo está vazio.")
            return

        pos = nx.spring_layout(self.G, seed=42) # Usar um seed para reprodutibilidade
        
        node_colors = []
        for node in self.G.nodes():
            if self.G.nodes[node]['tipo'] == 'local':
                node_colors.append('skyblue') # Azul mais claro para locais
            elif self.G.nodes[node]['tipo'] == 'decada':
                node_colors.append('lightgreen') # Verde claro para décadas
            else:
                node_colors.append('gray') # Cor padrão para outros tipos, se houver

        # Escala a espessura da aresta com base no 'weight' (julgamentos)
        # Normaliza os pesos para ter um range de largura razoável
        edge_weights = np.array([data.get('weight', 1) for u, v, data in self.G.edges(data=True)])
        if len(edge_weights) > 0 and edge_weights.max() > 0:
            edge_widths = 0.5 + 5 * (edge_weights / edge_weights.max()) # Ajuste o fator 5 para mudar a intensidade
        else:
            edge_widths = 0.5 # Largura padrão se não houver pesos ou max for 0

        plt.figure(figsize=(18, 12)) # Aumenta o tamanho da figura para melhor visibilidade
        nx.draw_networkx(self.G, pos, 
                         node_color=node_colors,
                         edge_color='gray', # Cor das arestas
                         width=edge_widths, 
                         alpha=0.7, 
                         with_labels=True, 
                         font_size=7, # Reduz o tamanho da fonte para evitar sobreposição
                         node_size=1200, # Ajusta o tamanho dos nós
                         arrows=True, # Mostrar setas para grafos direcionados
                         arrowsize=10) # Tamanho das setas

        plt.title(title, size=16) # Título maior
        plt.axis('off') # Desliga os eixos
        plt.tight_layout() # Ajusta o layout para evitar cortes
        plt.show()

    def top_paises(self, n=10):
        """
        Identifica os países com o maior número de julgamentos e mortes.
        Calcula a taxa de mortalidade por país e exibe gráficos.
        """
        if not hasattr(self, 'G') or not self.G.nodes:
            print("Grafo não foi construído corretamente ou está vazio.")
            return pd.DataFrame()
            
        paises = defaultdict(lambda: {'Julgamentos': 0, 'Mortes': 0})
        
        for no in self.G.nodes():
            if self.G.nodes.get(no, {}).get('tipo') == 'local' and 'pais' in self.G.nodes[no]:
                pais = self.G.nodes[no]['pais']
                # Soma os julgamentos e mortes das arestas de saída de cada local para as décadas
                julgamentos = sum(data.get('weight', 0) for _, _, data in self.G.out_edges(no, data=True))
                mortes = sum(data.get('mortes', 0) for _, _, data in self.G.out_edges(no, data=True))
                
                paises[pais]['Julgamentos'] += julgamentos
                paises[pais]['Mortes'] += mortes
        
        # Converte o dicionário de resultados para um DataFrame
        resultados = [{
            'País': pais,
            'Julgamentos': dados['Julgamentos'],
            'Mortes': dados['Mortes'],
            'Taxa Mortalidade': dados['Mortes']/dados['Julgamentos'] if dados['Julgamentos'] > 0 else 0
        } for pais, dados in paises.items() if dados['Julgamentos'] > 0] # Filtra países sem julgamentos
        
        df = pd.DataFrame(resultados).sort_values('Julgamentos', ascending=False).head(n)
        
        if df.empty:
            print("Nenhum país com julgamentos encontrado.")
            return df
            
        # Plot dos resultados
        fig, ax = plt.subplots(1, 2, figsize=(16, 6))
        
        df.plot.barh(x='País', y='Julgamentos', ax=ax[0], 
                     color='darkred', legend=False)
        ax[0].set_title('Top Países por Julgamentos')
        ax[0].set_xlabel('Número de Julgamentos')
        
        df.plot.barh(x='País', y='Taxa Mortalidade', ax=ax[1],
                     color='darkblue', legend=False)
        ax[1].set_title('Taxa de Mortalidade por País')
        ax[1].set_xlabel('Mortes/Julgamentos')
        
        plt.tight_layout()
        plt.show()
        
        self._visualizar_grafo(title="Grafo após Análise de Top Países")
        
        return df

    def top_localizacoes(self, n=10):
        """
        Identifica as localizações (cidade, estado, país) com o maior número de julgamentos e mortes.
        Responde à parte da pergunta sobre 'Quais regiões concentraram o maior número...'.
        """
        if not hasattr(self, 'G') or not self.G.nodes:
            print("Grafo não foi construído corretamente ou está vazio.")
            return pd.DataFrame()

        locais_data = defaultdict(lambda: {'Julgamentos': 0, 'Mortes': 0})

        for no in self.G.nodes():
            if self.G.nodes.get(no, {}).get('tipo') == 'local':
                localizacao = no # O nome do nó já é a 'localizacao' consolidada
                julgamentos = sum(data.get('weight', 0) for _, _, data in self.G.out_edges(no, data=True))
                mortes = sum(data.get('mortes', 0) for _, _, data in self.G.out_edges(no, data=True))

                locais_data[localizacao]['Julgamentos'] += julgamentos
                locais_data[localizacao]['Mortes'] += mortes

        resultados = [{
            'Localização': loc,
            'Julgamentos': dados['Julgamentos'],
            'Mortes': dados['Mortes'],
            'Taxa Mortalidade': dados['Mortes'] / dados['Julgamentos'] if dados['Julgamentos'] > 0 else 0
        } for loc, dados in locais_data.items() if dados['Julgamentos'] > 0]

        df_locais = pd.DataFrame(resultados).sort_values('Julgamentos', ascending=False).head(n)

        if df_locais.empty:
            print("Nenhuma localização com julgamentos encontrada.")
            return df_locais

        # Plot para Top Localizações por Julgamentos
        fig, ax = plt.subplots(1, 2, figsize=(18, 7)) # Ajustado para melhor visualização de nomes longos
        
        df_locais.plot.barh(x='Localização', y='Julgamentos', ax=ax[0], 
                            color='darkgreen', legend=False)
        ax[0].set_title(f'Top {n} Localizações por Julgamentos')
        ax[0].set_xlabel('Número de Julgamentos')
        ax[0].invert_yaxis() # Para ter o maior no topo

        df_locais.plot.barh(x='Localização', y='Taxa Mortalidade', ax=ax[1],
                            color='darkorange', legend=False)
        ax[1].set_title(f'Taxa de Mortalidade nas Top {n} Localizações')
        ax[1].set_xlabel('Mortes/Julgamentos')
        ax[1].invert_yaxis() # Para ter o maior no topo
        
        plt.tight_layout()
        plt.show()

        self._visualizar_grafo(title="Grafo após Análise de Top Localizações")

        return df_locais

    def analisar_temporal(self):
        """
        Analisa padrões temporais nos julgamentos e mortes por década.
        Responde à pergunta sobre 'Há padrões temporais que indiquem ondas de perseguição?'.
        """
        if not hasattr(self, 'G') or not self.G.nodes:
            print("Grafo não foi construído corretamente ou está vazio.")
            return pd.DataFrame()
            
        # Pega todas as décadas válidas e as ordena
        # Garante que 'n' seja uma string antes de chamar isdigit()
        decadas = sorted([n for n in self.G.nodes() 
                          if self.G.nodes.get(n, {}).get('tipo') == 'decada' 
                          and str(n) != 'ND' and str(n).isdigit()],
                         key=lambda x: int(x)) # Converte para int para ordenar numericamente
        
        ts_data = []
        for decada in decadas:
            # Soma os julgamentos e mortes das arestas que chegam na década
            julgamentos = sum(data.get('weight', 0) for _, _, data in self.G.in_edges(decada, data=True))
            mortes = sum(data.get('mortes', 0) for _, _, data in self.G.in_edges(decada, data=True))
            
            if julgamentos > 0:
                ts_data.append({
                    'Década': int(decada), # Converte para int para plotar como número
                    'Julgamentos': julgamentos,
                    'Mortes': mortes,
                    'Taxa Mortalidade': mortes/julgamentos if julgamentos > 0 else 0
                })
            
        ts_df = pd.DataFrame(ts_data).set_index('Década').sort_index() # Ordena pelo índice da década
        
        if ts_df.empty:
            print("Nenhuma década com julgamentos encontrada.")
            return ts_df
            
        # Plot dos resultados temporais
        fig, ax = plt.subplots(2, 1, figsize=(12, 8))
        
        ts_df['Julgamentos'].plot(ax=ax[0], marker='o', color='darkred', 
                                  title='Julgamentos por Década (Padrões Temporais)')
        ax[0].set_ylabel('Número de Julgamentos')
        ax[0].grid(True, linestyle='--', alpha=0.6)
        
        ts_df['Taxa Mortalidade'].plot(ax=ax[1], marker='o', color='darkblue',
                                      title='Taxa de Mortalidade por Década')
        ax[1].set_ylabel('Taxa de Mortalidade (Mortes/Julgamentos)')
        ax[1].grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        plt.show()
        
        self._visualizar_grafo(title="Grafo após Análise Temporal")
        
        return ts_df

    def analisar_distribuicao_geografica(self):
        """
        Analisa a distribuição geográfica dos julgamentos e retorna um DataFrame com os dados.
        A visualização ajuda a inferir sobre 'correlação entre proximidade geográfica e número de julgamentos'.
        """
        if not hasattr(self, 'G') or not self.G.nodes:
            print("Grafo não foi construído corretamente ou está vazio.")
            return pd.DataFrame()
            
        # Filtra apenas nós de localização que possuem latitude e longitude
        locais = [n for n in self.G.nodes() 
                  if (self.G.nodes.get(n, {}).get('tipo') == 'local' and 
                      self.G.nodes.get(n, {}).get('lat') is not None and 
                      self.G.nodes.get(n, {}).get('lon') is not None)]
        
        if not locais:
            print("Nenhum dado geográfico completo (latitude/longitude) disponível para análise.")
            return pd.DataFrame()
            
        geo_data = []
        for local in locais:
            julgamentos = sum(data.get('weight', 0) for _, _, data in self.G.out_edges(local, data=True))
            mortes = sum(data.get('mortes', 0) for _, _, data in self.G.out_edges(local, data=True)) # Inclui mortes
            geo_data.append({
                'Local': local,
                'Julgamentos': julgamentos,
                'Mortes': mortes,
                'Latitude': self.G.nodes[local]['lat'],
                'Longitude': self.G.nodes[local]['lon'],
                'País': self.G.nodes[local]['pais']
            })
            
        geo_df = pd.DataFrame(geo_data)
        
        # Plot da distribuição geográfica
        plt.figure(figsize=(14, 7))
        # Para melhorar a distinção, pode-se usar cores diferentes por país ou um esquema de cores baseado em uma variável.
        # Aqui, vamos usar um colormap e normalizar o tamanho dos pontos.
        scatter = plt.scatter(geo_df['Longitude'], geo_df['Latitude'],
                              s=geo_df['Julgamentos']/5 + 5, # Tamanho base + proporcional aos julgamentos
                              c=geo_df['Julgamentos'], cmap='YlOrRd', # Cor baseada no número de julgamentos
                              alpha=0.7, edgecolors='w', linewidth=0.5)
        
        plt.colorbar(scatter, label='Número de Julgamentos')
        plt.title('Distribuição Geográfica dos Julgamentos (Tamanho do Ponto = Julgamentos)')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        print("\nAnálise Geográfica: Observar o mapa para identificar concentrações.")
        print("Para uma correlação estatística entre proximidade geográfica e número de julgamentos,")
        print("seriam necessárias técnicas de estatística espacial (e.g., Moran's I), que são mais complexas.")
        
        self._visualizar_grafo(title="Grafo após Análise de Distribuição Geográfica")
        
        return geo_df

    def calcular_taxas_mortalidade_geral(self):
        """
        Calcula as taxas de mortalidade geral baseada na quantidade total de julgamentos e mortes.
        Responde diretamente à pergunta 'Quais são as taxas de mortalidade baseada na quantidade de julgamentos?'.
        """
        if not hasattr(self, 'df') or self.df.empty:
            print("DataFrame não carregado ou está vazio para calcular taxas de mortalidade.")
            return None, None, None # Retorna None para todos os valores em caso de erro
        
        total_julgamentos = self.df['tried'].sum()
        total_mortes = self.df['deaths'].sum()
        
        taxa_mortalidade_geral = (total_mortes / total_julgamentos) if total_julgamentos > 0 else 0
        
        print(f"\nTaxa de Mortalidade Geral:")
        print(f"Total de Julgamentos: {total_julgamentos}")
        print(f"Total de Mortes: {total_mortes}")
        print(f"Taxa de Mortalidade Geral (Mortes/Julgamentos): {taxa_mortalidade_geral:.4f}")

        # Não há um plot específico para a taxa geral, mas o grafo pode dar um contexto
        self._visualizar_grafo(title="Grafo após Cálculo de Taxas de Mortalidade Geral")

        return total_julgamentos, total_mortes, taxa_mortalidade_geral

    def executar_analise_completa(self):
        """
        Executa todas as análises automaticamente e imprime um resumo dos resultados.
        """
        print("\n=== ANÁLISE DE JULGAMENTOS HISTÓRICOS INICIADA ===\n")
        resultados = {}
        
        print("1. Regiões com maior número de julgamentos e mortes (por País):")
        resultados['top_paises'] = self.top_paises()
        
        print("\n1b. Regiões com maior número de julgamentos e mortes (por Localização detalhada):")
        resultados['top_localizacoes'] = self.top_localizacoes()

        print("\n2. Padrões temporais (ondas de perseguição):")
        resultados['analise_temporal'] = self.analisar_temporal()
        
        print("\n3. Distribuição geográfica (correlação entre proximidade geográfica e número de julgamentos - visual):")
        resultados['distribuicao_geografica'] = self.analisar_distribuicao_geografica()
        
        print("\n4. Taxas de mortalidade baseada na quantidade de julgamentos:")
        resultados['taxas_mortalidade_geral'] = self.calcular_taxas_mortalidade_geral()
        
        print("\n=== ANÁLISE CONCLUÍDA ===")
        return resultados

# Bloco de execução principal
if __name__ == "__main__":
    try:
        print("Configurando análise de julgamentos históricos...")
        # Certifique-se de que 'trials.csv' está no mesmo diretório ou forneça o caminho completo
        analisador = WitchTrialsAnalysis('trials.csv') 
        
        # Só executa as análises se o grafo foi construído com sucesso
        if analisador.G and analisador.G.nodes:
            resultados = analisador.executar_analise_completa()
            
            print("\n--- Resumo Detalhado dos Resultados ---")
            
            # Resposta para a pergunta 1: Quais regiões concentraram o maior número de julgamentos e mortes?
            if not resultados['top_paises'].empty:
                print("\n**1. Regiões (Países) com Mais Julgamentos e Mortes:**")
                print(resultados['top_paises'].head())
            if not resultados['top_localizacoes'].empty:
                print("\n**1b. Regiões (Localizações Detalhadas) com Mais Julgamentos e Mortes:**")
                print(resultados['top_localizacoes'].head())

            # Resposta para a pergunta 2: Há padrões temporais que indiquem ondas de perseguição?
            if not resultados['analise_temporal'].empty:
                print("\n**2. Padrões Temporais (Julgamentos e Taxa de Mortalidade por Década):**")
                print("Observe os gráficos gerados para identificar picos e vales, que podem indicar ondas de perseguição.")
                print("As décadas com mais julgamentos e as maiores taxas de mortalidade são:")
                print(resultados['analise_temporal'].sort_values('Julgamentos', ascending=False).head(5))
                print(resultados['analise_temporal'].sort_values('Taxa Mortalidade', ascending=False).head(5))

            # Resposta para a pergunta 3: Existe correlação entre proximidade geográfica e número de julgamentos?
            if not resultados['distribuicao_geografica'].empty:
                print("\n**3. Correlação entre Proximidade Geográfica e Julgamentos:**")
                print("O gráfico de dispersão com o tamanho dos pontos proporcionais aos julgamentos")
                print("permite uma inspeção visual de clusters geográficos. Regiões com pontos maiores e mais densos")
                print("sugerem maior concentração de julgamentos.")
                print("Para uma análise quantitativa de correlação espacial, seriam necessárias métricas como o I de Moran.")
                print("\nTop 5 Localizações por Julgamentos (Geográfico):")
                print(resultados['distribuicao_geografica'].sort_values('Julgamentos', ascending=False).head(5))

            # Resposta para a pergunta 4: Quais são as taxas de mortalidade baseada na quantidade de julgamentos?
            if resultados['taxas_mortalidade_geral'] is not None:
                print(f"\n**4. Taxa de Mortalidade Geral:** {resultados['taxas_mortalidade_geral'][2]:.4f}")
                print("As taxas de mortalidade também são apresentadas por país e por década nos gráficos específicos.")

        else:
            print("\nNão foi possível executar a análise completa devido a um erro na construção do grafo.")
            
    except FileNotFoundError:
        print("\nERRO: O arquivo 'trials.csv' não foi encontrado. Por favor, verifique o caminho do arquivo.")
    except Exception as e:
        print(f"\nERRO inesperado durante a execução principal: {str(e)}")
    finally:
        print("\nProcesso de análise concluído.")