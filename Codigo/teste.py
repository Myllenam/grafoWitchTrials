import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 8)
pd.set_option('display.max_columns', 50)

class WitchTrialsAnalysis:
    def __init__(self, arquivo_dados):
        """Inicializa a análise carregando os dados"""
        try:
            self.df = pd.read_csv(arquivo_dados, delimiter=';')
            print("Dados carregados com sucesso!")
            print(f"Total de registros: {len(self.df)}")
            
            self._preprocessar_dados()
            self._construir_grafo()
            
        except Exception as e:
            print(f"\nERRO durante inicialização: {str(e)}")
            self.G = nx.DiGraph()

    def _preprocessar_dados(self):
        """Prepara os dados para análise"""
        # Filtra registros onde o país é desconhecido
        self.df = self.df[self.df['gadm.adm0'].notna()]
        
        # Preenche valores faltantes
        self.df['city'] = self.df['city'].fillna('Desconhecido')
        self.df['decade'] = self.df['decade'].fillna('ND')
        
        # Converte números
        self.df['tried'] = pd.to_numeric(self.df['tried'], errors='coerce').fillna(0).astype(int)
        self.df['deaths'] = pd.to_numeric(self.df['deaths'], errors='coerce').fillna(0).astype(int)
        
        # Cria identificador de localização
        self.df['localizacao'] = self.df.apply(
            lambda x: f"{x['city']}, {x['gadm.adm1']}" if pd.notna(x['gadm.adm1']) else x['city'], axis=1)
        self.df['localizacao'] = self.df['localizacao'] + ", " + self.df['gadm.adm0']

    def _construir_grafo(self):
        """Constrói o grafo networkx a partir dos dados"""
        self.G = nx.DiGraph()
        
        for _, row in self.df.iterrows():
            if pd.isna(row['gadm.adm0']):
                continue
                
            origem = row['localizacao']
            decada = row['decade']
            
            julgamentos = row['tried']
            mortes = row['deaths']
            
            # Adiciona nós com atributos
            self.G.add_node(origem, tipo='local', 
                          lat=row['lat'] if pd.notna(row['lat']) else None,
                          lon=row['lon'] if pd.notna(row['lon']) else None,
                          pais=row['gadm.adm0'])
            
            self.G.add_node(decada, tipo='decada')
            
            # Adiciona arestas com pesos
            if self.G.has_edge(origem, decada):
                self.G[origem][decada]['weight'] += julgamentos
                self.G[origem][decada]['mortes'] += mortes
            else:
                self.G.add_edge(origem, decada, 
                              weight=julgamentos,
                              mortes=mortes)
                
        print(f"\nGrafo construído com:")
        print(f"- {len([n for n in self.G.nodes() if self.G.nodes[n]['tipo'] == 'local'])} locais")
        print(f"- {len([n for n in self.G.nodes() if self.G.nodes[n]['tipo'] == 'decada'])} décadas")
        print(f"- {len(self.G.edges())} conexões")

    def top_paises(self, n=10):
        """Identifica os países com mais julgamentos (agrupado por país)"""
        if not hasattr(self, 'G'):
            print("Grafo não foi construído corretamente")
            return pd.DataFrame()
            
        # Agrupa julgamentos por país
        paises = defaultdict(lambda: {'Julgamentos': 0, 'Mortes': 0})
        
        for no in self.G.nodes():
            if self.G.nodes[no]['tipo'] == 'local' and 'pais' in self.G.nodes[no]:
                pais = self.G.nodes[no]['pais']
                julgamentos = sum(data['weight'] for _, _, data in self.G.out_edges(no, data=True))
                mortes = sum(data['mortes'] for _, _, data in self.G.out_edges(no, data=True))
                
                paises[pais]['Julgamentos'] += julgamentos
                paises[pais]['Mortes'] += mortes
        
        # Converte para DataFrame
        resultados = [{
            'País': pais,
            'Julgamentos': dados['Julgamentos'],
            'Mortes': dados['Mortes'],
            'Taxa Mortalidade': dados['Mortes']/dados['Julgamentos'] if dados['Julgamentos'] > 0 else 0
        } for pais, dados in paises.items() if dados['Julgamentos'] > 0]
        
        df = pd.DataFrame(resultados).sort_values('Julgamentos', ascending=False).head(n)
        
        if df.empty:
            print("Nenhum país com julgamentos encontrado")
            return df
        
        # Plot
        fig, ax = plt.subplots(1, 2, figsize=(16, 6))
        
        df.plot.barh(x='País', y='Julgamentos', ax=ax[0], 
                    color='darkred', legend=False)
        ax[0].set_title('Top Países por Julgamentos')
        ax[0].set_xlabel('Número de Julgamentos')
        
        df.plot.barh(x='País', y='Taxa Mortalidade', ax=ax[1],
                    color='darkblue', legend=False)
        ax[1].set_title('Taxa de Mortalidade')
        ax[1].set_xlabel('Mortes/Julgamentos')
        
        plt.tight_layout()
        plt.show()
        
        return df

    def analisar_temporal(self):
        """Analisa padrões temporais nos julgamentos"""
        if not hasattr(self, 'G'):
            print("Grafo não foi construído corretamente")
            return pd.DataFrame()
            
        decadas = sorted([n for n in self.G.nodes() 
                         if self.G.nodes[n]['tipo'] == 'decada' 
                         and n != 'ND'])
        
        ts_data = []
        for decada in decadas:
            julgamentos = sum(data['weight'] for _, _, data in self.G.in_edges(decada, data=True))
            mortes = sum(data['mortes'] for _, _, data in self.G.in_edges(decada, data=True))
            
            if julgamentos > 0:
                ts_data.append({
                    'Década': decada,
                    'Julgamentos': julgamentos,
                    'Mortes': mortes,
                    'Taxa Mortalidade': mortes/julgamentos if julgamentos > 0 else 0
                })
            
        ts_df = pd.DataFrame(ts_data).set_index('Década')
        
        if ts_df.empty:
            print("Nenhuma década com julgamentos encontrada")
            return ts_df
        
        # Plot
        fig, ax = plt.subplots(2, 1, figsize=(12, 8))
        
        ts_df['Julgamentos'].plot(ax=ax[0], marker='o', color='darkred', 
                                title='Julgamentos por Década')
        ts_df['Taxa Mortalidade'].plot(ax=ax[1], marker='o', color='darkblue',
                                      title='Taxa de Mortalidade por Década')
        
        plt.tight_layout()
        plt.show()
        
        return ts_df

    def analisar_distribuicao_geografica(self):
        """Analisa a distribuição geográfica dos julgamentos (igual à versão consolidada)"""
        if not hasattr(self, 'G'):
            print("Grafo não foi construído corretamente")
            return
        
        locais = [n for n in self.G.nodes() 
                 if (self.G.nodes[n]['tipo'] == 'local' and 
                     self.G.nodes[n]['lat'] is not None and 
                     self.G.nodes[n]['lon'] is not None)]
        
        if not locais:
            print("Nenhum dado geográfico completo disponível")
            return
        
        geo_data = []
        for local in locais:
            julgamentos = sum(data['weight'] for _, _, data in self.G.out_edges(local, data=True))
            geo_data.append({
                'Local': local,
                'Julgamentos': julgamentos,
                'Latitude': self.G.nodes[local]['lat'],
                'Longitude': self.G.nodes[local]['lon'],
                'País': self.G.nodes[local]['pais']
            })
        
        geo_df = pd.DataFrame(geo_data)
        
        # Plot (idêntico à versão consolidada)
        plt.figure(figsize=(14, 7))
        for pais in geo_df['País'].unique():
            subset = geo_df[geo_df['País'] == pais]
            plt.scatter(subset['Longitude'], subset['Latitude'],
                       s=subset['Julgamentos']/10, alpha=0.6,
                       label=pais)
        
        plt.title('Distribuição Geográfica dos Julgamentos (por País)')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return geo_df

    def executar_analise_completa(self):
        """Executa todas as análises automaticamente"""
        print("\n=== ANÁLISE INICIADA ===\n")
        resultados = {}
        
        print("1. Países com mais julgamentos:")
        resultados['top_paises'] = self.top_paises()
        
        print("\n2. Padrões temporais:")
        resultados['analise_temporal'] = self.analisar_temporal()
        
        print("\n3. Distribuição geográfica:")
        resultados['distribuicao_geografica'] = self.analisar_distribuicao_geografica()
        
        print("\n=== ANÁLISE CONCLUÍDA ===")
        return resultados

# Execução
if __name__ == "__main__":
    try:
        print("Iniciando análise de julgamentos históricos...")
        analisador = WitchTrialsAnalysis('trials.csv')
        resultados = analisador.executar_analise_completa()
        
        print("\nResumo dos Resultados:")
        if not resultados['top_paises'].empty:
            print("\nTop 3 Países:")
            print(resultados['top_paises'].head(3))
        
        if not resultados['analise_temporal'].empty:
            print("\nDécadas com Mais Julgamentos:")
            print(resultados['analise_temporal'].sort_values('Julgamentos', ascending=False).head(3))
            
    except Exception as e:
        print(f"\nERRO durante execução: {str(e)}")
    finally:
        print("\nProcesso concluído")