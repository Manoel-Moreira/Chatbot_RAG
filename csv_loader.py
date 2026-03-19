'''
----------------------------------------
Extrair dados do csv
 ---------------------------------------
 '''
import pandas as pd

#Carregar os dois arquivos CSV e retorna dois DataFrames:
def carregar_csvs():
    df_estacoes = pd.read_csv("arquivos_leitura/vw_station_data2.csv", delimiter=';')
    df_veiculos = pd.read_csv("arquivos_leitura/vw_eletric_vehicle_population.csv", delimiter=';')
    return df_estacoes, df_veiculos

# Juntar e estruturar os dados em textos formatados para análise RAG. Retorna uma lista de strings.
def preparar_dados_para_rag(df_estacoes, df_veiculos):
    textos = []

    # Estações de carregamento
    for _, row in df_estacoes.iterrows():
        texto = f"""
        [Estação de Carregamento]
        Data: {row.get("created_date", "")}
        Hora: {row.get("created_time", "")}
        Energia (kWh): {row.get("kwhTotal", "")}
        Tempo de carga (h): {row.get("chargeTimeHrs", "")}
        Plataforma: {row.get("platform", "")}
        Distância: {row.get("distance", "")}
        Tipo de local: {row.get("facilityType", "")}
        """
        textos.append(texto.strip())

    # Veículos elétricos — usar campos corretos
    for _, row in df_veiculos.iterrows():
        texto = f"""
        [População de Veículos Elétricos]
        Cidade: {row.get("City", "")}
        Ano: {row.get("Model Year", "")}
        Tipo de Veículo: {row.get("Electric Vehicle Type", "")}
        Fabricante: {row.get("Make", "")}
        Modelo: {row.get("Model", "")}
        População estimada: {row.get("populacao", "")}
        """
        textos.append(texto.strip())

    return textos







