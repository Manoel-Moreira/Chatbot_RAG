from rag_engine import criar_chroma
from csv_loader import carregar_csvs, preparar_dados_para_rag
from utils import chunk_textos, deletar_chroma

print("🧹 Limpando banco vetorial antigo...")
deletar_chroma()

print("📥 Carregando dados dos CSVs...")
df_estacoes, df_veiculos = carregar_csvs()
print(f"🔌 Estações: {len(df_estacoes)} linhas | 🚗 Veículos: {len(df_veiculos)} linhas")

print("🛠️ Formatando textos para o RAG...")
textos = preparar_dados_para_rag(df_estacoes, df_veiculos)

print("🔗 Quebrando em chunks...")
docs = chunk_textos(textos)

print(f"📦 Gerando banco vetorial com {len(docs)} documentos...")
criar_chroma(docs)

print("✅ Banco vetorial recriado com sucesso!")
