# Serve para ver os 05 primeiros dados do chroma
from langchain_ollama.embeddings import OllamaEmbeddings

# Usa o novo Chroma se você tiver atualizado
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

import os

persist_directory = "data/chroma"
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Verifica se a pasta existe
if not os.path.exists(persist_directory):
    print("❌ Diretório do banco vetorial não encontrado:", persist_directory)
    exit()

print("✅ Diretório encontrado, carregando Chroma...")

try:
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    print("✅ Banco carregado com sucesso.")
except Exception as e:
    print("❌ Erro ao carregar banco:", str(e))
    exit()

# Inspeciona o conteúdo
dados = db.get()

if not dados["documents"]:
    print("⚠️ Nenhum documento encontrado no banco vetorial.")
else:
    print(f"✅ {len(dados['documents'])} documentos encontrados. Exibindo os primeiros 5:\n")
    for i, doc in enumerate(dados["documents"][:5]):
        print(f"\n📄 Documento {i+1}:\n{doc}")
