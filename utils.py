'''
----------------------------------------
Funções auxiliares para o projeto com CSV
----------------------------------------
'''

from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import shutil

def chunk_textos(textos):
    """
    Recebe uma lista de strings (textos estruturados) e retorna documentos
    prontos para o uso no Chroma, divididos em chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50
    )
    return splitter.create_documents(textos)


def garantir_pasta_chroma(path="data/chroma"):
    """
    Verifica se a pasta do ChromaDB existe, e cria caso não exista.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Pasta '{path}' criada com sucesso.")
    else:
        print(f"Pasta '{path}' já existe.")


def deletar_chroma(path="data/chroma"):
    """
    Deleta a pasta do banco vetorial Chroma se existir.
    """
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"Pasta '{path}' deletada com sucesso.")
    else:
        print(f"Pasta '{path}' não existe, nada a deletar.")
