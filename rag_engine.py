'''
----------------------------------------
Engine RAG: embeddings, retriever, QA chain (usando dados CSV)
----------------------------------------
'''
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from utils import garantir_pasta_chroma
from langchain.prompts import PromptTemplate
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from chromadb.config import Settings

# Criar o banco vetorial a partir de documentos
def criar_chroma(documents, path="data/chroma"):
    garantir_pasta_chroma(path)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    client_settings = Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=path
    )

    vectordb = Chroma.from_documents(
        documents,
        embedding=embeddings,
        persist_directory=path
    )
    return vectordb

# Carregar banco vetorial existente
def carregar_chroma():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory="data/chroma"
    )
    return vectordb

# Criar a cadeia de QA com RAG e prompt customizado
def criar_chain_rag_custom(vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    llm = OllamaLLM(model="gemma3:1b", temperature=0.5)

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        Você é um assistente que responde apenas com base nas informações abaixo.

        Essas informações são sobre estações de carregamento de veículos elétricos e a população de veículos elétricos em diferentes cidades.

        **Se a resposta não puder ser respondida com base nessas informações, diga apenas: 'Não encontrei informações suficientes para responder.'**

        Para cada item encontrado, use este formato para responder:

        {context}

        Pergunta:
        {question}

        Resposta:
        """
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template}
    )
