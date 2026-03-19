'''
-------------------------------------------------------------------
App principal com interface de chat via Streamlit (dados CSV)
--------------------------------------------------------------------
'''
import streamlit as st
from rag_engine import criar_chroma, criar_chain_rag_custom as criar_chain_rag, carregar_chroma
from csv_loader import carregar_csvs, preparar_dados_para_rag
from utils import chunk_textos
from langchain.memory import ConversationBufferMemory

# Inicializa memória da conversa
if "memoria" not in st.session_state:
    st.session_state.memoria = ConversationBufferMemory(return_messages=True)

if "chain" not in st.session_state:
    st.text("🔄 Carregando...")
    try:
        vectordb = carregar_chroma()
    except Exception as e:
        st.warning("⚠️ Problema ao carregar base vetorial, criando do zero...")
        df_estacoes, df_veiculos = carregar_csvs()
        textos = preparar_dados_para_rag(df_estacoes, df_veiculos)
        docs = chunk_textos(textos)
        vectordb = criar_chroma(docs)

    st.session_state.chain = criar_chain_rag(vectordb)

# Sidebar
with st.sidebar:
    recarregar = st.toggle("🔁 Recarregar base ", value=False)

# Título principal
st.title("🔌 Análise de Infraestrutura e Veículos Elétricos")

# Entrada de pergunta
pergunta = st.chat_input("Digite sua pergunta sobre os veículos...")

if pergunta:
    with st.spinner("🔍 Buscando resposta..."):
        if recarregar:
            df_estacoes, df_veiculos = carregar_csvs()
            textos = preparar_dados_para_rag(df_estacoes, df_veiculos)
            docs = chunk_textos(textos)
            vectordb = criar_chroma(docs)
            chain = criar_chain_rag(vectordb)
        else:
            chain = st.session_state.chain

        resposta = chain.run(pergunta)

        st.chat_message("user").write(pergunta)
        st.chat_message("assistant").markdown(resposta, unsafe_allow_html=True)
