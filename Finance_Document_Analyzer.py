import os
import shutil
import tempfile
import streamlit as st
import pandas as pd
import altair as alt

from ctransformers import AutoModelForCausalLM
from langchain.llms import CTransformers
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyMuPDFLoader, TextLoader, UnstructuredExcelLoader
from langchain.document_loaders.unstructured import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import tempfile

# === Configuration ===
MODEL_PATH = r"C:\Users\Admin\source\repos\Finance_Document_Analyzer\Finance_Document_Analyzer\models\mistral-7b-instruct-v0.2.Q4_K_M.gguf"
CHROMA_DB_DIR = tempfile.mkdtemp()

# === Load the Local Quantized Mistral LLM ===
@st.cache_resource
def load_local_llm():
    return CTransformers(
        model=MODEL_PATH,
        model_type="mistral",  # Adjust model type if needed (e.g., "llama")
        config={
            "max_new_tokens": 512,
            "context_length": 4096,
            "temperature": 0.7,
            "top_p": 0.9
        }
    )

# === Process and Embed Uploaded Docs ===
def process_documents(uploaded_files):
    all_docs = []
    for uploaded_file in uploaded_files:
        suffix = os.path.splitext(uploaded_file.name)[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        if suffix == ".pdf":
            loader = PyMuPDFLoader(tmp_path)
        elif suffix in [".txt", ".text"]:
            loader = TextLoader(tmp_path)
        elif suffix in [".xls", ".xlsx"]:
            loader = UnstructuredExcelLoader(tmp_path)
        else:
            loader = UnstructuredFileLoader(tmp_path)

        docs = loader.load()
        all_docs.extend(docs)
        os.remove(tmp_path)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(all_docs)

    if os.path.exists(CHROMA_DB_DIR):
        shutil.rmtree(CHROMA_DB_DIR)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(split_docs, embedding=embeddings, persist_directory=CHROMA_DB_DIR)
    vectordb.persist()
    return vectordb, split_docs

# === Build RAG QA Chain ===
def build_qa_chain(vectordb):
    retriever = vectordb.as_retriever()
    llm = load_local_llm()
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# === Extract Financial Keywords for Insights ===
def extract_metrics(split_docs):
    metrics = {"Revenue": [], "Net Income": [], "EBITDA": []}
    for doc in split_docs:
        content_lower = doc.page_content.lower()
        for key in metrics:
            if key.lower() in content_lower:
                metrics[key].append(doc.page_content)
    return metrics

# === Streamlit UI ===
def main():
    st.set_page_config(page_title="Private Financial Document Analyzer", layout="wide")
    st.title("🔐 Financial Document Analyzer")

    tab_chat, tab_insights, tab_charts, tab_data = st.tabs(["💬 Chat", "📊 Insights", "📈 Visualizations", "📁 Raw Data"])

    with tab_chat:
        mode = st.radio("Choose Chat Mode", ["Chat with LLM", "Chat with Uploaded Document"])

        if mode == "Chat with LLM":
            query = st.text_input("Ask something about finance:")
            if query:
                llm = load_local_llm()
                response = llm.invoke(query)  # Use 'invoke' instead of '__call__'
                st.write("**Answer:**", response.strip())  # Directly output the response string

        elif mode == "Chat with Uploaded Document":
            uploaded_files = st.file_uploader("Upload PDF, Excel, or TXT files", type=["pdf", "txt", "xls", "xlsx"], accept_multiple_files=True)
            if uploaded_files:
                with st.spinner("Processing and indexing your documents..."):
                    vectordb, split_docs = process_documents(uploaded_files)
                    qa_chain = build_qa_chain(vectordb)
                st.success("Documents indexed successfully!")

                query = st.text_input("Ask a question based on your uploaded documents:")
                if query:
                    with st.spinner("Generating answer..."):
                        result = qa_chain(query)
                        st.write("**Answer:**", result["result"])
                        with st.expander("📚 Source Documents"):
                            for doc in result["source_documents"]:
                                st.markdown(f"- {doc.metadata.get('source', 'Unknown')}")

                st.session_state["split_docs"] = split_docs

    with tab_insights:
        st.subheader("📊 Key Financial Metrics")
        if "split_docs" in st.session_state:
            metrics = extract_metrics(st.session_state["split_docs"])
            for key, values in metrics.items():
                st.markdown(f"**{key}:**")
                for val in values:
                    st.markdown(f"- {val[:150]}...")
        else:
            st.info("Upload and process documents in the Chat tab to view insights.")

    with tab_charts:
        st.subheader("📈 Auto Visualizations")
        if "split_docs" in st.session_state:
            dummy_data = pd.DataFrame({
                "Year": [2021, 2022, 2023],
                "Revenue": [120, 150, 180],
                "Net Income": [30, 45, 60]
            })
            chart = alt.Chart(dummy_data).mark_line(point=True).encode(
                x="Year:O",
                y="Revenue:Q",
                tooltip=["Year", "Revenue", "Net Income"]
            ).properties(title="Revenue over Years")
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Upload and process documents in the Chat tab to view visualizations.")

    with tab_data:
        st.subheader("📁 Raw Document Chunks")
        if "split_docs" in st.session_state:
            for i, doc in enumerate(st.session_state["split_docs"]):
                st.markdown(f"**Chunk {i+1}:**")
                st.code(doc.page_content[:1000])
        else:
            st.info("Upload and process documents in the Chat tab to see raw chunks.")

if __name__ == "__main__":
    main()
