import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
import tempfile
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="AI Document Q&A Chatbot", page_icon="📄")
st.title("📄 AI Document Q&A Chatbot")
st.write(
    "Upload a PDF and ask questions about it. This app uses RAG "
    "(Retrieval-Augmented Generation) to answer based on your document's actual content."
)

groq_api_key = st.sidebar.text_input(
    "Groq API Key",
    type="password",
    value=os.getenv("GROQ_API_KEY", ""),
    help="Get a free key at https://console.groq.com/keys",
)

if not groq_api_key:
    st.info("Enter your Groq API key in the sidebar to get started. It's free: https://console.groq.com/keys")
    st.stop()

uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
    st.session_state.file_name = None

if uploaded_file is not None and uploaded_file.name != st.session_state.file_name:
    with st.spinner("Reading and processing document... (first run downloads the embedding model, ~90MB)"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        vectorstore = FAISS.from_documents(chunks, embeddings)

        st.session_state.vectorstore = vectorstore
        st.session_state.file_name = uploaded_file.name
        os.remove(tmp_path)

    st.success(f"'{uploaded_file.name}' processed into {len(chunks)} chunks. Ask away!")

if st.session_state.vectorstore is not None:
    question = st.text_input("Ask a question about your document:")

    if question:
        with st.spinner("Thinking..."):
            llm = ChatGroq(
                groq_api_key=groq_api_key,
                model_name="openai/gpt-oss-20b",
                temperature=0.2,
            )

            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                return_source_documents=True,
            )

            result = qa_chain.invoke({"query": question})

        st.markdown("### Answer")
        st.write(result["result"])

        with st.expander("Show source excerpts used to answer"):
            for i, doc in enumerate(result["source_documents"]):
                page = doc.metadata.get("page", "N/A")
                st.markdown(f"**Excerpt {i + 1}** (page {page})")
                st.write(doc.page_content[:400] + "...")