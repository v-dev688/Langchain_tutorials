import streamlit as st
import os
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader

from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# ──────────────────────────────────────────────
# HARDCODED PATH TO YOUR PDF FOLDER
# ──────────────────────────────────────────────
DOCS_FOLDER = r"C:\Users\vdrss\Downloads\Langchain_tutorials\3_SIMPLE_RAG_APP\CSA dcouments"

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Cricket South Africa Chatbot",
    page_icon="csa_logo.png",
    layout="centered",
)

# ──────────────────────────────────────────────
# Light green background + styling
# ──────────────────────────────────────────────
st.markdown(
    """
    <style>
    .stApp {
        background-color: #e8f5e9;
    }
    .main .block-container {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 2px 12px rgba(0, 100, 0, 0.08);
    }
    [data-testid="stSidebar"] {
        background-color: #f1f8e9;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# Header: logo + title (only once)
# ──────────────────────────────────────────────
col1, col2 = st.columns([1, 6], vertical_alignment="center")
with col1:
    st.image("csa_logo.png", width=90)
with col2:
    st.markdown(
        "<h1 style='color:#1b5e20; margin:0;'>Cricket South Africa Chatbot</h1>",
        unsafe_allow_html=True,
    )

st.caption("Ask questions based on Cricket South Africa documents")

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    default_key = os.getenv("GROQ_API_KEY", "")
    api_key = st.text_input(
        "Groq API Key",
        value=default_key,
        type="password",
        help="Get a free key at https://console.groq.com",
    )

    model_name = st.selectbox(
        "Model",
        options=[
            "openai/gpt-oss-120b",
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ],
        index=0,
    )

    st.divider()
    st.markdown("Documents are loaded automatically from the code.")

# ──────────────────────────────────────────────
# Simple TF-IDF retriever (no PyTorch)
# ──────────────────────────────────────────────
class SimpleRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.texts = [doc.page_content for doc in documents]
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.texts)

    def get_relevant_documents(self, query, k=6):
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:k]
        return [self.documents[i] for i in top_indices if scores[i] > 0]


# ──────────────────────────────────────────────
# Load & index documents (cached)
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading & indexing CSA PDFs…")
def load_and_index_documents(folder_path: str):
    path = Path(folder_path)
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    pdf_files = list(path.glob("**/*.pdf"))
    if not pdf_files:
        raise ValueError("No PDF files found in the folder.")

    documents = []
    failed = []

    for pdf in pdf_files:
        try:
            reader = PdfReader(str(pdf))
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={"source": pdf.name, "page": i + 1},
                        )
                    )
        except Exception as e:
            failed.append((pdf.name, str(e)))

    if not documents:
        error_msg = "Could not load any PDFs.\n"
        error_msg += "\n".join(f"- {name}: {err}" for name, err in failed)
        raise ValueError(error_msg)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)
    retriever = SimpleRetriever(chunks)

    return retriever, len(documents), len(chunks), failed


# Load documents automatically
try:
    retriever, num_pages, num_chunks, failed = load_and_index_documents(DOCS_FOLDER)
    st.sidebar.success(f"✅ Loaded {num_pages} pages → {num_chunks} chunks")
    if failed:
        st.sidebar.warning(
            f"⚠️ Skipped {len(failed)} file(s):\n"
            + "\n".join(f"• {name}" for name, _ in failed)
        )
except Exception as e:
    st.error(f"Failed to load documents: {e}")
    st.stop()

# ──────────────────────────────────────────────
# Prompt
# ──────────────────────────────────────────────
PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an intelligent assistant trained to answer questions using only the provided context from Cricket South Africa documents.

Context:
{context}

Question: {question}

Answer the question clearly and concisely using only the information in the context.
If the answer is not present in the context, say "I don't know based on the available documents."
""",
)

# ──────────────────────────────────────────────
# Chat history
# ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ──────────────────────────────────────────────
# Chat UI
# ──────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask a question about the CSA documents…"):
    if not api_key:
        st.error("Please enter your Groq API key in the sidebar.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            relevant_docs = retriever.get_relevant_documents(question, k=6)
            context = "\n\n".join(doc.page_content for doc in relevant_docs)

            llm = ChatGroq(
                model=model_name,
                api_key=api_key,
                temperature=0.2,
            )

            formatted_prompt = PROMPT.format(
                context=context,
                question=question,
            )

            for chunk in llm.stream(formatted_prompt):
                if chunk.content:
                    full_response += chunk.content
                    message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

        except Exception as e:
            full_response = f"⚠️ Error: {str(e)}"
            message_placeholder.error(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Clear chat button
if st.session_state.messages:
    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()