import os
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel
from langchain_community.vectorstores import FAISS

# -------------------------------------------------
# Load environment variables
# -------------------------------------------------
load_dotenv()

# -------------------------------------------------
# Streamlit config
# -------------------------------------------------
st.set_page_config(
    page_title="AI-Powered Research Manager",
    layout="centered"
)

st.title("📚 AI-Powered Research Manager")

# -------------------------------------------------
# Load OpenAI API Key (Streamlit Cloud safe)
# -------------------------------------------------
if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
else:
    api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error(
        "Missing OPENAI_API_KEY.\n\n"
        "Add it in Streamlit → Manage App → Secrets"
    )
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key

# -------------------------------------------------
# Initialize LLM & Embeddings
# -------------------------------------------------
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

parser = StrOutputParser()

# -------------------------------------------------
# File Upload
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a research paper (PDF)",
    type="pdf"
)

if uploaded_file is not None:
    st.info("Processing your PDF…")

    temp_pdf_path = "temp_uploaded.pdf"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # -------------------------------------------------
    # Load & split document
    # -------------------------------------------------
    loader = PyPDFLoader(temp_pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50
    )

    split_docs = splitter.split_documents(documents)

    # -------------------------------------------------
    # Vector Store (FAISS)
    # -------------------------------------------------
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    retriever = vectorstore.as_retriever()

    full_document_text = "\n".join(
        [doc.page_content for doc in documents]
    )

    st.success("PDF processed successfully ✅")

    # -------------------------------------------------
    # User Query
    # -------------------------------------------------
    query = st.text_input(
        "Ask a question about the research paper:"
    )

    if query:
        relevant_docs = retriever.get_relevant_documents(query)
        relevant_context = "\n".join(
            [doc.page_content for doc in relevant_docs]
        )

        # -------------------------------------------------
        # Prompt Templates
        # -------------------------------------------------
        summary_prompt = PromptTemplate(
            template=(
                "Given the following research paper, "
                "generate a clear and concise summary:\n\n{context}"
            ),
            input_variables=["context"]
        )

        citation_prompt = PromptTemplate(
            template=(
                "Extract all citations and references from the following "
                "research paper and format them in APA style:\n\n{context}"
            ),
            input_variables=["context"]
        )

        qa_prompt = PromptTemplate(
            template=(
                "Answer the following question using the given context.\n\n"
                "Question: {query}\n\nContext:\n{context}"
            ),
            input_variables=["query", "context"]
        )

        # -------------------------------------------------
        # Chains
        # -------------------------------------------------
        summary_chain = summary_prompt | model | parser
        citation_chain = citation_prompt | model | parser
        qa_chain = qa_prompt | model | parser

        parallel_chain = RunnableParallel({
            "summary": summary_chain,
            "citations": citation_chain,
            "answer": qa_chain
        })

        # -------------------------------------------------
        # Run chains
        # -------------------------------------------------
        with st.spinner("Generating insights…"):
            results = parallel_chain.invoke({
                "context": full_document_text,
                "query": query,
                "answer": {
                    "query": query,
                    "context": relevant_context
                }
            })

        # -------------------------------------------------
        # Display results
        # -------------------------------------------------
        st.subheader("📄 Research Paper Summary")
        st.write(results["summary"])

        st.subheader("📚 Citations & References")
        st.write(results["citations"])

        st.subheader("❓ Answer to Your Question")
        st.write(results["answer"])
