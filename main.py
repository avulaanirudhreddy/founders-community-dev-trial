from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import os, warnings
warnings.filterwarnings('ignore')
import streamlit as st

load_dotenv()

st.title('AI-Powered Research Paper Assistant')

if "OPENAI_API_KEY" in st.secrets['secrets']:
    api_key = st.secrets['secrets']["OPENAI_API_KEY"]
else:
    api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("Missing OPENAI_API_KEY! Please set it in the secrets.toml file or as an environment variable.")

model = ChatOpenAI(api_key=api_key)
embeddings = OpenAIEmbeddings(api_key=api_key)
parser = StrOutputParser()

uploaded_file = st.file_uploader("Upload a research paper in PDF format:", type="pdf")

if uploaded_file is not None:
    st.write('Processing your PDF')

    temp_pdf_path = "temp_uploaded.pdf"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    loader = PyPDFLoader(temp_pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=50)
    splitted_docs = text_splitter.split_documents(documents)

    faiss = FAISS.from_documents(splitted_docs, embeddings)
    retriever = faiss.as_retriever()

    full_document_text = "\n".join([doc.page_content for doc in documents])
    st.success('PDF processed successfully')

    query = st.text_input('Ask a question about the research paper:')

    if query:
        relevant_docs = retriever.get_relevant_documents(query)
        relevant_context = "\n".join([doc.page_content for doc in relevant_docs])

        # Generate Research Paper Summary
        prompt1 = PromptTemplate(
            template='Given the following research paper, generate a clear and concise summary.\n{context}',
            input_variables=['context']
        )
        summary_chain = prompt1 | model | parser

        # Generate Research Paper Citations and References
        prompt2 = PromptTemplate(
            template='Extract all citations and references from the following research paper. Format them in APA style.\n{context}',
            input_variables=['context']
        )
        citation_chain = prompt2 | model | parser

        # Answer user query via RAG
        prompt3 = PromptTemplate(
            template='Answer the following question about the given research paper:\nQuestion: {query}\nContext: {context}',
            input_variables=['query', 'context']
        )
        qa_chain = prompt3 | model | parser

        parallel_chain = RunnableParallel({
            'summary': summary_chain,
            'citations': citation_chain,
            'answer': qa_chain
        }) 
        results = parallel_chain.invoke({'context': full_document_text, 
                               'query': query, 
                               'answer': {'query': query, 'context': relevant_context}})

        # Display results
        st.subheader("📃 Research Paper Summary")
        st.write(results["summary"])

        st.subheader("📖 Citations & References")
        st.write(results["citations"])

        st.subheader("❓ Query Answer")
        st.write(results["answer"])