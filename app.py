import chromadb
import streamlit as st
from llama_index.core import VectorStoreIndex, ServiceContext, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from streamlit_pdf_viewer import pdf_viewer
import spacy_streamlit
import spacy
import streamlit as st
from pdf2image import convert_from_bytes
from pytesseract import image_to_string
import numpy as np
import spacy_transformers


@st.cache_resource
def load_model():
    return spacy.load("../OCR_Extraction/model-best")


nlp = load_model()

colors = {
    "FROM": "#FFB6C1",
    "FROM_ADD": "#ADD8E6",
    "SUBJECT": "#F08080",
    "FROM_CONTACT_INFO": "#E0FFFF",
    "TO": "#FAFAD2",
    "TO_ADD": "#D3D3D3",
    "DATE": "#90EE90",
    "PHONE_NUM": "#FFFFE0",
}


@st.cache_resource
def load_db():
    return chromadb.PersistentClient("../OCR_Retrival_Malayalam/chromadb")


@st.cache_resource
def load_collection():
    return load_db().get_or_create_collection("complaints-2048")


@st.cache_resource
def load_chroma_vector_store():
    return ChromaVectorStore(load_collection())


@st.cache_data
def load_storage_context():
    return StorageContext.from_defaults(
        vector_store=load_chroma_vector_store(),
    )


@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbedding(model_name="sentence-transformers/LaBSE", device="cpu")


@st.cache_resource
def load_service_context():
    return ServiceContext.from_defaults(
        embed_model=load_embedding_model(),
        chunk_size=200,
        llm=None,
    )


@st.cache_resource
def load_vector_store_index():
    return VectorStoreIndex.from_vector_store(
        vector_store=load_chroma_vector_store(),
        storage_context=load_storage_context(),
        service_context=load_service_context(),
        show_progress=True,
    )


@st.cache_resource
def load_retriver():
    return load_vector_store_index().as_retriever(
        service_context=load_service_context(),
        similarity_top_k=3,
    )


query_retriver = load_retriver()

input_text = st.text_input("Enter text to search for")


if st.button("Search"):
    response = query_retriver.retrieve(input_text)
    for r in response:
        with st.container(border=True):
            st.markdown(r.text)
            doc = nlp(r.text)
            spacy_streamlit.visualize_ner(
                doc,
                title="",
                colors=colors,
                key=r.metadata["file_name"].split("/")[-1].replace(".txt", "")
                + r.node_id,
            )
            st.text("Model Score: " + str(r.score))
            st.text("PDF File Name: " + r.metadata["file_name"])
            with open(
                "../OCR_Retrival_Malayalam/"
                + r.metadata["file_name"].replace("text", "pdf").replace("txt", "pdf"),
                "rb",
            ) as f:
                st.download_button(
                    "Download PDF",
                    f,
                    r.metadata["file_name"].split("/")[-1].replace("txt", "pdf"),
                    key=r.metadata["file_name"].split("/")[-1].replace(".txt", "")
                    + r.node_id,
                )
            pdf_viewer(
                "../OCR_Retrival_Malayalam/"
                + r.metadata["file_name"].replace("text", "pdf").replace("txt", "pdf"),
                key=r.node_id,
            )
