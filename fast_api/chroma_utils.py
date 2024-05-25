import chromadb
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import (
    Document,
    StorageContext,
    ServiceContext,
    VectorStoreIndex,
    SimpleKeywordTableIndex,
)
from typing import List
import crud
from database import SessionLocal

from models import OCR

import json


class ChromaUtils:
    def __init__(self):
        self.db = chromadb.PersistentClient("chromadb")
        self.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/LaBSE", device="cpu"
        )
        self.chroma_collection = self.db.get_or_create_collection("complaints-prod")
        self.vector_store = ChromaVectorStore(self.chroma_collection)
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        self.service_context = ServiceContext.from_defaults(
            embed_model=self.embed_model, chunk_size=200, llm=None
        )
        self.vector_store_index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            storage_context=self.storage_context,
            service_context=self.service_context,
            show_progress=True,
        )
        self.query_retriver = self.vector_store_index.as_retriever(
            service_context=self.service_context,
            similarity_top_k=20,
        )

        self.keyword_index = self.create_keyword_db()

        self.keyword_retriver = self.keyword_index.as_retriever(
            service_context=self.service_context,
            similarity_top_k=30,
        )

    def add_collections(self, doc_data):
        docs = []
        for i in range(len(doc_data["yolo_text"])):
            # for doc_data in doc_datas:
            doc = Document(
                text=doc_data["yolo_text"][i] + "\n" + doc_data["surya_text"][i],
                metadata={
                    "filename": doc_data["filename"],
                    "docetID": doc_data["docetID"],
                    "page_number": i,
                },
                excluded_llm_metadata_keys=["filename"],
                metadata_seperator="::",
                metadata_template="{key}=>{value}",
                text_template="Metadata: {metadata_str}\n-----\nContent: {content}",
            )
            self.keyword_index.insert(document=doc)
            docs.append(doc)
        self.chroma_collection.add(
            documents=[doc.text for doc in docs],
            embeddings=[self.embed_model.get_text_embedding(doc.text) for doc in docs],
            metadatas=[doc.metadata for doc in docs],
            ids=[
                doc.metadata["docetID"] + str(doc.metadata["page_number"])
                for doc in docs
            ],
        )
        self.vector_store_index.from_vector_store(
            vector_store=self.vector_store,
            storage_context=self.storage_context,
            service_context=self.service_context,
            show_progress=True,
        )
        self.query_retriver = self.vector_store_index.as_retriever(
            service_context=self.service_context,
            similarity_top_k=20,
        )

    def vector_search(self, query_str: str, db):
        retrived_results = crud.get_string_matches(db=db, search_text=query_str)
        retrived_results.extend(
            [
                r.node.metadata["docetID"]
                for r in self.keyword_retriver.retrieve(query_str)
                if r not in retrived_results
            ]
        )
        retrived_results.extend(
            [
                r
                for r in crud.get_substring_matches(db=db, search_text=query_str)
                if r not in retrived_results
            ]
        )
        retrived_results.extend(
            [
                r.node.metadata["docetID"]
                for r in self.query_retriver.retrieve(query_str)
                if r not in retrived_results
            ]
        )

        return retrived_results

    def clean_ocr_text(self, ocr_text):
        output = (
            ocr_text.replace("\\n", " ")
            .lstrip("{")
            .replace("'", "")
            .rstrip("}")
            .replace("\x0c", " ")
            .replace("{", " ")
            .replace("}", " ")
            .replace("set()", " ")
            .replace(",", "")
            .replace("DATE: ", "")
            .replace("DOC_ID:", "")
            .replace("FROM:", "")
            .replace("FROM_ADD:", "")
            .replace("INDICATION:", "")
            .replace("PERSONAL_NAME:", "")
            .replace("PHONE:", "")
            .replace("PLACE:", "")
            .replace("SUBJECT:", "")
            .replace("TO:", "")
            .replace("TO_ADD:", "")
            .replace("ല്\\u200d", "ൽ")
            .replace("ള്\\u200d", "ൾ")
            .replace("ന്\\u200d", "ൻ")
            .replace("ര്\\u200d", "ർ")
            .replace("ണ്\\u200d", "ൺ")
            .replace("\\u200c", "")
            .replace("\\x0c", " ")
        )
        return output

    def create_keyword_db(self):
        db = SessionLocal()
        ocr_list = crud.get_db_all_ocr(db=db)
        docs = []
        for ocr in ocr_list:
            docs.append(
                Document(
                    text=self.clean_ocr_text(ocr.entities),
                    metadata={
                        "filename": ocr.filename,
                        "docetID": ocr.docetID,
                        "page_number": 0,
                    },
                    excluded_llm_metadata_keys=["filename"],
                    metadata_seperator="::",
                    metadata_template="{key}=>{value}",
                    text_template="Metadata: {metadata_str}\n-----\nContent: {content}",
                )
            )

        keyword_index = SimpleKeywordTableIndex.from_documents(
            docs,
            storage_context=self.storage_context,
            service_context=self.service_context,
            show_progress=True,
            workers=16,
        )

        db.close()

        return keyword_index
