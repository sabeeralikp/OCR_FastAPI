import chromadb
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import Document, StorageContext, ServiceContext, VectorStoreIndex


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
            similarity_top_k=50,
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
            similarity_top_k=50,
        )

    def vector_search(self, query_str: str):
        retrived_results = self.query_retriver.retrieve(query_str)
        return [rresult.node.metadata for rresult in retrived_results]
