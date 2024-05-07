import chromadb
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Document


db = chromadb.PersistentClient("chromadb")

embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/LaBSE", device="cpu"
)

chroma_collection = db.get_or_create_collection("complaints")


def add_collections(doc_data):
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
    chroma_collection.add(
        documents=[doc.text for doc in docs],
        embeddings=[embed_model.get_text_embedding(doc.text) for doc in docs],
        metadatas=[doc.metadata for doc in docs],
        ids=[
            doc.metadata["docetID"] + str(doc.metadata["page_number"]) for doc in docs
        ],
    )
