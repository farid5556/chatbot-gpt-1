from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Weaviate as WeaviateStore
from langchain.schema import Document
from dotenv import load_dotenv
import weaviate
import os

load_dotenv()

def get_client():
    return weaviate.Client(
        url=os.getenv("WEAVIATE_URL"),
        auth_client_secret=weaviate.AuthApiKey(os.getenv("WEAVIATE_API_KEY"))
    )

def upload_documents(texts):
    client = get_client()

    if client.schema.exists("MyDocuments"):
        client.schema.delete_class("MyDocuments")

    client.schema.create_class({
        "class": "MyDocuments",
        "properties": [
            {"name": "content", "dataType": ["text"]}
        ],
        "vectorizer": "none"
    })

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/distiluse-base-multilingual-cased-v1")

    vectorstore = WeaviateStore(client, "MyDocuments", "content", embeddings)

    docs = [Document(page_content=text) for text in texts]
    vectorstore.add_documents(docs)
    print("✅ Dokumen berhasil di-upload ke Weaviate.")

# 👇 Ini akan langsung jalan saat kamu jalankan file ini
if __name__ == "__main__":
    texts = [
        "Indonesia adalah negara dengan jumlah pulau terbanyak di dunia.",
        "Presiden pertama Indonesia adalah Soekarno.",
        "Bahasa resmi Indonesia adalah Bahasa Indonesia.",
        "Gunung tertinggi di Indonesia adalah Puncak Jaya."
    ]
    upload_documents(texts)
