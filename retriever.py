import os
import weaviate
from dotenv import load_dotenv
from langchain_community.vectorstores import Weaviate
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

def get_vectorstore():
    client = weaviate.Client(
        url=os.getenv("WEAVIATE_URL"),
        auth_client_secret=weaviate.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
    )

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/distiluse-base-multilingual-cased-v1")

    # Schema jika belum ada
    class_obj = {
        "class": "MyDocuments",
        "vectorizer": "none",
        "properties": [
            {"name": "content", "dataType": ["text"]}
        ]
    }

    if not client.schema.contains(class_obj):
        client.schema.create_class(class_obj)

        # Tambahkan dokumen awal (dummy)
        docs = [
            "Presiden pertama Indonesia adalah Soekarno.",
            "Presiden Indonesia saat ini adalah Joko Widodo.",
            "Indonesia adalah negara di Asia Tenggara."
        ]
        for doc in docs:
            vec = embeddings.embed_query(doc)
            client.data_object.create(
                data_object={"content": doc},
                class_name="MyDocuments",
                vector=vec
            )

    # Explicitly tell LangChain to use `by_vector`
    vectorstore = Weaviate(
        client=client,
        index_name="MyDocuments",
        text_key="content",
        embedding=embeddings,
        by_text=False  # <--- PENTING!
    )

    return vectorstore
