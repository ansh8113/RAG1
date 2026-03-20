import os
from dotenv import load_dotenv
from backend.Ingestion.document_loader import load_documents
from backend.Ingestion.chunking import advanced_chunking
from backend.Ingestion.vector_store import create_vector_store
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()
    
def main():

    docs_path = "docs"
    persistent_directory = "db/chroma_db"

    if os.path.exists(persistent_directory):

        print("Vector store already exists.")

        embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

        vectorstore = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embedding_model,
            collection_metadata={"hnsw:space": "cosine"}
        )

        print(f"Loaded {vectorstore._collection.count()} chunks")

        return vectorstore

    print("Running ingestion pipeline...\n")

    documents = load_documents(docs_path)

    chunks = advanced_chunking(documents)

    vectorstore = create_vector_store(chunks, persistent_directory)
    print("Total vectors stored:", vectorstore._collection.count())

    print("\nIngestion complete!")

  #  return vectorstore


if __name__ == "__main__":
    main()