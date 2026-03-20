import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


def create_vector_store(chunks, persistent_directory):

    documents = []

    for chunk in chunks:

        if isinstance(chunk, Document):
            documents.append(chunk)

        else:
            documents.append(
                Document(
                    page_content=str(chunk),
                    metadata={"source": "multimodal"}
                )
            )
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

  
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persistent_directory
    )
    
   

    return vectorstore
print("Done ")