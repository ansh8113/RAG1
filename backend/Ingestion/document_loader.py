import os
from langchain_community.document_loaders import UnstructuredFileLoader, DirectoryLoader


def load_documents(docs_path="docs"):
    print(f"Loading documents from {docs_path}...")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"{docs_path} does not exist.")

    loader = DirectoryLoader(
        docs_path,
        glob="**/*",
        loader_cls=UnstructuredFileLoader
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError("No documents found.")

    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i+1}")
        print(f"Source: {doc.metadata['source']}")
        print(f"Preview: {doc.page_content[:100]}")

    return documents