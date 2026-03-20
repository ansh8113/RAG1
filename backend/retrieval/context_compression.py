from dotenv import load_dotenv
import os
from langchain_core.documents import Document

load_dotenv()



def compress_context(query, docs):

    compressed_docs = []

    for doc in docs:

        text = doc.page_content

        if len(text) > 1500:
            text = text[:1500]

        compressed_docs.append(text)

    return compressed_docs