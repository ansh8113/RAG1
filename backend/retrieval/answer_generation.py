import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from backend.retrieval.query_generation import generate_queries
from backend.retrieval.rrf import reciprocal_rank_fusion
from backend.retrieval.reranker import rerank
from backend.retrieval.context_compression import compress_context
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
#import logging
#logging.getLogger("transformers").setLevel(logging.ERROR)


load_dotenv()

persistent_directory = "db/chroma_db"

embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}  
)

model = ChatGroq(model="llama-3.3-70b-versatile",temperature=0.2)

def generate_answer(query):

    queries = generate_queries(query)

    #vector_retriever = db.as_retriever(search_kwargs={"k": 20})
    vector_retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 20,
        "lambda_mult": 0.5
        }
    )

    all_docs = db.get()

    bm25_docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(all_docs["documents"], all_docs["metadatas"])
]

    bm25_retriever = BM25Retriever.from_documents(bm25_docs)
    bm25_retriever.k = 20

    all_results = []

    for q in queries:

        vector_docs = vector_retriever.invoke(q)
        bm25_docs = bm25_retriever.invoke(q)

        combined = vector_docs + bm25_docs
        all_results.append(combined)

    fused_docs = reciprocal_rank_fusion(all_results)
    # print("\nFused documents:")
    # for d in fused_docs[:3]:
    #     print(d.page_content[:200])
    
    final_docs = rerank(query, fused_docs)
    top_docs = final_docs[:5]
    # print("\nTop reranked docs:")
    # for d in top_docs[:3]:
    #     print(d.page_content[:200])

    compressed_docs = compress_context(query, top_docs)
    # print("\nCompressed context:")
    # print(compressed_docs)


    combined_input = f"""
You are an expert AI assistant using Retrieval Augmented Generation (RAG).

Rules:
1. Use ONLY the provided context to answer.
2. If the answer is not present in the context, say:
   "I don't have enough information in the documents."
3. Do NOT use prior knowledge.
4. Give concise answers.
5. If asked to explain or explain in detail then explain whole topic's information from given context

Context:
{chr(10).join(compressed_docs)}

Question:
{query}
"""

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=combined_input),
    ]
    # print("\nFINAL CONTEXT SENT TO MODEL:\n")
    # print(chr(10).join(compressed_docs))    

    result = model.invoke(messages)

    return result.content


