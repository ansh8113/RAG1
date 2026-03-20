import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")
def rerank(query, docs):

    docs = docs[:10]

    context = "\n\n".join(
        [f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)]
    )

    prompt = f"""
Query:
{query}

Documents:
{context}

Rank the documents from most relevant to least relevant.
Return only the document numbers in order.
Example: 3,1,5,2
"""

    result = model.invoke(prompt)

    order = result.content.strip().replace(" ", "").split(",")

    ranked_docs = []

    for idx in order:
        try:
            ranked_docs.append(docs[int(idx)-1])
        except:
            pass

    return ranked_docs