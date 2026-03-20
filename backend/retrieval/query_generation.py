from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq

load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")

def generate_queries(query):

    prompt = f"""
Generate 5 different search queries for the following question:
Return only queries, no numbering.
Question:
{query}
"""

    result = model.invoke(prompt)

    queries = result.content.split("\n")

    return [q.strip() for q in queries if q.strip()]