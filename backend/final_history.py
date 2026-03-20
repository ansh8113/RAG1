from dotenv import load_dotenv
import os
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from backend.retrieval.answer_generation import generate_answer
# import logging
# logging.getLogger("transformers").setLevel(logging.ERROR)

# Load environment variables
load_dotenv()

# Connect to your document database
persistent_directory = "db/chroma_db"
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Set up AI model
model = ChatGroq(model="llama-3.3-70b-versatile")

# Store our conversation as messages
chat_history = []

def ask_question(user_question):
    print(f"\n--- You asked: {user_question} ---")
    
    # Step 1: Make the question using conversation history
    if chat_history:
        # Ask AI to make the question standalone
        messages = [SystemMessage(content="""
You are a query rewriting assistant for a retrieval system.

Your task is to convert follow-up questions into standalone search queries.

Instructions:
- Use the conversation history to understand the topic.
- Always include the main subject from the earlier conversation.
- Do NOT remove important keywords.
- Do NOT answer the question.
- Only return the rewritten search query.
"""),
    HumanMessage(content=f"""
Conversation history:
{chat_history}

Follow-up question:
{user_question}

Rewrite the question as a standalone search query.
""")
]
        
        result = model.invoke(messages)
        search_question = result.content.strip()
        print(f"Searching for: {search_question}")
    else:
        search_question = user_question
    
    answer = generate_answer(search_question)
    
    # Step 5: Remember this conversation
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))
    
    print(f"Answer: {answer}")
    return answer

# Simple chat loop
def start_chat():
    print("Ask me questions! Type 'quit' to exit.")
    
    while True:
        question = input("\nYour question: ")
        
        if question.lower() == 'quit':
            print("Goodbye!")
            break
       
            
        ask_question(question)

if __name__ == "__main__":
    start_chat()