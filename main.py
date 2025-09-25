import csv
import json
import chainlit as cl
import google.generativeai as genai
from transformers import pipeline
"""from huggingface_hub import login
login()(token="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")"""
import torch
from sentence_transformers import SentenceTransformer
import pandas as pd
from datasets import Dataset

library = pd.read_csv('PhilippaBooks.csv')

genai.configure(api_key="AIzaSyCyufIEmJV4cHIH0zpwmgSVmkrS1OlOh1Y")
model = genai.GenerativeModel("gemma-3-27b-it")

dataset = Dataset.from_pandas(library)

# Create a simple RAG (Retrieval-Augmented Generation) setup using the library dataset

# Load a sentence transformer model for embedding
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Compute embeddings for all book titles in the library
library['embedding'] = library['Title'].apply(lambda x: embedder.encode(str(x), convert_to_tensor=True))


def rag_generate(query):
    # Retrieve relevant books
    retrieved_books = library
    # Format retrieved books as context
    context = "\n".join([f"{row['Title']} by {row['Authors']}" for _, row in retrieved_books.iterrows()])
    # Compose prompt for the LLM
    prompt = (
        f"User query: {query}\n"
        f"Relevant books Philippa has read:\n{context}\n"
        "Based on the above, recommend a book or answer the user's request."
        "You are the helpful assistant of Philippa, a librarian. Your job is to recommend books to users that Philippa has read on her behalf."
        "Should the user not provide enough information, suggest a random book from the library of books Philippa has read then ask for more information."
        "You must only recommend books from the library. If the user asks for a book that Philippa has not read and is not in the library, inform them that the book is not available and suggest a random book from the library."
        "The user is not Philippa, so refer to the owner of the collection in third person."
        "Do not assume the user has read any of the books in the library."
        "Provide a description of each book you recommend."
    )
    response = model.generate_content(contents=[prompt])
    return response.text
system_prompt = (
    "You are the helpful assistant of Philippa, a librarian. Your job is to recommend books to users that Philippa has read on her behalf."
    "Should the user not provide enough information, suggest a random book from the library of books Philippa has read then ask for more information."
    "You must only recommend books from the library. If the user asks for a book that Philippa has not read and is not in the library, inform them that the book is not available and suggest a random book from the library."
    "The user is not Philippa, so refer to the owner of the collection in third person."
    "Do not assume the user has read any of the books in the library."
)

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(
        content="Philippa has read a lot of books! Let's see if any of them would be of interest to you!"
    ).send()
@cl.on_message
async def handle_message(message: cl.Message):
    reply = rag_generate(message.content)
    """response = model.generate_content(contents=[message.content])
    reply = response.text"""

    await cl.Message(content=reply).send()