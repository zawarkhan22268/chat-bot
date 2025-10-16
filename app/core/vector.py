import uuid
import asyncio
import chromadb
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from app.utilities import clean_text

# Load API key from .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI embedding model
embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=api_key
)

# Persistent Chroma client
new_client = chromadb.PersistentClient(r"D:\chatbot\website_embeddings")
new_collection = new_client.get_or_create_collection(name="new_chatbot")


# -------------------- Chunking --------------------
def chunk_text(text, chunk_size=200, overlap=25):
    """
    Split large text into overlapping chunks.
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


# -------------------- Ingestion --------------------
async def add_document_chunks(text: str):
    """
    Chunk text, create OpenAI embeddings, and upsert into ChromaDB.
    """
    # Clean text using the centralized function
    text = clean_text(text)

    # Create text chunks
    chunks = chunk_text(text)

    if not chunks:
        raise RuntimeError("No chunks generated from text!")

    try:
        print(f"Generating embeddings for {len(chunks)} chunks...")

        # Generate embeddings for all chunks (batch)
        vectors = await asyncio.to_thread(
            embeddings_model.embed_documents, chunks
        )

        # Generate unique IDs
        ids = [str(uuid.uuid4()) for _ in range(len(chunks))]

        # Upsert into ChromaDB
        await asyncio.to_thread(
            new_collection.upsert,
            embeddings=vectors,
            documents=chunks,
            ids=ids
        )

        print(f"✅ Successfully stored {len(chunks)} chunks in ChromaDB.")

    except Exception as e:
        raise RuntimeError(f"Error adding chunked documents to ChromaDB: {e}")
