from dotenv import load_dotenv
import os
from openai import OpenAI
import chromadb
import asyncio
from app.prompts import SYSTEM_PROMPT
from langchain_openai import OpenAIEmbeddings
import hashlib
from collections import OrderedDict
import json
from pathlib import Path

from app.core.vector import new_client, new_collection


# ---------------- Setup ----------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
llm_client = None

# Response cache - stores last 40 queries
response_cache = OrderedDict()
MAX_CACHE_SIZE = 40
CACHE_FILE = Path(__file__).parent.parent.parent / "cache" / "response_cache.json"

# Load cache from file on startup
def load_cache():
    """Load cache from file if it exists"""
    global response_cache
    try:
        if CACHE_FILE.exists():
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                response_cache = OrderedDict(data)
                print(f"💾 Loaded {len(response_cache)} cached responses from file")
        else:
            print("📝 No cache file found, starting with empty cache")
    except Exception as e:
        print(f"⚠️ Error loading cache: {e}")
        response_cache = OrderedDict()

def save_cache():
    """Save cache to file"""
    try:
        if len(response_cache) > MAX_CACHE_SIZE:
            pass
        # Create cache directory if it doesn't exist
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            
        # Save to file
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(dict(response_cache), f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ Error saving cache: {e}")

# Load cache on module import (when server starts)
load_cache()

def get_llm_client():
    global llm_client
    if llm_client is None and api_key:
        try:
            llm_client = OpenAI(api_key=api_key)
        except Exception as e:
            print(f"Failed to initialize OpenAI client: {e}")
            llm_client = None
    return llm_client

# Initialize OpenAI embedding model (same as vector.py)
embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=api_key
)

# Persistent ChromaDB client
new_client = chromadb.PersistentClient(r"D:\chatbot\website_embeddings")

# Create or load collection
new_collection = new_client.get_or_create_collection(name="new_chatbot")

# Preload function for server startup
async def initialize_llm():
    """
    Pre-load LLM client at server startup to warm up the connection.
    """
    client = get_llm_client()
    if client:
        print("✅ LLM client pre-loaded and ready")




# -------------------------------------------------
# Query Function
# -------------------------------------------------
async def query_func(query: str):
    try:
       
        query_embedding = await asyncio.to_thread(
            embeddings_model.embed_query, query
        )
        
        # Query ChromaDB using the embedding 
        results = await asyncio.to_thread(
            lambda: new_collection.query(
                query_embeddings=[query_embedding],
                n_results=3,  # Total results returned
            )
        )
        return results
    except chromadb.errors.ChromaError as ce:
        raise RuntimeError(f"ChromaDB Error: {ce}")
    except Exception as e:
        raise RuntimeError(f"Unexpected Error: {e}.")    




def get_cache_key(query: str) -> str:
    """Generate cache key from query (case-insensitive)"""
    normalized_query = query.lower().strip()
    return hashlib.md5(normalized_query.encode()).hexdigest()


# === Plain text streaming ===
async def generate_plain_stream(query: str):
   
    import time
    start_time = time.time()
    
    # Check cache first
    cache_key = get_cache_key(query)
    if cache_key in response_cache:
        print(f"💾 Cache hit! Returning cached response for query")
        cached_response = response_cache[cache_key]
        
        # Stream cached response in 3-character chunks
        chunk_size = 3
        for i in range(0, len(cached_response), chunk_size):
            yield cached_response[i:i+chunk_size]
            await asyncio.sleep(0)  # Allow other tasks to run
        
        print(f"⚡ Cached response streamed in {time.time() - start_time:.2f}s")
        return
    
    # Step 1: Search in ChromaDB for relevant context (OPTIMIZED)
    results = await query_func(query)
    print(f"⏱️ Vector search took: {time.time() - start_time:.2f}s")
    

    documents = results.get('documents', [[]])[0] if results.get('documents') else []
    
    #  REDUCED for faster response
    context = "\n\n".join(documents[:3])  
    
    # (max ~1000 chars for faster processing)
    if len(context) > 1000:
        context = context[:1000] + "..."

    # Step 2: Build full prompt
    full_prompt = f"""
     User Query: {query}

     Relevant Context:
     {context}

     Please respond based only on the above context.
     """

    # Step 3: Stream response from OpenAI
    client = get_llm_client()
    if client is None:
        yield "OpenAI client not initialized. Please set OPENAI_API_KEY environment variable."
        return
    
    print(f"⏱️ Starting OpenAI stream at: {time.time() - start_time:.2f}s")
        
    stream = client.chat.completions.create(
        model="gpt-4.1-nano-2025-04-14",  # Fast and efficient model
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": full_prompt},
        ],
        temperature=0.2,
        max_tokens=500,
        stream=True,
        stream_options={"include_usage": False} 
    )

    # Step 4: Yield tokens as plain text in 3-character chunks
    first_token = True
    token_count = 0
    buffer = ""
    full_response = ""  # Store complete response for caching
    chunk_size = 8 # Send 3 characters at a time
    
    try:
        for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    if first_token:
                        print(f"⏱️ First token received at: {time.time() - start_time:.2f}s")
                        first_token = False
                    
                    content = delta.content
                    token_count += 1
                    
                    # Add to buffer and full response
                    buffer += content
                    full_response += content
                    
                    # Yield in chunks of 3 characters
                    while len(buffer) >= chunk_size:
                        yield buffer[:chunk_size]
                        buffer = buffer[chunk_size:]
        
        # Yield any remaining characters in buffer
        if buffer:
            yield buffer
        
        print(f"✅ Streamed {token_count} tokens successfully (plain text)")
        
        # Cache the complete response
        if full_response:
            response_cache[cache_key] = full_response
            # Limit cache size (LRU - remove oldest)
            if len(response_cache) > MAX_CACHE_SIZE:
               print(f"💾 Response cached (cache size: {len(response_cache)})")
               
            else:# Save cache to file
              save_cache()
    except Exception as e:
        print(f"❌ Error during plain text streaming: {e}")
        yield f"\n[Error: {str(e)}]"