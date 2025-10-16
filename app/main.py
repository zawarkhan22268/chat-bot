from fastapi import FastAPI, HTTPException,UploadFile,File,Form,Request
from fastapi.responses import PlainTextResponse,StreamingResponse,HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from pathlib import Path
from app.utilities import file_format, clean_text
from app.core.vector import add_document_chunks
from app.core.llm import generate_plain_stream, initialize_llm, response_cache, save_cache

# ------------------- Router Setup --------------------
app = FastAPI(
    title="ChatBot_Dexterz_SOL",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------- Startup Event --------------------
@app.on_event("startup")
async def startup_event():
    """
    Pre-load LLM client at server startup to reduce first request latency.
    """
    print("🚀 Pre-loading models...")
    await initialize_llm()
    print("✅ Server ready!")


#--------------------------------Routes-----------------------------------

# === Upload file ===
# The route to make the embeddings of the data of the website to be used for the chatbot
@app.post("/upload_file", response_class=PlainTextResponse)
async def upload_regulations(
    file: UploadFile = File(..., description="The File to be uploaded"),
):
    # Extract plain text from file
    text = await asyncio.to_thread(file_format, file)

    if not text or len(text.strip()) < 10:
        raise HTTPException(status_code=400, detail="Document is empty or too short.")

    # Clean text before processing
    print("Original text length:", len(text))
    cleaned_text = clean_text(text)
    print("Cleaned text length:", len(cleaned_text))
    print("--------------------------------------------------")
    
    try:
        # Store processed text into vector database
        await add_document_chunks(cleaned_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding data to collection: {e}")

    return PlainTextResponse(content=f"Document uploaded and chunked successfully.")


# === Serve the chatbot widget interface ===
@app.get("/", response_class=HTMLResponse)
async def serve_chatbot():
    html_path = Path(__file__).parent.parent / "frontend" / "chatbot_widget.html"
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())



# === Streaming chat (Plain text format) ===
@app.post('/stream_chat')
async def stream_chat(req: Request):
   
    body = await req.json()
    user_message = body.get('message', '')
    
    if not user_message:
        raise HTTPException(status_code=400, detail="Message field is required")
    
    async def event_stream():
        """Stream plain text chunks directly to client"""
        try:
            async for chunk in generate_plain_stream(user_message):
                # Yield each token/character as plain text
                yield chunk
                # print(chunk)  if to display the chunks recieved 
                # Minimal delay to ensure proper flushing (no artificial delay)
                await asyncio.sleep(0)
        except Exception as e:
            print(f" Error in stream_chat: {e}")
            yield f"\n[Error: {str(e)}]"
    
    headers = {
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no',
        'Content-Type': 'text/plain; charset=utf-8'
    }
    
    return StreamingResponse(event_stream(), headers=headers, media_type='text/plain')


# === Cache management endpoints ===
@app.get('/cache/stats')
async def get_cache_stats():
    """
    Get cache statistics.
    """
    return {
        "cache_size": len(response_cache),
        "max_cache_size": 40,
        "cached_queries": len(response_cache)
    }


@app.post('/cache/clear')
async def clear_cache():
    """
    Clear the response cache and save to file.
    """
    response_cache.clear()
    save_cache()  # Save empty cache to file
    return {"message": "Cache cleared successfully", "cache_size": 0}
