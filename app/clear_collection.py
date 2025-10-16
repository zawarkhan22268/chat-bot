"""Simple script to delete the old ChromaDB collection"""
import chromadb

# Connect to ChromaDB
client = chromadb.PersistentClient(r"D:\chatbot\website_embeddings")

try:
    # Delete the old collection
    print("🗑️  Deleting old collection...")
    client.delete_collection(name="new_chatbot")
    print("✅ Collection 'new_chatbot' deleted successfully!")
except Exception as e:
    print(f"⚠️  Error: {e}")

# Create new empty collection
print("\n📦 Creating new empty collection...")
new_collection = client.create_collection(name="new_chatbot")
print("✅ New collection created!")

print("\n" + "="*60)
print("DONE! Now re-upload your documents via /upload_file")
print("="*60)
