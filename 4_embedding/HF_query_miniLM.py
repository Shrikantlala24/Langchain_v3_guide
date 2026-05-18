from langchain_huggingface import HuggingFaceEmbeddings
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vec = emb.embed_query("test")


try:
    print(f"✓ Success! Embedding shape: {len(vec)}")
except Exception as e:
    print(f"✗ Error: {e}")