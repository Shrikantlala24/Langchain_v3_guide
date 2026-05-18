from langchain_huggingface import HuggingFaceEmbeddings

emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

doc = [
    'what is deep learning?',
    'what is machine learning?',
    'what is artificial intelligence?'
]

def save_vector():
    with open("vector.txt", "w") as f:
        f.write(str(vec))

try:
    vec = emb.embed_documents(doc)
    save_vector()


except Exception as e:
    print(f"✗ Error: {e}")