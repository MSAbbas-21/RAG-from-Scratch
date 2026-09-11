import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from load_document import load_document
from ollama import chat


# -----------------------------
# 1. Load embedding model
# -----------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# 2. Load PDF and create chunks
# -----------------------------

chunks = load_document("data/nimbus_retail_employee_policy.pdf")

print("Number of chunks:", len(chunks))


# -----------------------------
# 3. Convert chunks into vectors
# -----------------------------

embeddings = model.encode(chunks)

embeddings = np.array(embeddings).astype("float32")


# -----------------------------
# 4. Create FAISS index
# -----------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("Number of vectors stored:", index.ntotal)


# -----------------------------
# 5. Get user's question
# -----------------------------

query = input("\nAsk your Question: ")


# -----------------------------
# 6. Convert question into vector
# -----------------------------

query_embedding = model.encode([query])

query_embedding = np.array(query_embedding).astype("float32")


# -----------------------------
# 7. Retrieve Top-K chunks
# -----------------------------

k = 2

distances, indices = index.search(query_embedding, k)


# -----------------------------
# 8. Build context
# -----------------------------

retrieved_chunks = []

for i in range(k):

    chunk = chunks[indices[0][i]]

    retrieved_chunks.append(chunk)

    print(f"\nResult {i + 1}")
    print("Chunk:", chunk)
    print("Distance:", distances[0][i])


context = "\n\n".join(retrieved_chunks)


# -----------------------------
# 9. Send context + question
#    to Ollama
# -----------------------------

prompt = f"""
You are a helpful assistant answering questions about the provided document.

Answer the user's question ONLY using the information in the context below.

If the answer is not present in the context, say:
"I don't know based on the provided document."

Do not make up information.

Context:
{context}

Question:
{query}
"""


response = chat(
    model="llama3.2:3b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)


# -----------------------------
# 10. Display grounded answer
# -----------------------------

print("\n--- Grounded Answer ---")
print(response.message.content)