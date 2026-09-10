import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from ollama import chat

chunks = [
    "Company Leave Policy",
    "Employees receive 20 days of annual leave per year.",
    "Employees receive 10 days of sick leave per year.",
    "Employees should request annual leave at least 3 days before the planned leave.",
    "Company working hours are from 9 AM to 6 PM.",
    "Employees can work remotely two days per week."
]

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(chunks)

embeddings = np.array(embeddings).astype("float32")

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

query = input("Ask a question: ")

query_embedding = model.encode([query])

query_embedding = np.array(query_embedding).astype("float32")

k = 2

distances, indices = index.search(query_embedding, k)

retrieved_chunks = []

for i in range(k):
    retrieved_chunks.append(chunks[indices[0][i]])

print("Retrieved context:")

for chunk in retrieved_chunks:
    print("-", chunk)


# Combine retrieved chunks
context = "\n".join(retrieved_chunks)

# Create prompt
prompt = f"""
Answer the question using only the provided context.

Context:
{context}

Question:
{query}
"""

# Send context + question to LLM
response = chat(
    model="llama3.2:3b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)
# Display final answer
print("\nFinal answer:")
print(response.message.content)