import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Our document chunks
chunks = [
    "Company Leave Policy",
    "Employees receive 20 days of annual leave per year.",
    "Employees receive 10 days of sick leave per year.",
    "Employees should request annual leave at least 3 days before the planned leave.",
    "Company working hours are from 9 AM to 6 PM.",
    "Employees can work remotely two days per week."
]

# Convert chunks into vectors
embeddings = model.encode(chunks)

# Convert to FAISS-compatible format
embeddings = np.array(embeddings).astype("float32")

# Get vector size
dimension = embeddings.shape[1]

# Create FAISS index
index = faiss.IndexFlatL2(dimension)

# Store our vectors in FAISS
index.add(embeddings)

print("Number of vectors stored:", index.ntotal)

# User's question
query = "Can employees work from home?"

# Convert the question into an embedding
query_embedding = model.encode([query])

# Convert to FAISS-compatible format
query_embedding = np.array(query_embedding).astype("float32")

# Search for the 2 most similar chunks
k = 2

distances, indices = index.search(query_embedding, k)

# Display results
for i in range(k):
    print(f"\nResult {i + 1}")
    print("Chunk:", chunks[indices[0][i]])
    print("Distance:", distances[0][i])