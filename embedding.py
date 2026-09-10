from sentence_transformers import SentenceTransformer

# Load the embedding model
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

# Convert chunks into embeddings
embeddings = model.encode(chunks)

print("Number of chunks:", len(chunks))
print("Embedding shape:", embeddings.shape)

print("\nFirst chunk:")
print(chunks[0])

print("\nFirst chunk's vector:")
print(embeddings[0])