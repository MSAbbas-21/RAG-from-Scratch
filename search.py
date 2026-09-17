import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from load_document import load_document
from ollama import chat


# -----------------------------
# 1. Employee function/tool
# -----------------------------

def get_employee_leave(employee_id: str) -> int:
    """
    Get the remaining annual leave balance for an employee.

    Args:
        employee_id: The employee ID.

    Returns:
        The number of remaining leave days.
    """

    leave_balance = {
        "101": 15,
        "102": 10,
        "103": 20
    }

    return leave_balance.get(employee_id, 0)


# -----------------------------
# 2. Load embedding model
# -----------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# 3. Load PDF and create chunks
# -----------------------------

chunks = load_document(
    "data/nimbus_retail_employee_policy.pdf"
)

print("Number of chunks:", len(chunks))


# -----------------------------
# 4. Convert chunks into vectors
# -----------------------------

embeddings = model.encode(chunks)

embeddings = np.array(
    embeddings
).astype("float32")


# -----------------------------
# 5. Create FAISS index
# -----------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print(
    "Number of vectors stored:",
    index.ntotal
)


# -----------------------------
# 6. Get user's question
# -----------------------------

query = input("\nAsk your Question: ")


# -----------------------------
# 7. Convert question into vector
# -----------------------------

query_embedding = model.encode([query])

query_embedding = np.array(
    query_embedding
).astype("float32")


# -----------------------------
# 8. Retrieve Top-K chunks
# -----------------------------

k = 2

distances, indices = index.search(
    query_embedding,
    k
)


# -----------------------------
# 9. Build context
# -----------------------------

retrieved_chunks = []

for i in range(k):

    chunk = chunks[indices[0][i]]

    retrieved_chunks.append(chunk)

    print(f"\nResult {i + 1}")
    print("Chunk:", chunk)
    print("Distance:", distances[0][i])


context = "\n\n".join(
    retrieved_chunks
)


# -----------------------------
# 10. Build prompt
# -----------------------------

prompt = f"""
You are a helpful employee assistant.

You can answer questions using the provided
document context.

You also have access to a tool called
get_employee_leave.

Use get_employee_leave when the user asks
about an employee's remaining leave balance.

For questions about company policy, use
the document context.

Do not make up information.

Context:
{context}

Question:
{query}
"""


# -----------------------------
# 11. Create messages
# -----------------------------

messages = [
    {
        "role": "user",
        "content": prompt
    }
]


# -----------------------------
# 12. Ask LLM
# -----------------------------

response = chat(
    model="llama3.2:3b",
    messages=messages,
    tools=[get_employee_leave]
)


# -----------------------------
# 13. Check for tool calls
# -----------------------------

if response.message.tool_calls:

    # Add LLM's tool-call message
    messages.append(response.message)

    for tool in response.message.tool_calls:

        if tool.function.name == "get_employee_leave":

            print("\n--- Tool Calling ---")

            print(
                "Tool:",
                tool.function.name
            )

            print(
                "Arguments:",
                tool.function.arguments
            )

            # Execute Python function
            result = get_employee_leave(
                **tool.function.arguments
            )

            print(
                "Tool result:",
                result
            )

            # Send result back to LLM
            messages.append(
                {
                    "role": "tool",
                    "content": str(result),
                    "tool_name": tool.function.name
                }
            )


    # -----------------------------
    # 14. Ask LLM for final answer
    # -----------------------------

    final_response = chat(
        model="llama3.2:3b",
        messages=messages,
        tools=[get_employee_leave]
    )

    print("\n--- Final Answer ---")

    print(
        final_response.message.content
    )


else:

    # No tool needed
    print("\n--- Grounded Answer ---")

    print(
        response.message.content
    )