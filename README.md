# Basic RAG (Retrieval-Augmented Generation)

A beginner-friendly **Retrieval-Augmented Generation (RAG)** project built with Python.

This project demonstrates the basic RAG pipeline: loading documents, splitting them into chunks, converting chunks into embeddings, storing/searching vectors, retrieving relevant information, and using the retrieved context to answer questions.

---

## 📌 What is RAG?

**RAG (Retrieval-Augmented Generation)** is a technique that allows an AI application to retrieve relevant information from external documents before generating an answer.

Instead of asking an LLM to answer only from its trained knowledge, RAG provides the model with relevant information from a custom knowledge base.

### Simple RAG Flow

```text
Documents
    ↓
Load Documents
    ↓
Chunking
    ↓
Create Embeddings
    ↓
Store Vectors
    ↓
User Question
    ↓
Convert Question to Embedding
    ↓
Similarity Search
    ↓
Retrieve Relevant Chunks
    ↓
Send Context + Question to LLM
    ↓
Generate Answer
```

---

## 🎯 Why RAG?

RAG is useful when an application needs to answer questions using **custom or private information**.

For example:

* Company policies
* Product documentation
* PDFs
* Technical documentation
* Internal knowledge bases
* Research papers
* Customer support documents

### Problem without RAG

An LLM may not know about your private documents or may provide outdated information.

### With RAG

The application can retrieve relevant information from your own documents and provide that information to the LLM as context.

---

## 🛠️ Technologies Used

* Python
* Sentence Transformers
* Hugging Face
* `all-MiniLM-L6-v2`
* NumPy
* Vector Embeddings
* Cosine Similarity
* RAG concepts

---

## 🤖 Embedding Model

This project uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The model converts text into numerical vectors called **embeddings**.

The embedding size produced by this model is:

```text
384 dimensions
```

For example:

```text
Text
 ↓
Embedding Model
 ↓
[0.021, -0.143, 0.562, ...]
 ↓
384-dimensional vector
```

These vectors allow us to compare the semantic meaning of different pieces of text.

---

## 📂 Project Structure

```text
BASIC_RAG/
│
├── data/
│   └── documents.txt
│
├── embedding.py
│
├── requirements.txt
│
├── README.md
│
└── venv/
```

### `data/`

Contains the documents used as the knowledge source for the RAG system.

Example:

```text
data/
└── company_policy.txt
```

### `embedding.py`

Responsible for:

* Loading documents
* Splitting documents into chunks
* Loading the embedding model
* Generating embeddings
* Checking embedding dimensions

### `requirements.txt`

Contains the Python dependencies required by the project.

### `README.md`

Project documentation.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
```

Move into the project directory:

```bash
cd BASIC_RAG
```

---

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 Requirements

Example `requirements.txt`:

```text
sentence-transformers
numpy
```

Install them manually if required:

```bash
pip install sentence-transformers numpy
```

---

## ▶️ Running the Project

Make sure the virtual environment is activated.

Run:

```bash
python embedding.py
```

The program loads the documents, creates chunks, and generates embeddings.

Example output:

```text
Number of chunks: 6
Embedding shape: (6, 384)
```

This means:

```text
6 chunks
×
384-dimensional embedding
```

were generated.

---

## 🧩 Important RAG Concepts

### 1. Documents

Documents are the original source of information.

Example:

```text
Company Leave Policy

Employees are eligible for...
```

---

### 2. Chunking

Large documents are divided into smaller pieces called **chunks**.

Example:

```text
Large Document
      ↓
 ┌─────────────┐
 │ Chunk 1     │
 ├─────────────┤
 │ Chunk 2     │
 ├─────────────┤
 │ Chunk 3     │
 └─────────────┘
```

Chunking makes it easier to retrieve only the relevant information.

---

### 3. Embeddings

An embedding represents the semantic meaning of text as numbers.

Example:

```text
"Employees can take 20 days of leave"
```

becomes something similar to:

```text
[0.12, -0.45, 0.78, ...]
```

The actual vector contains 384 numbers when using `all-MiniLM-L6-v2`.

---

### 4. Vector Similarity

After converting text into vectors, we can compare vectors to determine how semantically similar two pieces of text are.

A common method is:

```text
Cosine Similarity
```

Conceptually:

```text
Similar meaning
      ↓
Vectors are closer
      ↓
Higher similarity score
```

---

### 5. Similarity Search

When a user asks a question:

```text
"What is the company's leave policy?"
```

the question is converted into an embedding.

The system compares the question embedding against document embeddings and finds the most relevant chunks.

---

### 6. Top-K Retrieval

Instead of retrieving every chunk, the system retrieves the best `K` chunks.

For example:

```text
Top-K = 3
```

means:

```text
Question
   ↓
Similarity Search
   ↓
Chunk 4   ← Most relevant
Chunk 1   ← Second
Chunk 6   ← Third
```

These retrieved chunks are then passed to the LLM.

---

## 🧠 Complete RAG Architecture

```text
                 KNOWLEDGE BASE
                       │
                       ▼
                  Documents
                       │
                       ▼
                    Chunking
                       │
                       ▼
                  Embeddings
                       │
                       ▼
                Vector Database
                       │
                       │
                       │
User Question ────────┘
       │
       ▼
Question Embedding
       │
       ▼
Similarity Search
       │
       ▼
   Top-K Chunks
       │
       ▼
Retrieved Context
       │
       ▼
       ┌─────────────────────┐
       │ Question + Context  │
       └─────────────────────┘
                  │
                  ▼
                 LLM
                  │
                  ▼
             Final Answer
```

---

## 🔍 Current Project Progress

The current version focuses on understanding the fundamental building blocks of RAG.

### Completed

* [x] Python environment setup
* [x] Document storage
* [x] Document chunking
* [x] Sentence Transformer setup
* [x] Embedding generation
* [x] Understanding embedding dimensions
* [x] Understanding vectors
* [x] Understanding semantic similarity

---

## 🚀 Future Improvements

The project will gradually evolve from a basic RAG implementation into a more complete production-style RAG system.

Planned improvements include:

```text
Basic RAG
   ↓
Similarity Search
   ↓
Top-K Retrieval
   ↓
Vector Database
   ↓
LLM Integration
   ↓
Prompt Engineering
   ↓
Metadata Filtering
   ↓
Hybrid Search
   ↓
Reranking
   ↓
Query Rewriting
   ↓
Evaluation
   ↓
Production RAG
```

---

## 📚 What I Am Learning

This project is being built to understand RAG **from the fundamentals rather than only using frameworks**.

Key concepts being explored:

* LLMs
* Tokens
* Context windows
* Prompts
* Embeddings
* Vector representations
* Semantic similarity
* Chunking
* Chunk overlap
* Vector databases
* Similarity search
* Top-K retrieval
* Context injection
* Hallucinations
* Hybrid search
* Reranking
* Query rewriting
* RAG evaluation
* Latency
* Cost
* Security

---

## 💡 Example Use Case

Imagine a company has this document:

```text
Company Leave Policy

Employees receive 20 paid leave days per year.
Employees must request leave through the HR portal.
```

A user asks:

```text
How many paid leave days do employees receive?
```

The RAG system performs:

```text
Question
   ↓
Embedding
   ↓
Similarity Search
   ↓
Find relevant chunk
   ↓
Retrieve:
"Employees receive 20 paid leave days per year."
   ↓
Send context to LLM
   ↓
Answer:
"Employees receive 20 paid leave days per year."
```

The important idea is that the answer is grounded in the retrieved company document.

---

## 🎓 Learning Goal

The main goal of this project is to understand **how RAG works internally**, rather than treating it as a black box.

By completing this project, I aim to understand how to:

1. Prepare documents
2. Split documents into chunks
3. Generate embeddings
4. Store vectors
5. Search for similar information
6. Retrieve relevant context
7. Provide context to an LLM
8. Generate grounded answers
9. Evaluate RAG performance
10. Improve a RAG system for real-world use cases

---

## 👨‍💻 Author

**Mohammas Abbas**

This project is part of my learning journey in:

```text
Python
AI
LLMs
Embeddings
Vector Search
RAG
Backend Development
```

---

## ⭐ Project Status

🚧 **Currently under development**

This project is being developed step-by-step to understand the complete RAG pipeline from fundamentals to a practical implementation.
