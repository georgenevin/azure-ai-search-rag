# Azure AI Search — Simple Explanation

## 1. What is Azure AI Search?

Azure AI Search can be thought of as a **smart search engine for your documents**.

For example, suppose you have a 500-page PDF. Instead of sending the entire PDF to an LLM every time a user asks a question, Azure AI Search finds the most relevant parts of the document.

The basic idea is:

```text
PDF
 ↓
Extract text
 ↓
Split into chunks
 ↓
Create embeddings
 ↓
Store chunks + embeddings in Azure AI Search
 ↓
User asks a question
 ↓
Search for relevant chunks
 ↓
Send relevant chunks to LLM
 ↓
LLM generates answer
```

This is the basic architecture of **RAG (Retrieval-Augmented Generation)**.

---

# 2. Why do we need Azure AI Search?

Suppose we have a large PDF:

```text
ncert-science.pdf
```

If we send the entire PDF to an LLM for every question, we may have:

- Large context
- Higher cost
- Slower responses
- Context-window limitations
- A lot of irrelevant information

Instead, Azure AI Search retrieves only the relevant information.

```text
Large PDF
   ↓
Azure AI Search
   ↓
Relevant chunks
   ↓
LLM
   ↓
Answer
```

---

# 3. What is an Embedding?

An embedding converts text into a list of numbers called a **vector**.

For example:

```text
"Plants use sunlight to produce food through photosynthesis."
```

can be converted into something like:

```text
[0.12, -0.34, 0.72, 0.41, ...]
```

The numbers themselves are not meaningful to us.

The important idea is:

> Text with similar meaning produces similar vectors.

For example:

```text
"Plants produce food using sunlight"
```

and

```text
"How do plants make their food?"
```

have different words but similar meanings.

Their embeddings will therefore be relatively close to each other.

---

# 4. What does the PDF processing code do?

Your `get_pdf_data()` function extracts text from the PDF.

```python
reader = PdfReader(file_path)
```

Then:

```python
current_page.extract_text()
```

extracts text from each page.

The result is one large string containing the document text.

```text
PDF
 ↓
Page 1
Page 2
Page 3
...
 ↓
Full document text
```

---

# 5. Why do we split the document into chunks?

A large document should not normally be embedded as one huge piece.

Your code splits the document into smaller chunks:

```python
get_chunks(full_doc_text, 500)
```

Conceptually:

```text
Large document
       ↓
┌──────────────┐
│ Chunk 1      │
├──────────────┤
│ Chunk 2      │
├──────────────┤
│ Chunk 3      │
├──────────────┤
│ Chunk 4      │
└──────────────┘
```

Each chunk can then be embedded and searched independently.

---

# 6. Creating Embeddings

Your code uses the Azure OpenAI embedding model:

```python
client.embeddings.create(
    input=[text],
    model=model
)
```

For every chunk:

```text
Chunk
 ↓
Embedding Model
 ↓
Vector
```

For example:

```text
Chunk:
"Photosynthesis is the process by which plants make food."

        ↓

Embedding:

[0.12, 0.43, -0.21, 0.76, ...]
```

The vector is then stored along with the original text.

---

# 7. What is an Azure AI Search Index?

An **index** is the searchable structure where Azure AI Search stores your document data.

Your index contains fields such as:

```text
Azure AI Search Index
│
├── id
├── line
├── filename
└── embedding
```

Your code defines these fields:

```python
fields = [
    SimpleField(name="id", ...),

    SearchableField(name="line", ...),

    SearchableField(name="filename", ...),

    SearchField(
        name="embedding",
        ...
    )
]
```

You can think of an Azure AI Search index as a **database table optimized for search**.

---

# 8. What is the Vector Field?

Your code contains:

```python
SearchField(
    name="embedding",
    type=SearchFieldDataType.Collection(
        SearchFieldDataType.Single
    ),
    searchable=True,
    vector_search_dimensions=3072,
    vector_search_profile_name="myHnswProfile"
)
```

This tells Azure AI Search:

> The `embedding` field contains vectors and can be used for vector search.

The value:

```python
vector_search_dimensions=3072
```

means your embedding vectors have 3072 dimensions.

The vector dimensions must match the dimensions produced by the embedding model.

---

# 9. What is HNSW?

Your code configures:

```python
HnswAlgorithmConfiguration(...)
```

HNSW is an algorithm used to efficiently find similar vectors.

Imagine you have:

```text
1,000,000 document chunks
```

and the user asks:

> "How do plants produce food?"

Azure AI Search needs to find the chunks whose meaning is closest to the question.

HNSW provides an efficient way to search through a large number of vectors.

A simple way to remember it:

> **HNSW = an efficient algorithm for finding similar vectors.**

---

# 10. What happens when a user asks a question?

Suppose the user asks:

```text
What is the main topic of the document?
```

Your code first creates an embedding for the question:

```python
quer_vector = generate_embeddings(
    query,
    model,
    client
)
```

The flow is:

```text
User Question
      ↓
Embedding Model
      ↓
Question Vector
```

For example:

```text
"What is the main topic?"

        ↓

[0.11, 0.42, -0.20, ...]
```

---

# 11. Vector Search

Your code then sends the question vector to Azure AI Search:

```python
results = search_client.search(
    search_text=query,
    vector_queries=[
        VectorizedQuery(
            vector=quer_vector,
            fields="embedding"
        )
    ],
    top=3
)
```

This essentially means:

> Find the top 3 document chunks whose vectors are most similar to the question vector.

For example:

```text
Question
   ↓
Question Vector
   ↓
Azure AI Search
   ↓
Top 3 similar chunks
```

The result might be:

```text
Chunk 17
Chunk 32
Chunk 45
```

These chunks become the **retrieved context**.

---

# 12. Keyword Search vs Vector Search

There are two important concepts.

## Keyword Search

Keyword search looks for matching words.

For example:

```text
Query:
"photosynthesis"
```

It can find documents containing:

```text
"Photosynthesis is..."
```

---

## Vector Search

Vector search looks at **semantic similarity**.

For example:

```text
Query:
"How do plants make their food?"
```

It can find:

```text
"Plants use sunlight to produce food through photosynthesis."
```

The exact words are different, but the meanings are similar.

---

# 13. What is Hybrid Search?

Your code contains both:

```python
search_text=query
```

and:

```python
vector_queries=[
    VectorizedQuery(...)
]
```

This allows you to combine traditional text search with vector search.

Conceptually:

```text
Keyword Search
      +
Vector Search
      ↓
Hybrid Search
```

Keyword search is good at finding exact terms.

Vector search is good at finding similar meanings.

Combining them can improve retrieval quality.

---

# 14. What is Semantic Search?

Your code also defines:

```python
semantic_config = SemanticConfiguration(
    name="my-semantic-config",
    ...
)
```

Semantic search is another search capability that focuses more on understanding the meaning and language context of search results.

A simple way to distinguish the concepts:

```text
Keyword Search
    ↓
Exact words

Vector Search
    ↓
Semantic similarity using embeddings

Semantic Search
    ↓
Language-aware search/ranking
```

Azure AI Search can support these different search capabilities.

---

# 15. Azure AI Search Does Not Generate the Final Answer

This is one of the most important concepts.

Azure AI Search is mainly responsible for:

```text
Finding relevant information
```

Azure OpenAI is responsible for:

```text
Generating the answer
```

So:

```text
                  ┌─────────────────────┐
Question ────────►│   Azure AI Search   │
                  │                     │
                  │ Find relevant       │
                  │ document chunks     │
                  └──────────┬──────────┘
                             │
                             │ Context
                             ▼
                  ┌─────────────────────┐
                  │     Azure OpenAI    │
                  │                     │
                  │ Generate answer     │
                  └──────────┬──────────┘
                             │
                             ▼
                           Answer
```

---

# 16. What does the `context` variable do?

Your code has:

```python
context = "\n".join(
    [doc["line"] for doc in result]
)
```

Suppose Azure AI Search returns:

```text
Chunk 1:
Science is the systematic study of the natural world.

Chunk 2:
Physics, chemistry and biology are major branches of science.

Chunk 3:
Science helps us understand natural phenomena.
```

The code combines them:

```text
Context:
Science is the systematic study of the natural world.
Physics, chemistry and biology are major branches of science.
Science helps us understand natural phenomena.
```

This becomes the context given to Azure OpenAI.

---

# 17. How Azure OpenAI uses the Context

Your prompt tells the model:

```text
Answer the question based only on the provided context.
If the answer isn't in the context, say you don't know.
```

Then you provide:

```text
Context:
<retrieved chunks>

Question:
<user question>
```

The LLM uses the retrieved chunks to generate the final answer.

This helps reduce the chance of the model answering from unrelated knowledge.

---

# 18. Complete RAG Flow

Your complete application can be understood in two major phases.

## Phase 1 — Indexing

This happens when you add documents.

```text
PDF
 ↓
Extract text
 ↓
Split into chunks
 ↓
Generate embeddings
 ↓
Create Azure AI Search index
 ↓
Upload chunks + embeddings
```

This is called **indexing**.

---

## Phase 2 — Querying

This happens when the user asks a question.

```text
User Question
 ↓
Generate question embedding
 ↓
Azure AI Search
 ↓
Find relevant chunks
 ↓
Build context
 ↓
Send context + question to Azure OpenAI
 ↓
Generate answer
```

This is the **retrieval + generation** part of RAG.

---

# 19. Simple Analogy

Imagine a 1,000-page textbook and a smart librarian.

You ask:

> "What is photosynthesis?"

The librarian does not give you the entire textbook.

Instead, the librarian finds:

```text
Page 124 → Photosynthesis definition
Page 125 → Photosynthesis process
Page 126 → Photosynthesis explanation
```

You give those pages to GPT.

GPT reads them and generates the answer.

Therefore:

```text
Azure AI Search
        =
Smart Librarian
```

and:

```text
Azure OpenAI
        =
Person who reads the selected pages
and explains the answer
```

---

# 20. Key Concepts to Remember

| Concept | Simple Meaning |
|---|---|
| Document | Your PDF or other source |
| Chunk | Small piece of a document |
| Embedding | Text converted into numbers |
| Vector | The numerical representation of text |
| Index | Searchable structure containing your data |
| Vector Search | Search based on semantic similarity |
| Keyword Search | Search based on words |
| Hybrid Search | Combination of keyword + vector search |
| HNSW | Efficient vector similarity search algorithm |
| Semantic Search | Language-aware search capability |
| Retrieval | Finding relevant document chunks |
| Generation | LLM creating the final answer |
| RAG | Retrieval + Generation |

---

# 21. One-Line Explanation

The easiest way to remember the entire architecture is:

> **Azure AI Search finds the relevant information, and Azure OpenAI uses that information to generate the answer.**

```text
                RAG

PDF
 ↓
Chunks
 ↓
Embeddings
 ↓
Azure AI Search
 ↓
Relevant Context
 ↓
Azure OpenAI
 ↓
Final Answer
```

---

# 22. Your Code in One Sentence

Your Python application:

> **Reads an NCERT PDF, splits it into chunks, creates embeddings for those chunks, stores them in Azure AI Search, retrieves the most relevant chunks when a question is asked, and sends those chunks to Azure OpenAI to generate the final answer.**