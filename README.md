# VoyageAI-PythonClient

Python client for embedding and querying documents using VoyageAI and MongoDB Vector Search.
Demos assume paid 🔑 you can grab one here -> https://www.voyageai.com/ <br>
Untested on Antikythera Mechanism ⚙️

🧠 Basics: Setup & Usage
🔹 Install
```
pip install voyageai
```
🔹 Authentication
```
import voyageai
client = voyageai.Client(api_key="your-voyageai-api-key")
```
🔍 Embedding Tips
🔹 Create Embeddings
```
response = client.embed(
    texts=["What are the storage options?", "How does multi-region replication work?"],
    model="voyage-2",  # or "voyage-lite-02-instruct"
    input_type="query"  # "query" or "document"
)
embeddings = response.embeddings
```
✅ Tip: Use "query" for search queries and "document" for corpus data — they are trained to embed differently!

📏 Performance & Size Tricks

voyage-lite-02-instruct is faster and cheaper (~150ms latency), and ideal for large-scale ingestion.
voyage-2 is more accurate and better for reranking or production search.

Batch embeddings — the API supports bulk input for embedding up to 96 texts at once for voyage-2.

📦 Integrate with MongoDB Vector Search
```
# Insert into MongoDB with PyMongo
doc = {
    "text": "What are the storage options?",
    "embedding": embeddings[0]
}
collection.insert_one(doc)
```
✅ Tip: Store both "text" and "embedding" fields. Use MongoDB’s $vectorSearch for efficient hybrid retrieval.

⚡ Hybrid Search Pattern

Use MongoDB Search for keyword/text filters (e.g., "project_id": 123).
Use VoyageAI embeddings to do vector similarity via $vectorSearch.

Optionally use VoyageAI reranker for final reranking:
```
response = client.rerank(
    query="What are the storage options?",
    documents=[
        "Blob storage in Azure Iowa",
        "Online Archive is not available in Iowa"
    ],
    model="voyage-2"
)
```
🧪 Debugging & Quality Checks
Log response.usage to track token counts and cost:
```
print(response.usage)
```
Monitor vector norm and outliers:
```
import numpy as np
print(np.linalg.norm(embeddings[0]))
```
Ensure your vectors are normalized (MongoDB doesn’t do this for you):
```
def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v
```
