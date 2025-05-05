# example_1_embed_store.py
# Embed healthcare texts and store in MongoDB

import voyageai
import numpy as np
from pymongo import MongoClient

# --- Setup ---
VOYAGE_API_KEY = "your-voyageai-api-key"
ATLAS_URI = "your-mongodb-uri"

client = voyageai.Client(api_key=VOYAGE_API_KEY)
mongo_client = MongoClient(ATLAS_URI)
collection = mongo_client["healthcare"]["documents"]

# --- Normalize helper ---
def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

# --- Texts to embed ---
texts = [
    "Patient exhibits symptoms of type 2 diabetes with elevated A1C.",
    "Eligibility criteria for cardiac rehabilitation include a recent MI or CABG.",
    "Flu vaccine recommended for all patients over age 65."
]

# --- Embed and store ---
response = client.embed(texts=texts, model="voyage-2", input_type="document")
for text, emb in zip(texts, response.embeddings):
    doc = {"text": text, "embedding": normalize(emb)}
    collection.insert_one(doc)
print("Documents embedded and stored.")


# example_2_search.py
# Search healthcare texts using VoyageAI + MongoDB Atlas Vector Search

import voyageai
import numpy as np
from pymongo import MongoClient

# --- Setup ---
VOYAGE_API_KEY = "your-voyageai-api-key"
ATLAS_URI = "your-mongodb-uri"

client = voyageai.Client(api_key=VOYAGE_API_KEY)
mongo_client = MongoClient(ATLAS_URI)
collection = mongo_client["healthcare"]["documents"]

# --- Normalize helper ---
def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

# --- Query embedding ---
query = "What are the guidelines for flu shots for seniors?"
query_emb = client.embed(texts=[query], model="voyage-2", input_type="query").embeddings[0]
query_emb = normalize(query_emb)

# --- Vector search ---
results = collection.aggregate([
    {
        "$vectorSearch": {
            "queryVector": query_emb,
            "path": "embedding",
            "numCandidates": 100,
            "limit": 5,
            "index": "healthcare_vector_index"
        }
    },
    {"$project": {"text": 1, "_id": 0}}
])

print("Search results:")
for doc in results:
    print("-", doc["text"])


# example_3_rerank.py
# Rerank candidate texts with VoyageAI

import voyageai

client = voyageai.Client(api_key="your-voyageai-api-key")

query = "What are the guidelines for flu shots for seniors?"
documents = [
    "Flu vaccine recommended for all patients over age 65.",
    "Cardiac rehab includes exercise and counseling.",
    "Diabetes treatment involves diet and exercise."
]

response = client.rerank(query=query, documents=documents, model="voyage-2")
print("Top result:")
print(response.results[0].document)
