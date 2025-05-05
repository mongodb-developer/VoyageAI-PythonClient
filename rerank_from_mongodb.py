# Pull vector search results from Atlas, rerank with Voyage

query = "What are the diabetes care guidelines?"
query_emb = normalize(client.embed(texts=[query], model="voyage-2", input_type="query").embeddings[0])

cursor = collection.aggregate([
    {
        "$vectorSearch": {
            "queryVector": query_emb,
            "path": "embedding",
            "numCandidates": 100,
            "limit": 10,
            "index": "healthcare_vector_index"
        }
    },
    {"$project": {"text": 1, "_id": 0}}
])

docs = [doc["text"] for doc in cursor]
reranked = client.rerank(query=query, documents=docs, model="voyage-2")

print("Reranked top result:")
print(reranked.results[0].document)
