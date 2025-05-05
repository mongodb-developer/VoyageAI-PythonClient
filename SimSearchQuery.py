query = "What are the guidelines for flu vaccines?"
query_embedding = normalize(client.embed(
    texts=[query],
    model="voyage-2",
    input_type="query"
).embeddings[0])

results = collection.aggregate([
    {
        "$vectorSearch": {
            "queryVector": query_embedding,
            "path": "embedding",
            "numCandidates": 100,
            "limit": 5,
            "index": "healthcare_vector_index"
        }
    },
    {"$project": {"text": 1, "_id": 0}}
])
