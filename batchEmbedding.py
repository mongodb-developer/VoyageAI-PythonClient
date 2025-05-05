# Embed in batches for performance and rate limits

texts = [f"Clinical note {i}: ...health summary..." for i in range(100)]

BATCH_SIZE = 32
embeddings = []

for i in range(0, len(texts), BATCH_SIZE):
    batch = texts[i:i+BATCH_SIZE]
    response = client.embed(texts=batch, model="voyage-lite-02-instruct", input_type="document")
    embeddings.extend(response.embeddings)

print(f"Embedded {len(embeddings)} documents.")
