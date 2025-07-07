# Semantic Clinical Notes Retrieval
query = "diabetic patient with neuropathy and retinopathy"
q_emb = client.embed([query], model="voyage-2", input_type="query").embeddings[0]
# Run vector + filter search to fetch matching notes
