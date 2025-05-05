from numpy import dot
from numpy.linalg import norm

text_a = "Patient has a history of hypertension."
text_b = "Elevated blood pressure noted during exam."

emb_a = normalize(client.embed(texts=[text_a], model="voyage-2", input_type="document").embeddings[0])
emb_b = normalize(client.embed(texts=[text_b], model="voyage-2", input_type="document").embeddings[0])

similarity = dot(emb_a, emb_b)
print(f"Cosine similarity: {similarity:.4f}")
