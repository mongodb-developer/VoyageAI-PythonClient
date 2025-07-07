claims = [...]  # load claim text data
embeddings = client.embed(texts=claims, model="voyage-2", input_type="document").embeddings
# Insert into MongoDB and run vector search for a new claim
