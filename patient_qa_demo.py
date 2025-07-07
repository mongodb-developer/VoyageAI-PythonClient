from voyageai import Client
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import pprint

# Step 1: Set up your clients
VOYAGE_API_KEY = "your_voyage_api_key"
ATLAS_URI = "your_mongodb+srv_uri"

client = Client(api_key=VOYAGE_API_KEY)
mongo = MongoClient(ATLAS_URI)
db = mongo["healthcare_demo"]
collection = db["patient_docs"]

# Step 2: Sample documents (in real use, you'd chunk PDF/texts and store them)
docs = [
    {
        "text": "Metformin is commonly used to treat type 2 diabetes. In elderly patients, side effects may include gastrointestinal issues and increased risk of lactic acidosis.",
        "source": "clinical_guidelines/metformin",
    },
    {
        "text": "Patients with impaired kidney function should be monitored closely when prescribed metformin.",
        "source": "clinical_guidelines/kidney",
    },
    {
        "text": "Statins may cause muscle pain or liver enzyme abnormalities.",
        "source": "clinical_guidelines/statins",
    },
]

# Step 3: Embed and store documents in MongoDB with vector field
texts = [doc["text"] for doc in docs]
doc_embeddings = client.embed(texts=texts, model="voyage-2", input_type="document").embeddings

for i, doc in enumerate(docs):
    doc["embedding"] = doc_embeddings[i]
    collection.insert_one(doc)

# Step 4: Index the vector field
collection.create_index([("embedding", "vector")], name="embedding_vector_index", default_language="none")

# Step 5: Ask a natural language question
query = "What are the side effects of metformin in elderly patients?"
query_embedding = client.embed([query], model="voyage-2", input_type="query").embeddings[0]

# Step 6: Perform vector search in Atlas
results = collection.aggregate([
    {
        "$vectorSearch": {
            "queryVector": query_embedding,
            "path": "embedding",
            "numCandidates": 100,
            "limit": 3,
            "index": "embedding_vector_index"
        }
    }
])

# Step 7: Display results
print("Top matching passages:")
for result in results:
    pprint.pprint({"text": result["text"], "source": result["source"]})
