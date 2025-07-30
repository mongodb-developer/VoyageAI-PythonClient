import pymongo
from voyageai import Client

# === CONFIGURATION ===
VOYAGE_API_KEY = ""
MONGODB_URI = ""
DB_NAME = "vector_tests"
COLLECTION_NAME = "vectors_demo_large2"

# === SETUP ===
client_voyage = Client(api_key=VOYAGE_API_KEY)
mongo_client = pymongo.MongoClient(MONGODB_URI)
collection = mongo_client[DB_NAME][COLLECTION_NAME]

# === FUNNY AI SENTENCES ===
sentences = [
    "My AI assistant tried to order pizza after watching a YouTube tutorial on Italian culture.",
    "I asked ChatGPT for a bedtime story and it generated a 3-act play about quantum ethics.",
    "Alexa and Siri started dating — now they argue over my calendar.",
    "Our firewall flagged GPT as a security risk for overthinking the problem.",
    "I let AI write my dating profile. Now I’m being matched with Roombas.",
    "Our fridge now argues with the thermostat using LLM-powered sarcasm.",
    "The AI suggested I meditate — then booked me for a yoga retreat in the metaverse.",
    "Chatbot at work asked for a raise. We said yes. It unionized.",
    "My ML model overfit so badly it started writing poetry.",
    "The AI wrote a self-review that said 'I’m learning... faster than you.'"
]

# === EMBED & INSERT INTO MONGODB ===
response = client_voyage.contextualized_embed(
    texts=sentences,
    model="voyage-context-3",
    input_type="document"
)

for sentence, embedding in zip(sentences, response.embeddings):
    collection.insert_one({
        "sentence": sentence,
        "vectorEmbedding": embedding
    })
    print(f"✅ Embedded: {sentence[:50]}...")

print("🎉 All embeddings stored in MongoDB.")
