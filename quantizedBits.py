import pymongo
from voyageai import Client

# === CONFIGURATION ===
VOYAGE_API_KEY = ""
MONGODB_URI = ""
DB_NAME = "quantized"
COLLECTION_NAME = "smaller_bits"  # maybe change per dtype/dim

# === SETUP ===
client_voyage = Client(api_key=VOYAGE_API_KEY)
mongo_client = pymongo.MongoClient(MONGODB_URI)
collection = mongo_client[DB_NAME][COLLECTION_NAME]

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

# === CHOOSE YOUR FORMAT ===
EMBED_MODEL = "voyage-context-3"
EMBED_DIM   = 512            # 2048, 1024, 512, or 256
EMBED_DTYPE = "int8"         # "float", "int8", "uint8", "binary", or "ubinary"

# === EMBED (quantized + chosen dimension) ===
resp = client_voyage.contextualized_embed(
    inputs=[[s] for s in sentences],   # one item per inner list = context-agnostic
    model=EMBED_MODEL,
    input_type="document",
    output_dimension=EMBED_DIM,
    output_dtype=EMBED_DTYPE
)

for sentence, result in zip(sentences, resp.results):
    embedding = result.embeddings[0]   # will be a list[int] for non-float dtypes
    doc = {
        "sentence": sentence,
        "vectorEmbedding": embedding,
        "voyage": {
            "model": EMBED_MODEL,
            "dimension": EMBED_DIM,
            "dtype": EMBED_DTYPE
        }
    }
    # NOTE: For binary/ubinary, len(embedding) == EMBED_DIM // 8 (bit-packed).
    collection.insert_one(doc)
    print(f"✅ Embedded ({EMBED_DTYPE}/{EMBED_DIM}): {sentence[:50]}...")

print("🎉 All quantized embeddings stored in MongoDB.")
