import pymongo
from voyageai import Client as VoyageClient
import openai

# --- Mongo ---
client = pymongo.MongoClient("")
db = client.voyagenew
collection = db.demo_rag

# --- APIs ---
voyage = VoyageClient(api_key="VOYAGE_KEY_HERE")
openai.api_key = "OPENAI_KEY_HERE"

# --- 1) Retrieve (vector + optional category filter inside $vectorSearch) ---
def retrieve(query, category=None, k=3):
    qvec = voyage.embed([query], model="voyage-4-large").embeddings[0]

    stage = {
        "$vectorSearch": {
            "index": "vector_index",
            "path": "embedding",
            "queryVector": qvec,
            "numCandidates": 50,
            "limit": k,
        }
    }
    if category:
        stage["$vectorSearch"]["filter"] = {"category": category}

    return list(collection.aggregate([
        stage,
        {"$project": {"_id": 0, "text": 1, "category": 1, "score": {"$meta": "vectorSearchScore"}}}
    ]))

# --- 2) Tiny router (the "Mixture" part) ---
def route(q):
    ql = q.lower()

    # Optional: make heart queries obviously MoE by always using 2 experts
    if "heart" in ql:
        return ["nutrition", "exercise"]

    if any(w in ql for w in ["food", "diet", "nutrition", "cholesterol"]): return ["nutrition"]
    if any(w in ql for w in ["diabetes", "glucose", "blood sugar", "a1c"]): return ["diabetes"]
    if any(w in ql for w in ["stress", "anxiety", "sleep"]): return ["mental_health"]
    if any(w in ql for w in ["exercise", "workout", "walking", "activity"]): return ["exercise"]

    return ["nutrition", "exercise"]  # fallback: 2 experts

# --- 3) Expert answer ---
def expert_answer(expert, question):
    docs = retrieve(question, category=expert, k=3)
    ctx = "\n".join([f"- {d['text']}" for d in docs]) or "No context."

    system = {
        "nutrition": "You are a nutrition expert. Give practical, safe dietary guidance.",
        "diabetes": "You are a diabetes educator. Give clear, patient-friendly guidance.",
        "exercise": "You are a fitness expert. Give safe exercise guidance.",
        "mental_health": "You are a mental health coach. Give supportive stress guidance.",
    }[expert]

    r = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Q: {question}\n\nContext:\n{ctx}\n\nAnswer briefly:"}
        ],
        max_tokens=180
    )
    return r.choices[0].message["content"].strip()

# --- 4) Aggregate (combine experts into one final answer) ---
def aggregate(question, answers):
    combined = "\n\n".join([f"[{k}]\n{v}" for k, v in answers.items()])
    r = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Combine the expert answers into ONE concise response. Remove repeats."},
            {"role": "user", "content": f"Q: {question}\n\nExpert answers:\n{combined}\n\nFinal:"}
        ],
        max_tokens=220
    )
    return r.choices[0].message["content"].strip()

# --- Run MoE ---
q = "What are some healthy foods for heart health?"
experts = route(q)

print("Router selected experts:", experts)

answers = {e: expert_answer(e, q) for e in experts}

for e, a in answers.items():
    print(f"\n--- Expert: {e} ---\n{a}\n")

final = aggregate(q, answers) if len(answers) > 1 else list(answers.values())[0]

print("\n=== FINAL (Aggregated) ===\n")
print(final)
