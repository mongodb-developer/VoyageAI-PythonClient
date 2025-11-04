# rerank_with_auto_embeddings.py
# don't need key to query but need key to access the re-ranker

import textwrap, math
from collections import defaultdict
from pymongo import MongoClient
import voyageai

# ==== CONFIG ====
VOYAGE_API_KEY = "YOUR_VOYAGE_KEY"
MONGODB_URI    = "YOUR_MONGODB_ATLAS_URI"
DB_NAME        = "vector_demo"
COLL_NAME      = "docs"            # collection where auto-embedded docs live
EMBED_PATH     = "embedding"       # field the index uses for vectors
INDEX_NAME     = "vs_index"        # your Atlas Vector Search index name
NUM_CANDIDATES = 60                # ANN candidate pool
TOP_K          = 6                 # how many docs to show/rerank

# ==== CLIENTS ====
vo = voyageai.Client(api_key=VOYAGE_API_KEY)
mongo = MongoClient(MONGODB_URI)
coll  = mongo[DB_NAME][COLL_NAME]

# ==== QUERIES ====
no_instruction_query = "What tool is best for remote collaboration?"
instruction_query = (
    "Focus on video conferencing and real-time meetings. "
    "What tool is best for remote collaboration?"
)
intent_expansions = [
    "Best tool for high-quality video conferencing and webinars",
    "Real-time meetings with large groups",
    "Stable video calls for distributed teams",
    "Screen sharing and recording for live sessions",
]

# ==== Vector Search (Stage 1) ====
def embed_query_vec(q: str, model="voyage-3"):
    # embed just the query. You can also use voyage-context-3; either works for retrieval
    r = vo.embed(model=model, input=q)
    return r.embeddings[0]

def atlas_vector_search(query_text: str, top_k=TOP_K, num_candidates=NUM_CANDIDATES):
    qv = embed_query_vec(query_text)  # query vector
    pipeline = [
        {"$vectorSearch": {
            "index": INDEX_NAME,
            "path": EMBED_PATH,
            "queryVector": qv,
            "numCandidates": num_candidates,
            "limit": top_k
        }},
        {"$project": {
            "_id": 0, "text": 1, "score": {"$meta": "vectorSearchScore"}
        }}
    ]
    return list(coll.aggregate(pipeline))  # [{text, score}, ...]

# ==== Rerank (Stage 2) ====
def run_rerank(query, docs, model="rerank-2.5", top_k=TOP_K):
    # docs = list[str]
    resp = vo.rerank(query=query, documents=docs, model=model, top_k=top_k)
    return resp.results  # each has .document, .index (position in docs), .relevance_score

# ==== Display helpers ====
def ascii_bar(score, width=18):
    blocks = "▁▂▃▄▅▆▇█"
    n = max(0, min(width, int(round(score * width))))
    return blocks[-1] * n or "·"

def show_table(title, results, baseline_docs=None):
    print(f"\n=== {title} ===")
    print("Rank  Score    Doc")
    print("----  -------  -------------------------------------------------------------")
    for i, r in enumerate(results, 1):
        bar = ascii_bar(getattr(r, "relevance_score", 0.0))
        snippet = textwrap.shorten(r.document, width=60, placeholder="…")
        delta = ""
        if baseline_docs is not None:
            try:
                prev = baseline_docs.index(r.document)
                delta_rank = prev - (i - 1)
                sign = "+" if delta_rank > 0 else ""
                delta = f"  ({sign}{delta_rank}↑)" if delta_rank != 0 else "  (0)"
            except ValueError:
                pass
        print(f"{i:<4}  {getattr(r,'relevance_score',0.0):<7.4f}  {bar}  {snippet}{delta}")

def to_rank_list(results):
    # list[(idx, score)] with idx in the original docs array
    arr = []
    for r in results:
        arr.append((getattr(r, "index", None), getattr(r, "relevance_score", 0.0)))
    return arr

def rrf_fuse(list_of_rank_lists, k=TOP_K, k_rrf=60):
    scores = defaultdict(float)
    for ranks in list_of_rank_lists:
        for r, (idx, _) in enumerate(ranks, 1):
            scores[idx] += 1.0 / (k_rrf + r)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    class R: pass
    out = []
    for idx, s in fused:
        rr = R()
        rr.document = idx  # placeholder; caller will map back to text
        rr.relevance_score = s
        rr.index = idx
        out.append(rr)
    return out

# ==== Demo flow ====
def run_demo():
    # 1) Retrieve candidates from Atlas (already auto-embedded)
    base_hits = atlas_vector_search(no_instruction_query)
    inst_hits = atlas_vector_search(instruction_query)

    base_docs = [d["text"] for d in base_hits]
    inst_docs = [d["text"] for d in inst_hits]

    # 2) Rerank those candidates with Voyage
    base_rr = run_rerank(no_instruction_query, base_docs, model="rerank-2.5", top_k=TOP_K)
    inst_rr = run_rerank(instruction_query, inst_docs, model="rerank-2.5", top_k=TOP_K)

    show_table("Atlas Vector Search → rerank-2.5 | Baseline (no instruction)", base_rr)
    show_table("Atlas Vector Search → rerank-2.5 | With instruction", inst_rr, baseline_docs=[r.document for r in base_rr])

    # 3) Optional: multi-intent fusion over Atlas candidates, then rerank
    expanded_docs = []
    expanded_ranklists = []
    for q in intent_expansions:
        hits = atlas_vector_search(q)
        docs = [h["text"] for h in hits]
        expanded_docs.append(docs)  # keep for mapping
        rr = run_rerank(q, docs, model="rerank-2.5", top_k=TOP_K)
        # convert to (idx, score) relative to a unified pool; easiest is per-query local index
        expanded_ranklists.append([(r.index, r.relevance_score) for r in rr])

    # Simple fusion over the last candidate set (for demo clarity)
    fused = rrf_fuse(expanded_ranklists, k=TOP_K)
    # Map fused.index back to text from the last expansion set (demo simplification)
    fused = [{
        "document": expanded_docs[-1][r.index],
        "relevance_score": r.relevance_score
    } for r in fused]

    print("\n=== RRF Fusion over sub-intents (demo) ===")
    print("Rank  Score    Doc")
    print("----  -------  -------------------------------------------------------------")
    for i, it in enumerate(fused, 1):
        bar = ascii_bar(it["relevance_score"])
        snippet = textwrap.shorten(it["document"], width=60, placeholder="…")
        print(f"{i:<4}  {it['relevance_score']:<7.4f}  {bar}  {snippet}")

if __name__ == "__main__":
    run_demo()
