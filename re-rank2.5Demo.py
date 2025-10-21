import textwrap
import voyageai

# Initialize client with your demo key directly
vo = voyageai.Client(api_key="Add your voyage key here")  # replace with your real key


import math, textwrap
from collections import defaultdict

# --- Demo corpus: tech/productivity articles --- change the corpus to tweak the outcomes 
documents = [
    "Slack integrates with thousands of tools, making it a central hub for workplace communication.",
    "Microsoft Teams offers built-in video calls and deep integration with Office 365.",
    "Notion is popular for personal productivity, wikis, and project management.",
    "Zoom excels at video conferencing and webinars, especially for large groups.",
    "Trello is a visual task management tool based on boards, lists, and cards.",
    "Google Meet is simple, browser-based, and well-suited for quick team meetings."
]

# --- Queries ---
no_instruction_query = "What tool is best for remote collaboration?"
instruction_query = (
    "Focus on video conferencing and real-time meetings. "
    "What tool is best for remote collaboration?"
)

# Optional: sub-intents for "video-first" (used by RRF)
intent_expansions = [
    "Best tool for high-quality video conferencing and webinars",
    "Real-time meetings with large groups",
    "Stable video calls for distributed teams",
    "Screen sharing and recording for live sessions",
]

# --- Tiny ground truth for metrics (for the demo) ---
# Relevant docs (by index) when the intent is "video-first"
ground_truth = {3, 5, 1}  # Zoom, Google Meet, Microsoft Teams

# --- VoyageAI calls (you already have this) ---
def run_rerank(query, model="rerank-2.5", top_k=6):
    resp = vo.rerank(query=query, documents=documents, model=model, top_k=top_k)
    return resp.results  # each has .document, .index (optional), .relevance_score

# --- Helpers ---
def ndcg_at_k(rank_indices, k=3):
    """rank_indices: list of doc indices returned in order (best->worst)."""
    dcg = 0.0
    for i, idx in enumerate(rank_indices[:k], 1):
        rel = 1 if idx in ground_truth else 0
        dcg += (2**rel - 1) / math.log2(i + 1)
    # ideal DCG
    ideal = [1, 1, 1][:k]
    idcg = sum((2**r - 1) / math.log2(i + 1) for i, r in enumerate(ideal, 1))
    return dcg / idcg if idcg > 0 else 0.0

def mrr(rank_indices):
    for i, idx in enumerate(rank_indices, 1):
        if idx in ground_truth:
            return 1.0 / i
    return 0.0

def kendall_tau(a, b):
    """Kendall-τ on overlapping items (indices)."""
    pos_a = {idx: i for i, idx in enumerate(a)}
    pos_b = {idx: i for i, idx in enumerate(b)}
    common = [idx for idx in a if idx in pos_b]
    n = len(common)
    if n < 2: return 0.0
    concordant = discordant = 0
    for i in range(n):
        for j in range(i+1, n):
            ia, ja = pos_a[common[i]], pos_a[common[j]]
            ib, jb = pos_b[common[i]], pos_b[common[j]]
            concordant += (ia - ja) * (ib - jb) > 0
            discordant += (ia - ja) * (ib - jb) < 0
    denom = concordant + discordant
    return (concordant - discordant) / denom if denom else 0.0

def ascii_bar(score, width=18):
    blocks = "▁▂▃▄▅▆▇█"
    n = max(0, min(width, int(round(score * width))))
    return blocks[-1] * n or "·"

def show_table(title, results, baseline_indices=None):
    print(f"\n=== {title} ===")
    print("Rank  Score    Doc")
    print("----  -------  -------------------------------------------------------------")
    idxs = []
    for i, r in enumerate(results, 1):
        # Try to recover the original index; if not provided, map by string match
        idx = getattr(r, "index", None)
        if idx is None:
            idx = documents.index(r.document)
        idxs.append(idx)
        bar = ascii_bar(getattr(r, "relevance_score", 0.0))
        snippet = textwrap.shorten(r.document, width=60, placeholder="…")
        delta = ""
        if baseline_indices is not None and idx in baseline_indices:
            delta_rank = baseline_indices.index(idx) - (i - 1)
            sign = "+" if delta_rank > 0 else ""
            delta = f"  ({sign}{delta_rank}↑)" if delta_rank != 0 else "  (0)"
        print(f"{i:<4}  {getattr(r, 'relevance_score', 0.0):<7.4f}  {bar}  {snippet}{delta}")
    return idxs

# --- Baseline vs Instruction (single-shot) ---
base = run_rerank(no_instruction_query, model="rerank-2.5", top_k=6)
inst = run_rerank(instruction_query, model="rerank-2.5", top_k=6)

base_indices = show_table("rerank-2.5 | Baseline (no instruction)", base)
inst_indices  = show_table("rerank-2.5 | With instruction", inst, baseline_indices=base_indices)

# Metrics
print("\n--- Metrics (video-first relevance) ---")
print(f"nDCG@3  baseline={ndcg_at_k(base_indices, 3):.3f}  instruction={ndcg_at_k(inst_indices, 3):.3f}")
print(f"MRR     baseline={mrr(base_indices):.3f}         instruction={mrr(inst_indices):.3f}")
print(f"Kendall-τ (rank shift) = {kendall_tau(base_indices, inst_indices):.3f}")

# --- Multi-query fusion (RRF) for an “agentic” feel ---
def rrf_fuse(list_of_rank_lists, k=6, k_rrf=60):
    """list_of_rank_lists: each is a list of (doc_index, score) in rank order."""
    scores = defaultdict(float)
    for ranks in list_of_rank_lists:
        for r, (idx, _) in enumerate(ranks, 1):
            scores[idx] += 1.0 / (k_rrf + r)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    # create fake result objects-like for display
    class R: pass
    out = []
    for idx, s in fused[:k]:
        rr = R()
        rr.document = documents[idx]
        rr.relevance_score = s
        rr.index = idx
        out.append(rr)
    return out

def to_rank_list(results):
    arr = []
    for r in results:
        idx = getattr(r, "index", None)
        if idx is None:
            idx = documents.index(r.document)
        arr.append((idx, getattr(r, "relevance_score", 0.0)))
    return arr

# Run multiple focused re-ranks and fuse
expanded_runs = [run_rerank(q, model="rerank-2.5", top_k=6) for q in intent_expansions]
fused = rrf_fuse([to_rank_list(r) for r in expanded_runs], k=6)

fused_indices = show_table("RRF Fusion | 4 sub-intents (video-first)", fused, baseline_indices=base_indices)
print("\n--- Metrics (fusion) ---")
print(f"nDCG@3 = {ndcg_at_k(fused_indices, 3):.3f}   MRR = {mrr(fused_indices):.3f}")

# --- Light model comparison headline (optional) ---
lite_base = run_rerank(no_instruction_query, model="rerank-2.5-lite", top_k=6)
lite_inst = run_rerank(instruction_query, model="rerank-2.5-lite", top_k=6)
print("\n--- Model Agreement (2.5 vs 2.5-lite) ---")
print(f"τ(baseline)   = {kendall_tau(base_indices, [getattr(r,'index', documents.index(r.document)) for r in lite_base]):.3f}")
print(f"τ(instruction)= {kendall_tau(inst_indices,  [getattr(r,'index', documents.index(r.document)) for r in lite_inst]):.3f}")

# --- One-line executive summary ---
def headline():
    b, i, f = ndcg_at_k(base_indices,3), ndcg_at_k(inst_indices,3), ndcg_at_k(fused_indices,3)
    lift_i = (i-b)*100
    lift_f = (f-b)*100
    print(f"\n▶ Exec Summary: Instruction +{lift_i:.1f} nDCG@3 pts; RRF fusion +{lift_f:.1f} pts vs baseline.")

headline()
