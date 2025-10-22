# re_rank_fusion_demo.py
# -------------------------------------------
# A compact demo showing:
# - Contrastive instructional query
# - Expanded passages (bigger candidate pool)
# - VoyageAI rerank (2.5 / 2.5-lite)
# - Tiny keyword tiebreak for measurable shifts
# - Multi-query RRF fusion
# - Metrics (nDCG@3, MRR) + Kendall-τ rank shift
#
# pip install voyageai

import math
import textwrap
from collections import defaultdict
import voyageai

# ========== Setup ==========
vo = voyageai.Client(api_key="PUT_YOUR_VOYAGE_KEY_HERE")  # <-- add your key

# Expanded passages (2+ per product) to create a richer candidate pool
documents = [
    "Zoom: video conferencing and webinars; strong for large groups, screen sharing, and recordings.",
    "Zoom: chat exists but core strength is synchronous meetings and webinars; built for reliable video.",
    "Google Meet: browser-based, quick meetings; good video calls and team huddles.",
    "Google Meet: lightweight video meetings in Google Workspace; easy join links, captions.",
    "Microsoft Teams: built-in video calls with Office 365; collaboration suite across channels and files.",
    "Microsoft Teams: meetings + channels; deep O365 integration, webinars and town halls available.",
    "Slack: central hub for communication; integrates thousands of tools; not video-first.",
    "Slack: huddles exist but primarily chat and integrations; async-first culture and channels.",
    "Notion: personal productivity, wikis, project docs; not for real-time video meetings.",
    "Trello: task boards, lists, and cards; project management; not for video calls.",
]

# Ground truth (indices) when the intent is "video-first"
ground_truth = {0, 1, 2, 3, 4, 5}  # Zoom, Google Meet, Teams variants

# ========== Queries ==========
def make_instructional_query(user_q: str) -> str:
    return (
        "Rerank the following documents for the user's intent.\n"
        "Prefer: video conferencing, real-time meetings, webinars, large meetings, call quality, reliability.\n"
        "Avoid/Downrank: chat hubs, wikis, task boards, project management, integrations without video focus.\n"
        "If two items tie, prefer tools purpose-built for synchronous video.\n\n"
        f"User question: {user_q}"
    )

no_instruction_query = "What tool is best for remote collaboration?"
instruction_query = make_instructional_query("What tool is best for remote collaboration?")

# Contrastive expansions for RRF
intent_expansions = [
    "Best tool for high-quality video conferencing and webinars",
    "Real-time meetings with large groups and robust screen sharing",
    "Stability for long video calls and recordings for distributed teams",
    "Deprioritize project management and chat-first platforms",
]

# ========== Metrics ==========
def ndcg_at_k(rank_indices, k=3):
    dcg = 0.0
    for i, idx in enumerate(rank_indices[:k], 1):
        rel = 1 if idx in ground_truth else 0
        dcg += (2**rel - 1) / math.log2(i + 1)
    ideal = [1, 1, 1][:k]
    idcg = sum((2**r - 1) / math.log2(i + 1) for i, r in enumerate(ideal, 1))
    return dcg / idcg if idcg > 0 else 0.0

def mrr(rank_indices):
    for i, idx in enumerate(rank_indices, 1):
        if idx in ground_truth:
            return 1.0 / i
    return 0.0

def kendall_tau(a, b):
    pos_a = {idx: i for i, idx in enumerate(a)}
    pos_b = {idx: i for i, idx in enumerate(b)}
    common = [idx for idx in a if idx in pos_b]
    n = len(common)
    if n < 2: return 0.0
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
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

# ========== Rerank helpers ==========
def run_rerank(query, model="rerank-2.5", top_k=10):
    return vo.rerank(query=query, documents=documents, model=model, top_k=top_k).results

VIDEO_TERMS = {"video", "meeting", "meetings", "webinar", "webinars", "call", "calls", "recordings", "screen"}

def keyword_bias(text: str) -> float:
    toks = {t.strip(".,;:()").lower() for t in text.split()}
    return sum(1 for t in toks if t in VIDEO_TERMS) / 10.0  # gentle nudge

def rerank_with_bias(query, model="rerank-2.5", top_k=10, alpha=0.9):
    resp = vo.rerank(query=query, documents=documents, model=model, top_k=top_k)
    rescored = []
    for r in resp.results:
        s = alpha * r.relevance_score + (1 - alpha) * keyword_bias(r.document)
        # preserve original index if provided; otherwise map by text
        idx = getattr(r, "index", None)
        if idx is None:
            idx = documents.index(r.document)
        rescored.append((idx, r.document, s))
    rescored.sort(key=lambda x: x[2], reverse=True)
    # convert back to a simple result-like object
    class R: pass
    out = []
    for idx, doc, s in rescored:
        rr = R()
        rr.document = doc
        rr.index = idx
        rr.relevance_score = s
        out.append(rr)
    return out

# ========== RRF Fusion ==========
def rrf_fuse(list_of_rank_lists, k=10, k_rrf=60):
    """Each list item is [(doc_index, score), ...] in rank order."""
    scores = defaultdict(float)
    for ranks in list_of_rank_lists:
        for r, (idx, _) in enumerate(ranks, 1):
            scores[idx] += 1.0 / (k_rrf + r)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
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

# ========== Run Demo ==========
if __name__ == "__main__":
    # Single-shot (no tiebreak)
    base = run_rerank(no_instruction_query, model="rerank-2.5", top_k=10)
    inst = run_rerank(instruction_query, model="rerank-2.5", top_k=10)
    base_indices = show_table("rerank-2.5 | Baseline (no instruction)", base)
    inst_indices  = show_table("rerank-2.5 | With instruction (raw)", inst, baseline_indices=base_indices)

    print("\n--- Metrics (video-first relevance, raw) ---")
    print(f"nDCG@3  baseline={ndcg_at_k(base_indices, 3):.3f}  instruction={ndcg_at_k(inst_indices, 3):.3f}")
    print(f"MRR     baseline={mrr(base_indices):.3f}         instruction={mrr(inst_indices):.3f}")
    print(f"Kendall-τ (rank shift) = {kendall_tau(base_indices, inst_indices):.3f}")

    # With tiny keyword tiebreak (measurable shift)
    base_b = rerank_with_bias(no_instruction_query, model="rerank-2.5", top_k=10, alpha=0.9)
    inst_b = rerank_with_bias(instruction_query, model="rerank-2.5", top_k=10, alpha=0.9)
    base_b_indices = show_table("rerank-2.5 | Baseline + keyword tiebreak", base_b)
    inst_b_indices  = show_table("rerank-2.5 | Instruction + keyword tiebreak", inst_b, baseline_indices=base_b_indices)

    print("\n--- Metrics (video-first relevance, biased) ---")
    print(f"nDCG@3  baseline={ndcg_at_k(base_b_indices, 3):.3f}  instruction={ndcg_at_k(inst_b_indices, 3):.3f}")
    print(f"MRR     baseline={mrr(base_b_indices):.3f}         instruction={mrr(inst_b_indices):.3f}")
    print(f"Kendall-τ (rank shift) = {kendall_tau(base_b_indices, inst_b_indices):.3f}")

    # Multi-query fusion (RRF) using contrastive expansions
    expanded_runs = [run_rerank(q, model="rerank-2.5", top_k=10) for q in intent_expansions]
    fused = rrf_fuse([to_rank_list(r) for r in expanded_runs], k=10)
    fused_indices = show_table("RRF Fusion | 4 contrastive sub-intents", fused, baseline_indices=base_indices)

    print("\n--- Metrics (fusion) ---")
    print(f"nDCG@3 = {ndcg_at_k(fused_indices, 3):.3f}   MRR = {mrr(fused_indices):.3f}")

    # Light model comparison
    lite_base = run_rerank(no_instruction_query, model="rerank-2.5-lite", top_k=10)
    lite_inst = run_rerank(instruction_query, model="rerank-2.5-lite", top_k=10)
    print("\n--- Model Agreement (2.5 vs 2.5-lite) ---")
    print(f"τ(baseline)   = {kendall_tau(base_indices, [getattr(r,'index', documents.index(r.document)) for r in lite_base]):.3f}")
    print(f"τ(instruction)= {kendall_tau(inst_indices,  [getattr(r,'index', documents.index(r.document)) for r in lite_inst]):.3f}")

    # Executive summary
    b, i, f = ndcg_at_k(base_indices, 3), ndcg_at_k(inst_indices, 3), ndcg_at_k(fused_indices, 3)
    print(f"\n▶ Exec Summary (raw): Instruction +{(i-b)*100:.1f} nDCG@3 pts; RRF fusion +{(f-b)*100:.1f} pts vs baseline.")

    b2, i2 = ndcg_at_k(base_b_indices, 3), ndcg_at_k(inst_b_indices, 3)
    print(f"▶ Exec Summary (biased): Instruction +{(i2-b2)*100:.1f} nDCG@3 pts vs baseline + tiebreak.")
