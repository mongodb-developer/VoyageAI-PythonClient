import textwrap
import math
from collections import defaultdict
from typing import List, Tuple
import numpy as np
import voyageai

# ============ Setup ============
vo = voyageai.Client(api_key=" voyage key here ")

documents = [
    "Slack integrates with thousands of tools, making it a central hub for workplace communication.",
    "Microsoft Teams offers built-in video calls and deep integration with Office 365.",
    "Notion is popular for personal productivity, wikis, and project management.",
    "Zoom excels at video conferencing and webinars, especially for large groups.",
    "Trello is a visual task management tool based on boards, lists, and cards.",
    "Google Meet is simple, browser-based, and well-suited for quick team meetings."
]

no_instruction_query = "What tool is best for remote collaboration?"
instruction_query = (
    "Focus on video conferencing and real-time meetings. "
    "What tool is best for remote collaboration?"
)

# ============ Utilities ============
def tok(s: str) -> List[str]:
    return [t for t in ''.join(ch.lower() if ch.isalnum() else ' ' for ch in s).split() if t]

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0

# ============ Retriever 1: Lexical (very light BM25-ish) ============
def lexical_retrieve(query: str, docs: List[str], k: int = 5) -> List[Tuple[int, float]]:
    q = set(tok(query))
    scores = []
    for i, d in enumerate(docs):
        dtoks = tok(d)
        overlap = len(q.intersection(dtoks))
        # small length normalization
        score = overlap / (len(set(dtoks)) + 1e-6)
        if score > 0:
            scores.append((i, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]

# ============ Retriever 2: Embedding (Voyage) ============
def embedding_retrieve(query: str, docs: List[str], k: int = 5, model: str = "voyage-3"):
    q_emb = vo.embed(texts=[query], model=model).embeddings[0]
    d_embs = vo.embed(texts=docs, model=model).embeddings
    q_vec = np.array(q_emb, dtype=np.float32)
    sims = [(i, cosine(q_vec, np.array(e, dtype=np.float32))) for i, e in enumerate(d_embs)]
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:k]

# ============ Fusion: Reciprocal Rank Fusion (RRF) ============
def rrf_fuse(rank_lists: List[List[Tuple[int, float]]], k: int = 60) -> List[int]:
    """
    rank_lists: each list is [(doc_idx, score)] in descending order.
    Returns doc indices sorted by fused score (higher first).
    """
    rrf = defaultdict(float)
    for lst in rank_lists:
        for rank, (doc_idx, _) in enumerate(lst, start=1):
            rrf[doc_idx] += 1.0 / (k + rank)
    return [i for i, _ in sorted(rrf.items(), key=lambda x: x[1], reverse=True)]

# ============ Final Stage: Voyage Re-rank over fused candidates ============
def rerank_fusion(query: str,
                  docs: List[str],
                  top_k_candidates_per_retriever: int = 5,
                  final_top_k: int = 3,
                  rerank_model: str = "rerank-2.5"):

    # 1) Gather candidates from multiple retrievers
    lex = lexical_retrieve(query, docs, k=top_k_candidates_per_retriever)
    emb = embedding_retrieve(query, docs, k=top_k_candidates_per_retriever)

    # 2) Fuse (RRF)
    fused_indices = rrf_fuse([lex, emb])

    # 3) Prepare candidate texts (maintain original order for reproducibility)
    candidates = [docs[i] for i in fused_indices]

    if not candidates:
        return []  # nothing matched lexically; fall back could be full rerank if desired

    # 4) Re-rank candidates with Voyage
    resp = vo.rerank(
        query=query,
        documents=candidates,
        model=rerank_model,
        top_k=min(final_top_k, len(candidates))
    )

    # Map back to original doc indices
    results = []
    for r in resp.results:
        # r.index is position within 'candidates'
        orig_idx = fused_indices[r.index]
        results.append({
            "orig_index": orig_idx,
            "document": docs[orig_idx],
            "relevance_score": r.relevance_score
        })
    return results

def show(results, title):
    print(f"\n=== {title} ===")
    if not results:
        print("No results.")
        return
    for i, r in enumerate(results, 1):
        snippet = textwrap.shorten(r["document"], width=100, placeholder="…")
        print(f"{i}. [score={r['relevance_score']:.4f}] {snippet}")

# ============ Run: Fusion vs your original single-step rerank ============
# Fusion runs (two query flavors to show effect)
fusion_base = rerank_fusion(no_instruction_query, documents, final_top_k=3, rerank_model="rerank-2.5")
fusion_inst = rerank_fusion(instruction_query, documents, final_top_k=3, rerank_model="rerank-2.5")

show(fusion_base, "Re-rank FUSION | Baseline query")
show(fusion_inst, "Re-rank FUSION | With instruction")

# Optionally compare lightweight model
fusion_base_lite = rerank_fusion(no_instruction_query, documents, final_top_k=3, rerank_model="rerank-2.5-lite")
fusion_inst_lite = rerank_fusion(instruction_query, documents, final_top_k=3, rerank_model="rerank-2.5-lite")

show(fusion_base_lite, "Re-rank FUSION (lite) | Baseline query")
show(fusion_inst_lite, "Re-rank FUSION (lite) | With instruction")
