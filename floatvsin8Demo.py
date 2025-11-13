import numpy as np
import streamlit as st
from voyageai import Client

# =========================
# CONFIG
# =========================
MODEL = "voyage-context-3"
DIM   = 512
TOPK  = 5

# =========================
# HEALTHCARE CORPUS
# =========================
DOCUMENTS = [
    "Clinical decision support systems are increasingly used to help physicians identify high-risk patients for early intervention.",
    "Telemedicine platforms have improved access to care but require robust workflows for triage, documentation, and follow-up.",
    "Radiology AI models can assist with early detection of abnormalities but must be validated for sensitivity across patient populations.",
    "Electronic health record (EHR) burden continues to contribute to clinician burnout, leading to interest in ambient documentation tools.",
    "Predictive analytics in hospitals are being implemented to detect patient deterioration, such as sepsis or respiratory decline.",
    "Health systems are piloting AI-based scheduling platforms to optimize operating room usage and reduce patient wait times.",
    "Medication-dispensing automation can reduce errors but requires accurate inventory tracking and alert management.",
    "AI-driven population-health tools analyze longitudinal data to identify gaps in preventive care and chronic-disease management.",
    "Natural-language processing is being applied to unstructured clinical notes to improve coding accuracy and reduce administrative overhead.",
    "Remote patient monitoring programs depend on reliable data ingestion from wearable devices and structured clinician review protocols."
]

QUERIES = [
    "AI tools that support clinicians with diagnosis or risk assessment",
    "Operational efficiency improvements using healthcare automation",
    "EHR workflow challenges and tools that reduce documentation burden",
    "Applications of NLP to clinical notes or medical coding",
    "Remote monitoring and predictive analytics for patient deterioration"
]

# =========================
# HELPERS
# =========================
def _as_float32(vecs):
    return np.array(vecs, dtype=np.float32)

def _norm_rows(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

def embed_docs(client, model, dim, dtype):
    r = client.contextualized_embed(
        inputs=[[d] for d in DOCUMENTS],
        model=model,
        input_type="document",
        output_dimension=dim,
        output_dtype=dtype
    )
    return _as_float32([res.embeddings[0] for res in r.results])

def embed_queries(client, model, dim, dtype):
    r = client.contextualized_embed(
        inputs=[[q] for q in QUERIES],
        model=model,
        input_type="query",
        output_dimension=dim,
        output_dtype=dtype
    )
    return _as_float32([res.embeddings[0] for res in r.results])

def cosine_query_doc(Q, D):
    Qn, Dn = _norm_rows(Q), _norm_rows(D)
    return Qn @ Dn.T

def topk_from_scores(scores, k):
    idx = np.argsort(-scores, axis=1)[:, :k]
    val = np.take_along_axis(scores, idx, axis=1)
    return idx, val

def overlap_at_k(ref_topk, alt_topk, k):
    overlaps = []
    for r, a in zip(ref_topk[:, :k], alt_topk[:, :k]):
        overlaps.append(len(set(r).intersection(set(a))) / k)
    return float(np.mean(overlaps)) if overlaps else 0.0

def mrr_agreement(ref_topk, alt_topk):
    rr = []
    for r, a in zip(ref_topk, alt_topk):
        gold = r[0]
        pos = np.where(a == gold)[0]
        rr.append(1.0 / (pos[0] + 1) if len(pos) else 0.0)
    return float(np.mean(rr)) if rr else 0.0

def spearman_rowwise(ref_scores, alt_scores):
    try:
        from scipy.stats import spearmanr
        use_scipy = True
    except Exception:
        use_scipy = False

    cors = []
    for r, a in zip(ref_scores, alt_scores):
        if use_scipy:
            corr, _ = spearmanr(r, a)
            if np.isfinite(corr):
                cors.append(float(corr))
        else:
            def _rank(x): return np.argsort(np.argsort(-x))
            rr, ar = _rank(r), _rank(a)
            rrn = (rr - rr.mean()) / (rr.std() + 1e-12)
            arn = (ar - ar.mean()) / (ar.std() + 1e-12)
            cors.append(float((rrn @ arn) / (len(rrn) - 1)))
    return float(np.mean(cors)) if cors else 0.0

# =========================
# STREAMLIT APP
# =========================
st.set_page_config(page_title="Healthcare Quantization Compare", layout="wide")
st.title("Healthcare Embedding Quantization Comparison")

# --- Sidebar ---
st.sidebar.header("Settings")
api_key = st.sidebar.text_input("Voyage API Key", type="password")
run_button = st.sidebar.button("Run Comparison")

if not run_button:
    st.info("Enter your API key in the sidebar and click **Run Comparison**.")
    st.stop()

if not api_key:
    st.error("Please enter your Voyage API key.")
    st.stop()

client = Client(api_key=api_key)

# --- Compute embeddings ---
with st.spinner("Embedding documents and queries (float32 & int8)…"):
    D_float = embed_docs(client, MODEL, DIM, "float")
    D_int8  = embed_docs(client, MODEL, DIM, "int8")
    Q_float = embed_queries(client, MODEL, DIM, "float")
    Q_int8  = embed_queries(client, MODEL, DIM, "int8")

S_float = cosine_query_doc(Q_float, D_float)
S_int8  = cosine_query_doc(Q_int8,  D_int8)

topk_float_idx, topk_float_val = topk_from_scores(S_float, TOPK)
topk_int8_idx,  topk_int8_val  = topk_from_scores(S_int8,  TOPK)

# --- Agreement metrics ---
st.subheader("Agreement Metrics (float32 vs int8)")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Top-1 overlap", f"{overlap_at_k(topk_float_idx, topk_int8_idx, 1):.3f}")
col2.metric("Top-3 overlap", f"{overlap_at_k(topk_float_idx, topk_int8_idx, 3):.3f}")
col3.metric("Top-5 overlap", f"{overlap_at_k(topk_float_idx, topk_int8_idx, 5):.3f}")
col4.metric("MRR agreement", f"{mrr_agreement(topk_float_idx, topk_int8_idx):.3f}")

spearman = spearman_rowwise(S_float, S_int8)
st.caption(f"Spearman rank correlation across queries: **{spearman:.3f}**")

# --- Per-query results ---
import pandas as pd

st.subheader("Per-Query Retrieval Comparison")

for qi, query in enumerate(QUERIES):
    rows = []
    fi = topk_float_idx[qi]
    ii = topk_int8_idx[qi]
    fv = topk_float_val[qi]
    iv = topk_int8_val[qi]

    for r in range(TOPK):
        rows.append({
            "Rank": r+1,
            "Float32 document": DOCUMENTS[fi[r]],
            "Float32 score": float(fv[r]),
            "Int8 document": DOCUMENTS[ii[r]],
            "Int8 score": float(iv[r]),
            "Same?": "✅" if fi[r] == ii[r] else "❌"
        })

    df = pd.DataFrame(rows)
    with st.expander(f"Query: {query}", expanded=(qi == 0)):
        st.table(df)

