import os
from datetime import datetime
from typing import Dict, Any, List

import streamlit as st
from pymongo import MongoClient
from dotenv import load_dotenv
import voyageai


st.set_page_config(
    page_title="RFP Agent Demo",
    layout="wide",
)

# =========================
# CONFIG goes here! 😄)
# =========================

MONGODB_URI = ""
DB_NAME = "RFP_Demo"

VOYAGE_API_KEY = ""

INDEX_NAME = "vector_index"
EMBED_MODEL = "voyage-4-large"

COLLECTIONS = [
    "knowledge_base",
    "historical_rfp_answers",
    "source_documents"
]

AGENT_RUNS_COLLECTION = "agent_runs"


@st.cache_resource
def get_db():
    if not MONGODB_URI:
        return None
    client = MongoClient(MONGODB_URI)
    return client[DB_NAME]


@st.cache_resource
def get_voyage():
    if not VOYAGE_API_KEY:
        return None
    return voyageai.Client(api_key=VOYAGE_API_KEY)


db = get_db()
vo = get_voyage()


def source_label(name: str) -> str:
    return {
        "knowledge_base": "Knowledge Base",
        "historical_rfp_answers": "Historical RFP Answers",
        "source_documents": "Source Documents",
    }.get(name, name)


def parse_questions(text: str) -> List[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def check_runtime_readiness() -> None:
    if not MONGODB_URI:
        st.error("MONGODB_URI is missing. Add it to your .env file or environment.")
        st.stop()

    if not VOYAGE_API_KEY:
        st.error("VOYAGE_API_KEY is missing. Add it to your .env file or environment.")
        st.stop()

    if db is None:
        st.error("MongoDB client could not be initialized.")
        st.stop()

    if vo is None:
        st.error("Voyage client could not be initialized.")
        st.stop()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def embed_query(text: str) -> List[float]:
    return vo.embed(
        texts=[text],
        model=EMBED_MODEL,
    ).embeddings[0]


def search_collection(name: str, query_vec: List[float], limit: int = 3) -> List[Dict[str, Any]]:
    pipeline = [
        {
            "$vectorSearch": {
                "index": INDEX_NAME,
                "path": "embedding",
                "queryVector": query_vec,
                "numCandidates": 20,
                "limit": limit,
            }
        },
        {
            "$project": {
                "_id": 1,
                "question_text": 1,
                "answer_text": 1,
                "title": 1,
                "chunk_text": 1,
                "review_status": 1,
                "outcome": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]

    results = list(db[name].aggregate(pipeline))
    for r in results:
        r["source_collection"] = name
        r["_id"] = str(r["_id"])
    return results


def rerank_score(doc: Dict[str, Any]) -> float:
    score = safe_float(doc.get("score", 0))

    if doc.get("review_status") == "approved":
        score += 0.08
    if doc.get("review_status") == "stale":
        score -= 0.08

    if doc.get("outcome") == "won":
        score += 0.08
    if doc.get("outcome") == "used":
        score += 0.03
    if doc.get("outcome") == "lost":
        score -= 0.08

    return score


def explain_signals(doc: Dict[str, Any]) -> List[str]:
    signals = []

    if doc.get("review_status") == "approved":
        signals.append("APPROVED")
    if doc.get("review_status") == "stale":
        signals.append("STALE")
    if doc.get("outcome") == "won":
        signals.append("PRIOR_WIN")
    if doc.get("outcome") == "used":
        signals.append("PREVIOUSLY_USED")
    if doc.get("outcome") == "lost":
        signals.append("PRIOR_LOSS")
    if doc.get("source_collection") == "source_documents":
        signals.append("POLICY_SOURCE")

    return signals


def run_search(query: str, per_collection_limit: int = 3) -> List[Dict[str, Any]]:
    query_vec = embed_query(query)

    all_results: List[Dict[str, Any]] = []
    for name in COLLECTIONS:
        all_results.extend(search_collection(name, query_vec, per_collection_limit))

    for r in all_results:
        r["final_score"] = rerank_score(r)
        r["signals"] = explain_signals(r)

    return sorted(all_results, key=lambda x: x["final_score"], reverse=True)


def classify_question(question: str) -> Dict[str, Any]:
    q = question.lower()

    if any(term in q for term in ["hipaa", "soc2", "iso", "compliance", "audit"]):
        category = "COMPLIANCE"
        route = "sme_team_compliance"
    elif any(term in q for term in ["security", "encryption", "access control", "incident"]):
        category = "SECURITY"
        route = "sme_team_security"
    elif any(term in q for term in ["sla", "uptime", "availability", "disaster recovery"]):
        category = "PLATFORM"
        route = "sme_team_platform"
    elif any(term in q for term in ["pricing", "cost", "commercial", "license"]):
        category = "COMMERCIAL"
        route = "sme_team_commercial"
    else:
        category = "GENERAL"
        route = "sme_team_general"

    return {
        "category": category,
        "suggested_route": route,
    }


def assess_result(top_result: Dict[str, Any]) -> Dict[str, Any]:
    score = safe_float(top_result.get("final_score", 0))
    signals = top_result.get("signals", [])

    risk_flags: List[str] = []

    if "STALE" in signals:
        risk_flags.append("Content may be stale")
    if "PRIOR_LOSS" in signals:
        risk_flags.append("Based on prior losing content")
    if top_result.get("source_collection") == "source_documents":
        risk_flags.append("Grounded in source policy document")

    if score >= 0.90:
        confidence = "HIGH"
    elif score >= 0.75:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    return {
        "confidence_band": confidence,
        "risk_flags": risk_flags,
    }


def build_trace(
    question: str,
    classification: Dict[str, Any],
    ranked: List[Dict[str, Any]],
    assessment: Dict[str, Any],
) -> List[Dict[str, Any]]:
    top = ranked[0] if ranked else {}
    now = datetime.utcnow().isoformat()

    return [
        {
            "stage": "QUESTION_CLASSIFICATION",
            "timestamp": now,
            "result": {
                "question": question,
                **classification,
            },
        },
        {
            "stage": "ANSWER_SELECTION",
            "timestamp": now,
            "result": {
                "selected_source": top.get("source_collection"),
                "selected_score": top.get("final_score"),
                "vector_score": top.get("score"),
                "signals": top.get("signals", []),
            },
        },
        {
            "stage": "RISK_ASSESSMENT",
            "timestamp": now,
            "result": assessment,
        },
    ]


def save_agent_run(
    question: str,
    classification: Dict[str, Any],
    ranked: List[Dict[str, Any]],
    assessment: Dict[str, Any],
    trace: List[Dict[str, Any]],
) -> str:
    top = ranked[0] if ranked else {}

    record = {
        "question": question,
        "agent_type": "RFP_SELECTION_AGENT",
        "classification": classification,
        "selected_result": top,
        "top_results": ranked[:5],
        "assessment": assessment,
        "trace": trace,
        "created_at": datetime.utcnow(),
    }

    result = db[AGENT_RUNS_COLLECTION].insert_one(record)
    return str(result.inserted_id)


def run_agent_pipeline(question: str, per_collection_limit: int) -> Dict[str, Any]:
    classification = classify_question(question)
    ranked = run_search(question, per_collection_limit=per_collection_limit)

    if not ranked:
        return {
            "success": False,
            "question": question,
            "message": "No ranked results returned from search engine.",
        }

    assessment = assess_result(ranked[0])
    trace = build_trace(question, classification, ranked, assessment)
    run_id = save_agent_run(question, classification, ranked, assessment, trace)

    return {
        "success": True,
        "question": question,
        "run_id": run_id,
        "classification": classification,
        "ranked": ranked,
        "assessment": assessment,
        "trace": trace,
    }


def get_past_runs(limit: int = 10) -> List[Dict[str, Any]]:
    docs = list(
        db[AGENT_RUNS_COLLECTION]
        .find(
            {},
            {
                "question": 1,
                "classification": 1,
                "assessment": 1,
                "selected_result.source_collection": 1,
                "selected_result.final_score": 1,
                "created_at": 1,
            },
        )
        .sort("created_at", -1)
        .limit(limit)
    )

    for doc in docs:
        doc["_id"] = str(doc["_id"])
    return docs


st.title("🤖 RFP Agent Demo")
st.caption("Multi-question agentic answer selection with Atlas trace persistence")

check_runtime_readiness()

with st.sidebar:
    st.subheader("Agent Settings")
    per_collection_limit = st.slider(
        "Results per collection",
        min_value=1,
        max_value=5,
        value=3,
    )

    st.subheader("Environment")
    st.write(f"**DB Name:** `{DB_NAME}`")
    st.write(f"**Saved Runs Collection:** `{AGENT_RUNS_COLLECTION}`")
    st.write(f"**Atlas Vector Index:** `{INDEX_NAME}`")
    st.write(f"**Embedding Model:** `{EMBED_MODEL}`")

    st.subheader("Past Runs")
    past_limit = st.slider(
        "Show recent runs",
        min_value=3,
        max_value=25,
        value=10,
    )

left_panel, right_panel = st.columns([2, 1])

with left_panel:
    questions_raw = st.text_area(
        "RFP Questions (one per line)",
        value="",
        height=180,
        placeholder="Paste one or more RFP questions, one per line...",
    )

    run_agent = st.button("Analyze Questions", type="primary", use_container_width=True)

with right_panel:
    st.subheader("How this agent works")
    st.write("1. Classifies the incoming question")
    st.write("2. Runs vector search across collections")
    st.write("3. Reranks using approval / outcome signals")
    st.write("4. Assesses confidence and risk")
    st.write("5. Saves the full trace in Atlas")

tab1, tab2 = st.tabs(["Analysis Results", "View Past Runs"])

with tab1:
    if run_agent:
        questions = parse_questions(questions_raw)

        if not questions:
            st.warning("Please enter at least one question.")
            st.stop()

        with st.spinner("Running agent pipeline..."):
            all_results = [run_agent_pipeline(question, per_collection_limit) for question in questions]

        success_count = sum(1 for r in all_results if r.get("success"))
        st.success(f"Processed {success_count} of {len(all_results)} questions.")

        for idx, result in enumerate(all_results, start=1):
            with st.container(border=True):
                st.subheader(f"Question #{idx}")
                st.write(f"**Question:** {result.get('question', '')}")

                if not result["success"]:
                    st.error(result["message"])
                    continue

                st.caption(f"Saved Run ID: {result['run_id']}")

                ranked = result["ranked"]
                top = ranked[0]
                classification = result["classification"]
                assessment = result["assessment"]
                trace = result["trace"]

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Category", classification["category"])
                with col2:
                    st.metric("Suggested Route", classification["suggested_route"])
                with col3:
                    st.metric("Confidence", assessment["confidence_band"])

                top_col, risk_col = st.columns([2, 1])

                with top_col:
                    st.subheader("Top Selected Result")
                    st.write(f"**Source:** {source_label(top.get('source_collection'))}")
                    st.write(f"**Final Score:** {safe_float(top.get('final_score')):.4f}")
                    st.write(f"**Vector Score:** {safe_float(top.get('score')):.4f}")
                    st.write(f"**Signals:** {', '.join(top.get('signals', [])) or 'None'}")

                    if top.get("review_status"):
                        st.write(f"**Status:** {top['review_status']}")
                    if top.get("outcome"):
                        st.write(f"**Outcome:** {top['outcome']}")

                    if top.get("source_collection") == "source_documents":
                        if top.get("title"):
                            st.write(f"**Title:** {top.get('title')}")
                        st.write(top.get("chunk_text"))
                    else:
                        if top.get("question_text"):
                            st.write(f"**Matched Question:** {top.get('question_text')}")
                        st.write(top.get("answer_text"))

                with risk_col:
                    st.subheader("Risk Assessment")
                    risk_flags = assessment.get("risk_flags", [])
                    if risk_flags:
                        for flag in risk_flags:
                            st.warning(flag)
                    else:
                        st.success("No major risk flags detected")

                st.subheader("Agent Trace")
                for step in trace:
                    with st.container(border=True):
                        st.write(f"**Stage:** {step['stage']}")
                        st.write(f"**Timestamp:** {step['timestamp']}")
                        st.json(step["result"])

                st.subheader("Ranked Results")
                for i, r in enumerate(ranked, start=1):
                    with st.container(border=True):
                        st.write(f"### Result #{i}")
                        st.write(f"**Source:** {source_label(r.get('source_collection'))}")
                        st.write(f"**Final Score:** {safe_float(r.get('final_score')):.4f}")
                        st.write(f"**Signals:** {', '.join(r.get('signals', [])) or 'None'}")

                        if r.get("source_collection") == "source_documents":
                            if r.get("title"):
                                st.write(r.get("title"))
                            st.write(r.get("chunk_text"))
                        else:
                            if r.get("question_text"):
                                st.write(r.get("question_text"))
                            st.write(r.get("answer_text"))

with tab2:
    st.subheader("Recent Agent Runs")
    st.caption("These are the traces already saved in Atlas.")

    runs = get_past_runs(limit=past_limit)

    if not runs:
        st.info("No past runs found yet.")
    else:
        for run in runs:
            with st.container(border=True):
                st.write(f"**Run ID:** {run['_id']}")
                st.write(f"**Created At:** {run.get('created_at')}")
                st.write(f"**Question:** {run.get('question', '')}")

                classification = run.get("classification", {})
                assessment = run.get("assessment", {})
                selected = run.get("selected_result", {})

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.write(f"**Category:** {classification.get('category', 'N/A')}")
                with c2:
                    st.write(f"**Route:** {classification.get('suggested_route', 'N/A')}")
                with c3:
                    st.write(f"**Confidence:** {assessment.get('confidence_band', 'N/A')}")

                st.write(f"**Selected Source:** {source_label(selected.get('source_collection'))}")
                st.write(f"**Top Final Score:** {safe_float(selected.get('final_score')):.4f}")
