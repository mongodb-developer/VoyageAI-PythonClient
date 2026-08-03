# If the customer is committed to OpenAI embeddings,
# position the Voyage reranker as a second-stage accuracy improvement.

import math
import time

import openai
from voyageai import Client as VoyageClient

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


console = Console()


# ============================================================
# Configuration
# ============================================================

VOYAGE_API_KEY = ""
OPENAI_API_KEY = ""

OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
RERANK_MODEL = "rerank-2.5"

# Retrieve at least 10 candidates so nDCG@10 can be calculated.
# This demo contains 12 documents, so retrieve all 12.
RETRIEVAL_LIMIT = 12

# Return the top 10 after reranking.
FINAL_RESULTS = 10

# Evaluation cutoff.
NDCG_K = 10


# ============================================================
# Setup
# ============================================================

# Preserves the pre-1.0 OpenAI Python SDK syntax
# used by the original script.
openai.api_key = OPENAI_API_KEY

voyage = VoyageClient(
    api_key=VOYAGE_API_KEY,
)


# ============================================================
# Inline Example Documents
# ============================================================

DOCUMENTS = [
    {
        "id": "DOC-001",
        "text": (
            "A heart-healthy diet should emphasize vegetables, fruits, "
            "whole grains, beans, nuts, fish, and foods low in saturated fat."
        ),
    },
    {
        "id": "DOC-002",
        "text": (
            "Oats contain soluble fiber, which can help reduce LDL cholesterol "
            "and support cardiovascular health."
        ),
    },
    {
        "id": "DOC-003",
        "text": (
            "Regular aerobic exercise such as brisk walking, cycling, "
            "and swimming can improve cardiovascular fitness and overall "
            "heart health."
        ),
    },
    {
        "id": "DOC-004",
        "text": (
            "Salmon and other fatty fish contain omega-3 fatty acids "
            "that may help support heart health and reduce triglycerides."
        ),
    },
    {
        "id": "DOC-005",
        "text": (
            "Avocados contain monounsaturated fats and can be part "
            "of a balanced diet focused on cardiovascular health."
        ),
    },
    {
        "id": "DOC-006",
        "text": (
            "Stress management techniques such as breathing exercises "
            "and meditation can improve overall mental wellbeing."
        ),
    },
    {
        "id": "DOC-007",
        "text": (
            "Foods high in saturated fat, sodium, and added sugar "
            "should generally be limited when following a heart-healthy "
            "eating pattern."
        ),
    },
    {
        "id": "DOC-008",
        "text": (
            "Walnuts, almonds, and other nuts provide unsaturated fats, "
            "fiber, and other nutrients associated with cardiovascular health."
        ),
    },
    {
        "id": "DOC-009",
        "text": (
            "People with diabetes should monitor blood glucose levels "
            "and follow their individualized treatment plan."
        ),
    },
    {
        "id": "DOC-010",
        "text": (
            "Leafy green vegetables such as spinach and kale provide "
            "vitamins, minerals, fiber, and dietary nitrates."
        ),
    },
    {
        "id": "DOC-011",
        "text": (
            "Whole grains such as oatmeal, brown rice, and whole-wheat "
            "products provide fiber and can support a heart-conscious diet."
        ),
    },
    {
        "id": "DOC-012",
        "text": (
            "Processed meats, fried foods, and foods containing large "
            "amounts of saturated fat may work against cardiovascular "
            "health goals."
        ),
    },
]


# ============================================================
# Human Relevance Judgments
# ============================================================

# These judgments are specific to this question:
#
# "Which foods can help lower cholesterol and improve heart health?"
#
# 3 = highly relevant
# 2 = relevant
# 1 = somewhat relevant
# 0 or missing = not relevant
#
# These labels should be created by a human reviewer.
# For a real benchmark, create separate judgments for each query.

RELEVANCE_JUDGMENTS = {
    "DOC-002": 3,  # Oats and LDL cholesterol
    "DOC-004": 2,  # Fatty fish and heart health
    "DOC-008": 2,  # Nuts and cardiovascular health
    "DOC-011": 2,  # Whole grains and fiber
    "DOC-001": 1,  # General heart-healthy diet
    "DOC-005": 1,  # Avocados and unsaturated fats
    "DOC-010": 1,  # Leafy green vegetables
}


# ============================================================
# OpenAI Embeddings
# ============================================================

def embed_texts(texts):
    """
    Generate embeddings using OpenAI.

    Uses the pre-1.0 OpenAI Python SDK syntax from
    the original demo.
    """

    response = openai.Embedding.create(
        model=OPENAI_EMBEDDING_MODEL,
        input=texts,
    )

    return [
        item["embedding"]
        for item in response["data"]
    ]


# ============================================================
# Local Vector Search
# ============================================================

def cosine_similarity(a, b):
    """
    Calculate cosine similarity between two vectors.
    """

    dot_product = sum(
        x * y
        for x, y in zip(a, b)
    )

    magnitude_a = math.sqrt(
        sum(x * x for x in a)
    )

    magnitude_b = math.sqrt(
        sum(x * x for x in b)
    )

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (
        magnitude_a * magnitude_b
    )


def retrieve(
    query_embedding,
    embedded_documents,
    k=RETRIEVAL_LIMIT,
):
    """
    First-stage semantic retrieval using OpenAI embeddings.

    This simulates vector search locally without requiring
    a database or vector store.
    """

    results = []

    for doc in embedded_documents:
        score = cosine_similarity(
            query_embedding,
            doc["embedding"],
        )

        results.append(
            {
                "id": doc["id"],
                "text": doc["text"],
                "vector_score": score,
            }
        )

    results.sort(
        key=lambda item: item["vector_score"],
        reverse=True,
    )

    return results[:k]


# ============================================================
# Voyage Reranking
# ============================================================

def rerank(
    query,
    candidates,
):
    """
    Second-stage reranking using Voyage rerank-2.5.

    Voyage does not receive the OpenAI vectors.

    Voyage receives:

        1. The original query text
        2. The candidate document text

    OpenAI remains the embedding model for the entire
    first-stage retrieval pipeline.
    """

    start = time.time()

    response = voyage.rerank(
        query=query,
        documents=[
            doc["text"]
            for doc in candidates
        ],
        model=RERANK_MODEL,
        top_k=min(
            FINAL_RESULTS,
            len(candidates),
        ),
    )

    elapsed_ms = (
        time.time() - start
    ) * 1000

    results = []

    for new_rank, result in enumerate(
        response.results,
        start=1,
    ):
        original = candidates[result.index]

        results.append(
            {
                "new_rank": new_rank,
                "original_rank": result.index + 1,
                "id": original["id"],
                "text": original["text"],
                "vector_score": original["vector_score"],
                "rerank_score": result.relevance_score,
            }
        )

    return results, elapsed_ms


# ============================================================
# nDCG@K Evaluation
# ============================================================

def dcg_at_k(
    ranked_ids,
    relevance_judgments,
    k=10,
):
    """
    Calculate Discounted Cumulative Gain at K.

    Uses graded relevance with exponential gain:

        gain = 2^relevance - 1
    """

    score = 0.0

    for rank, doc_id in enumerate(
        ranked_ids[:k],
        start=1,
    ):
        relevance = relevance_judgments.get(
            doc_id,
            0,
        )

        gain = (
            2 ** relevance
        ) - 1

        discount = math.log2(
            rank + 1
        )

        score += gain / discount

    return score


def ndcg_at_k(
    ranked_ids,
    relevance_judgments,
    k=10,
):
    """
    Calculate Normalized Discounted Cumulative Gain at K.
    """

    actual_dcg = dcg_at_k(
        ranked_ids,
        relevance_judgments,
        k,
    )

    ideal_relevances = sorted(
        relevance_judgments.values(),
        reverse=True,
    )

    ideal_dcg = sum(
        (
            (2 ** relevance) - 1
        ) / math.log2(rank + 1)
        for rank, relevance in enumerate(
            ideal_relevances[:k],
            start=1,
        )
    )

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg


# ============================================================
# Display: Vector Results
# ============================================================

def show_vector_results(results):
    table = Table(
        title="1️⃣ OpenAI Embedding Retrieval",
        box=box.ROUNDED,
        title_style="bold cyan",
    )

    table.add_column(
        "Rank",
        width=6,
    )

    table.add_column(
        "ID",
        width=10,
    )

    table.add_column(
        "Vector Score",
        width=14,
    )

    table.add_column(
        "Human Relevance",
        width=16,
    )

    table.add_column(
        "Document",
        style="white",
    )

    for rank, doc in enumerate(
        results,
        start=1,
    ):
        relevance = RELEVANCE_JUDGMENTS.get(
            doc["id"],
            0,
        )

        table.add_row(
            f"#{rank}",
            doc["id"],
            f"{doc['vector_score']:.4f}",
            str(relevance),
            doc["text"],
        )

    console.print(table)


# ============================================================
# Display: Reranked Results
# ============================================================

def show_reranked_results(results):
    table = Table(
        title="2️⃣ OpenAI Retrieval + Voyage rerank-2.5",
        box=box.ROUNDED,
        title_style="bold green",
    )

    table.add_column(
        "New Rank",
        width=10,
    )

    table.add_column(
        "Old Rank",
        width=12,
    )

    table.add_column(
        "ID",
        width=10,
    )

    table.add_column(
        "Rerank Score",
        width=14,
    )

    table.add_column(
        "Vector Score",
        width=14,
    )

    table.add_column(
        "Human Relevance",
        width=16,
    )

    table.add_column(
        "Document",
        style="white",
    )

    for doc in results:
        movement = (
            doc["original_rank"]
            - doc["new_rank"]
        )

        if movement > 0:
            movement_text = (
                f"#{doc['original_rank']} "
                f"[green]↑{movement}[/green]"
            )

        elif movement < 0:
            movement_text = (
                f"#{doc['original_rank']} "
                f"[red]↓{abs(movement)}[/red]"
            )

        else:
            movement_text = (
                f"#{doc['original_rank']}"
            )

        relevance = RELEVANCE_JUDGMENTS.get(
            doc["id"],
            0,
        )

        table.add_row(
            f"#{doc['new_rank']}",
            movement_text,
            doc["id"],
            f"{doc['rerank_score']:.4f}",
            f"{doc['vector_score']:.4f}",
            str(relevance),
            doc["text"],
        )

    console.print(table)


# ============================================================
# Display: nDCG Comparison
# ============================================================

def show_ndcg_comparison(
    embedding_ndcg,
    reranked_ndcg,
    k=NDCG_K,
):
    absolute_lift = (
        reranked_ndcg
        - embedding_ndcg
    )

    if embedding_ndcg > 0:
        relative_lift = (
            absolute_lift
            / embedding_ndcg
        ) * 100

        relative_lift_text = (
            f"{relative_lift:+.2f}%"
        )

    else:
        relative_lift_text = "N/A"

    console.print(
        Panel(
            f"[cyan]"
            f"Embeddings nDCG@{k}:"
            f"[/cyan] "
            f"[bold]"
            f"{embedding_ndcg:.4f}"
            f"[/bold]\n\n"

            f"[green]"
            f"Embeddings + Rerank nDCG@{k}:"
            f"[/green] "
            f"[bold]"
            f"{reranked_ndcg:.4f}"
            f"[/bold]\n\n"

            f"[yellow]"
            f"Absolute lift:"
            f"[/yellow] "
            f"[bold]"
            f"{absolute_lift:+.4f}"
            f"[/bold]\n"

            f"[yellow]"
            f"Relative lift:"
            f"[/yellow] "
            f"[bold]"
            f"{relative_lift_text}"
            f"[/bold]",
            title="📊 Ranking Quality",
            border_style="green",
            box=box.DOUBLE,
        )
    )


# ============================================================
# Demo
# ============================================================

def run_demo(question):
    console.print()

    console.print(
        Panel.fit(
            "[bold cyan]"
            "OpenAI Embeddings + Voyage Reranking"
            "[/bold cyan]\n\n"

            f"[yellow]Embedding Model:[/yellow] "
            f"{OPENAI_EMBEDDING_MODEL}\n"

            f"[yellow]Reranker:[/yellow] "
            f"{RERANK_MODEL}\n"

            f"[yellow]Evaluation Metric:[/yellow] "
            f"nDCG@{NDCG_K}\n\n"

            f"[bold]Question:[/bold] "
            f"{question}",
            border_style="cyan",
        )
    )

    console.print()

    # --------------------------------------------------------
    # STEP 1
    # Embed all documents using OpenAI
    # --------------------------------------------------------

    console.print(
        f"[dim]"
        f"Embedding {len(DOCUMENTS)} documents with "
        f"{OPENAI_EMBEDDING_MODEL}..."
        f"[/dim]"
    )

    start = time.time()

    document_embeddings = embed_texts(
        [
            doc["text"]
            for doc in DOCUMENTS
        ]
    )

    document_embed_ms = (
        time.time() - start
    ) * 1000

    embedded_documents = []

    for doc, embedding in zip(
        DOCUMENTS,
        document_embeddings,
    ):
        embedded_documents.append(
            {
                **doc,
                "embedding": embedding,
            }
        )

    console.print(
        f"[green]✓[/green] "
        f"Documents embedded with OpenAI in "
        f"[yellow]"
        f"{document_embed_ms:.1f} ms"
        f"[/yellow]"
    )

    # --------------------------------------------------------
    # STEP 2
    # Embed the query using the same OpenAI model
    # --------------------------------------------------------

    console.print(
        f"[dim]"
        f"Embedding query with "
        f"{OPENAI_EMBEDDING_MODEL}..."
        f"[/dim]"
    )

    start = time.time()

    query_embedding = embed_texts(
        [question]
    )[0]

    query_embed_ms = (
        time.time() - start
    ) * 1000

    console.print(
        f"[green]✓[/green] "
        f"Query embedded with OpenAI in "
        f"[yellow]"
        f"{query_embed_ms:.1f} ms"
        f"[/yellow]"
    )

    # --------------------------------------------------------
    # STEP 3
    # Vector retrieval
    # --------------------------------------------------------

    console.print(
        "[dim]"
        "Running cosine similarity retrieval..."
        "[/dim]"
    )

    start = time.time()

    candidates = retrieve(
        query_embedding,
        embedded_documents,
        k=RETRIEVAL_LIMIT,
    )

    retrieval_ms = (
        time.time() - start
    ) * 1000

    console.print(
        f"[green]✓[/green] "
        f"Retrieved "
        f"[bold]{len(candidates)}[/bold] "
        f"candidates in "
        f"[yellow]"
        f"{retrieval_ms:.1f} ms"
        f"[/yellow]"
    )

    console.print()

    show_vector_results(
        candidates
    )

    # --------------------------------------------------------
    # STEP 4
    # Voyage reranking
    # --------------------------------------------------------

    console.print()

    console.print(
        f"[dim]"
        f"Sending the same "
        f"{len(candidates)} candidates "
        f"to Voyage {RERANK_MODEL}..."
        f"[/dim]"
    )

    reranked, rerank_ms = rerank(
        question,
        candidates,
    )

    console.print(
        f"[green]✓[/green] "
        f"Voyage {RERANK_MODEL} completed in "
        f"[yellow]"
        f"{rerank_ms:.1f} ms"
        f"[/yellow]"
    )

    console.print()

    show_reranked_results(
        reranked
    )

    # --------------------------------------------------------
    # STEP 5
    # Calculate nDCG@10 for both rankings
    # --------------------------------------------------------

    vector_ranked_ids = [
        doc["id"]
        for doc in candidates
    ]

    reranked_ids = [
        doc["id"]
        for doc in reranked
    ]

    embedding_ndcg = ndcg_at_k(
        vector_ranked_ids,
        RELEVANCE_JUDGMENTS,
        k=NDCG_K,
    )

    reranked_ndcg = ndcg_at_k(
        reranked_ids,
        RELEVANCE_JUDGMENTS,
        k=NDCG_K,
    )

    console.print()

    show_ndcg_comparison(
        embedding_ndcg,
        reranked_ndcg,
        k=NDCG_K,
    )

    # --------------------------------------------------------
    # STEP 6
    # Calculate ranking movement
    # --------------------------------------------------------

    reordered = sum(
        1
        for doc in reranked
        if doc["original_rank"]
        != doc["new_rank"]
    )

    promoted = [
        doc
        for doc in reranked
        if doc["original_rank"]
        > doc["new_rank"]
    ]

    biggest_promotion = max(
        (
            doc["original_rank"]
            - doc["new_rank"]
            for doc in promoted
        ),
        default=0,
    )

    # --------------------------------------------------------
    # STEP 7
    # Show final summary
    # --------------------------------------------------------

    console.print()

    console.print(
        Panel(
            "[bold green]"
            "What Changed?"
            "[/bold green]\n\n"

            "OpenAI created every document embedding.\n"
            "OpenAI created the query embedding.\n"
            "Cosine similarity produced the initial ranking.\n\n"

            f"Voyage {RERANK_MODEL} then evaluated "
            "the original query against the candidate text "
            "and refined the ranking.\n\n"

            f"[yellow]"
            f"{reordered}/{len(reranked)}"
            f"[/yellow] "
            f"final results changed position.\n"

            f"Largest promotion: "
            f"[yellow]"
            f"{biggest_promotion} positions"
            f"[/yellow]\n\n"

            f"Embedding-only nDCG@{NDCG_K}: "
            f"[cyan]"
            f"{embedding_ndcg:.4f}"
            f"[/cyan]\n"

            f"Embedding + rerank nDCG@{NDCG_K}: "
            f"[green]"
            f"{reranked_ndcg:.4f}"
            f"[/green]\n\n"

            "[bold white]"
            "The embedding model never changed. "
            "Voyage was added only as a second-stage reranker."
            "[/bold white]",
            title="✨ RESULT ✨",
            border_style="green",
            box=box.DOUBLE,
        )
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    question = (
        "Which foods can help lower cholesterol "
        "and improve heart health?"
    )

    run_demo(question)
