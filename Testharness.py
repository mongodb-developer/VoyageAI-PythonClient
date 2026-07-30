#!/usr/bin/env python3

"""
C.H. Robinson logistics embedding evaluation

Compares:
    - Voyage AI voyage-4-lite
    - OpenAI text-embedding-3-small

Collection:
    CH_Robinson.Trucks

Stored vector fields:
    embedding_voyage_4_lite
    embedding_openai_3_small

Atlas Vector Search indexes:
    voyage_4_lite_index
    openai_3_small_index

Python 3.10 compatible.
Uses the older OpenAI SDK syntax:
    openai.Embedding.create(...)

Outputs:
    ch_robinson_eval_results.csv
    ch_robinson_eval_queries.csv
    ch_robinson_eval_summary.csv
"""

import math
import statistics
import time

import openai
import pandas as pd
import voyageai
from pymongo import MongoClient
from pymongo.errors import OperationFailure


# ---------- Config ----------
MONGODB_URI = ""
OPENAI_KEY = ""
VOYAGE_KEY = ""

DB = "CH_Robinson"
COLL = "Trucks"

DIMENSIONS = 1024
TOP_K = 5
NUM_CANDIDATES = 1000

# Exact search is ideal for comparing model quality on this 50-document set.
# Change to False to test ANN behavior with NUM_CANDIDATES.
EXACT_SEARCH = True

RESULTS_CSV = "ch_robinson_eval_results.csv"
QUERY_SCORES_CSV = "ch_robinson_eval_queries.csv"
SUMMARY_CSV = "ch_robinson_eval_summary.csv"

PRINT_TOP_RESULTS = 3
# ----------------------------


MODELS = [
    {
        "label": "voyage-4-lite",
        "provider": "voyage",
        "model": "voyage-4-lite",
        "vector_field": "embedding_voyage_4_lite",
        "index_name": "voyage_4_lite_index",
    },
    {
        "label": "text-embedding-3-small",
        "provider": "openai",
        "model": "text-embedding-3-small",
        "vector_field": "embedding_openai_3_small",
        "index_name": "openai_3_small_index",
    },
]


TEST_QUERIES = [
    {
        "query": "What paperwork and labeling are required for regulated dangerous goods?",
        "expected_title": "Hazardous Materials Shipping",
        "category": "compliance",
    },
    {
        "query": "Why am I being charged based on package size instead of actual weight?",
        "expected_title": "Dimensional Weight",
        "category": "pricing",
    },
    {
        "query": "How can I see where my load is and whether it is still moving?",
        "expected_title": "Shipment Tracking",
        "category": "visibility",
    },
    {
        "query": "How should we ship temperature-sensitive pharmaceuticals?",
        "expected_title": "Refrigerated Shipping",
        "category": "equipment",
    },
    {
        "query": "How do we choose a reliable carrier for a risky lane?",
        "expected_title": "Carrier Selection",
        "category": "carrier-management",
    },
    {
        "query": "How can freight move directly from inbound to outbound without storage?",
        "expected_title": "Cross-Docking",
        "category": "warehouse",
    },
    {
        "query": "What protects us when carrier liability does not cover the full cargo value?",
        "expected_title": "Freight Insurance",
        "category": "risk",
    },
    {
        "query": "Why are we being charged because an import container stayed at the terminal too long?",
        "expected_title": "Demurrage Charges",
        "category": "accessorials",
    },
    {
        "query": "We have a production line down and need the shipment there immediately.",
        "expected_title": "Expedited Freight",
        "category": "service-level",
    },
    {
        "query": "What is the fastest way to move a high-value lightweight shipment overseas?",
        "expected_title": "Air Freight",
        "category": "mode",
    },
    {
        "query": "What system can handle load planning, tendering, tracking, and freight audit?",
        "expected_title": "Transportation Management Systems",
        "category": "technology",
    },
    {
        "query": "How is a shipment's predicted arrival time calculated?",
        "expected_title": "Estimated Time of Arrival",
        "category": "visibility",
    },
    {
        "query": "What trailer should we use for packaged goods that do not need refrigeration?",
        "expected_title": "Dry Van Shipping",
        "category": "equipment",
    },
    {
        "query": "Who connects shippers with third-party motor carriers and manages exceptions?",
        "expected_title": "Freight Brokerage",
        "category": "brokerage",
    },
    {
        "query": "What determines the preferred carrier order when the first carrier rejects a load?",
        "expected_title": "Routing Guides",
        "category": "carrier-management",
    },
    {
        "query": "How do we measure how often carriers accept the loads we offer?",
        "expected_title": "Tender Acceptance",
        "category": "carrier-management",
    },
    {
        "query": "What are liftgate, inside delivery, and redelivery fees called?",
        "expected_title": "Accessorial Charges",
        "category": "accessorials",
    },
    {
        "query": "How should freight be stacked and wrapped on a pallet?",
        "expected_title": "Palletization",
        "category": "handling",
    },
    {
        "query": "How do we verify carrier invoices against contracted rates and approved charges?",
        "expected_title": "Freight Audit and Payment",
        "category": "finance",
    },
    {
        "query": "My shipment is too large for parcel but does not fill a trailer. What mode should I use?",
        "expected_title": "Less-than-Truckload Shipping",
        "category": "mode",
    },
    {
        "query": "How can we track trailers, dock assignments, and dwell time in the yard?",
        "expected_title": "Yard Management",
        "category": "warehouse",
    },
    {
        "query": "How do buyers and sellers define responsibility for freight, insurance, and customs?",
        "expected_title": "Incoterms",
        "category": "international",
    },
    {
        "query": "Can we combine several small shipments going in the same direction?",
        "expected_title": "Load Consolidation",
        "category": "optimization",
    },
    {
        "query": "How should transportation teams prepare for road and port closures caused by storms?",
        "expected_title": "Weather Disruptions",
        "category": "risk",
    },
    {
        "query": "What pricing arrangement works for recurring lanes with negotiated rates?",
        "expected_title": "Contract Freight",
        "category": "pricing",
    },
    {
        "query": "Can a third party run carrier procurement, routing, tracking, and freight payment for us?",
        "expected_title": "Managed Transportation",
        "category": "managed-services",
    },
    {
        "query": "How can two drivers reduce transit time on a long-haul truckload?",
        "expected_title": "Team Driver Service",
        "category": "service-level",
    },
    {
        "query": "What should we do when a pickup is missed or equipment breaks down?",
        "expected_title": "Transportation Exceptions",
        "category": "exceptions",
    },
    {
        "query": "Why was my LTL shipment reclassified after the carrier inspected it?",
        "expected_title": "Freight Classification",
        "category": "pricing",
    },
    {
        "query": "Why do we need reserved pickup and delivery time windows at distribution centers?",
        "expected_title": "Warehouse Appointments",
        "category": "warehouse",
    },
    {
        "query": "Can we move regional shipments together by truckload and separate them near the destination?",
        "expected_title": "Pool Distribution",
        "category": "distribution",
    },
    {
        "query": "How are transportation rates adjusted when diesel prices change?",
        "expected_title": "Fuel Surcharges",
        "category": "pricing",
    },
    {
        "query": "How do we find the best stop sequence while respecting delivery windows and driver hours?",
        "expected_title": "Route Optimization",
        "category": "optimization",
    },
    {
        "query": "How should we manage product returns, recalls, repairs, and recycling?",
        "expected_title": "Reverse Logistics",
        "category": "returns",
    },
    {
        "query": "What equipment should we use for steel or machinery loaded from the side or top?",
        "expected_title": "Flatbed Shipping",
        "category": "equipment",
    },
    {
        "query": "How can we reduce freight emissions and empty miles?",
        "expected_title": "Sustainable Transportation",
        "category": "sustainability",
    },
    {
        "query": "How can we combine order, inventory, and transportation data to spot downstream delays?",
        "expected_title": "Supply Chain Visibility",
        "category": "visibility",
    },
    {
        "query": "Can rail handle the long haul while trucks cover pickup and final delivery?",
        "expected_title": "Intermodal Transportation",
        "category": "mode",
    },
    {
        "query": "Can a carrier leave a trailer at our facility so we can load it later?",
        "expected_title": "Drop Trailer Programs",
        "category": "equipment",
    },
    {
        "query": "How do we compare carriers using on-time delivery, claims, and invoice accuracy?",
        "expected_title": "Carrier Scorecards",
        "category": "carrier-management",
    },
    {
        "query": "What document acts as both a freight receipt and transportation contract?",
        "expected_title": "Bill of Lading",
        "category": "documentation",
    },
    {
        "query": "How do we choose between parcel, LTL, truckload, rail, ocean, and air?",
        "expected_title": "Mode Optimization",
        "category": "optimization",
    },
    {
        "query": "What signed document confirms the consignee received the freight?",
        "expected_title": "Proof of Delivery",
        "category": "documentation",
    },
    {
        "query": "Why are we being charged because the driver waited too long at the dock?",
        "expected_title": "Detention Charges",
        "category": "accessorials",
    },
    {
        "query": "What service moves containers between a port, rail ramp, and nearby warehouse?",
        "expected_title": "Drayage",
        "category": "mode",
    },
    {
        "query": "What is the economical option for a large international shipment that can move slowly?",
        "expected_title": "Ocean Freight",
        "category": "mode",
    },
    {
        "query": "Who helps classify imported goods, calculate duties, and prepare entry documents?",
        "expected_title": "Customs Brokerage",
        "category": "international",
    },
    {
        "query": "When should we dedicate an entire trailer to one shipment?",
        "expected_title": "Full Truckload Shipping",
        "category": "mode",
    },
    {
        "query": "How do we request reimbursement for cargo that arrived damaged or short?",
        "expected_title": "Freight Claims",
        "category": "claims",
    },
    {
        "query": "How do we buy one-time truck capacity for an unplanned load?",
        "expected_title": "Spot Market Freight",
        "category": "pricing",
    },
]


def percentile(values, percentile_value):
    if not values:
        return 0.0

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    rank = (len(ordered) - 1) * percentile_value
    low = math.floor(rank)
    high = math.ceil(rank)

    if low == high:
        return ordered[low]

    fraction = rank - low

    return (
        ordered[low] * (1 - fraction)
        + ordered[high] * fraction
    )


def validate_collection(collection):
    document_count = collection.count_documents({})

    if document_count == 0:
        raise RuntimeError(
            f"No documents found in {DB}.{COLL}"
        )

    print(
        f"Connected to {DB}.{COLL} "
        f"with {document_count} documents."
    )

    for config in MODELS:
        field = config["vector_field"]

        sample = collection.find_one(
            {
                field: {
                    "$type": "array"
                },
                f"{field}.1023": {
                    "$exists": True
                },
            },
            {
                field: 1
            },
        )

        if not sample:
            raise RuntimeError(
                f"No 1024-dimensional vectors found in "
                f"'{field}'. Run the embedding script first."
            )

        vector = sample[field]

        if len(vector) != DIMENSIONS:
            raise RuntimeError(
                f"Stored vector in '{field}' has "
                f"{len(vector)} dimensions; "
                f"expected {DIMENSIONS}."
            )

        print(
            f"Validated {config['label']}: "
            f"field={field}, "
            f"index={config['index_name']}"
        )


def embed_query(config, query, voyage_client):
    started = time.perf_counter()

    if config["provider"] == "voyage":
        response = voyage_client.embed(
            texts=[query],
            model=config["model"],
            input_type="query",
            output_dimension=DIMENSIONS,
            output_dtype="float",
            truncation=True,
        )

        vector = response.embeddings[0]

    else:
        response = openai.Embedding.create(
            input=[query],
            model=config["model"],
            dimensions=DIMENSIONS,
        )

        vector = response["data"][0]["embedding"]

    elapsed_ms = (
        time.perf_counter() - started
    ) * 1000

    if len(vector) != DIMENSIONS:
        raise RuntimeError(
            f"{config['label']} returned "
            f"{len(vector)} dimensions; "
            f"expected {DIMENSIONS}."
        )

    return vector, elapsed_ms


def run_vector_search(
    collection,
    config,
    query_vector,
):
    vector_options = {
        "index": config["index_name"],
        "path": config["vector_field"],
        "queryVector": query_vector,
        "limit": TOP_K,
    }

    if EXACT_SEARCH:
        vector_options["exact"] = True
    else:
        vector_options["numCandidates"] = (
            NUM_CANDIDATES
        )

    pipeline = [
        {
            "$vectorSearch": vector_options
        },
        {
            "$project": {
                "_id": 0,
                "title": 1,
                "section": 1,
                "content": 1,
                "source": 1,
                "url": 1,
                "chunkNumber": 1,
                "score": {
                    "$meta": "vectorSearchScore"
                },
            }
        },
    ]

    started = time.perf_counter()

    try:
        documents = list(
            collection.aggregate(pipeline)
        )

    except OperationFailure as exc:
        raise RuntimeError(
            f"Vector search failed for "
            f"{config['label']}. Check Atlas index "
            f"'{config['index_name']}', field "
            f"'{config['vector_field']}', and "
            f"dimension {DIMENSIONS}. "
            f"Original error: {exc}"
        ) from exc

    elapsed_ms = (
        time.perf_counter() - started
    ) * 1000

    return documents, elapsed_ms


def find_expected_rank(
    documents,
    expected_title,
):
    expected_key = expected_title.casefold()

    for rank, document in enumerate(
        documents,
        start=1,
    ):
        title = str(
            document.get("title", "")
        ).strip()

        if title.casefold() == expected_key:
            return rank

    return None


def main():
    openai.api_key = OPENAI_KEY

    voyage_client = voyageai.Client(
        api_key=VOYAGE_KEY
    )

    mongo_client = MongoClient(
        MONGODB_URI,
        appname="ch-robinson-embedding-eval",
        serverSelectionTimeoutMS=15000,
    )

    mongo_client.admin.command("ping")

    collection = mongo_client[DB][COLL]

    validate_collection(collection)

    print(
        f"\nRunning {len(TEST_QUERIES)} queries "
        f"against {len(MODELS)} models."
    )

    print(
        f"Search mode: "
        f"{'exact' if EXACT_SEARCH else 'ANN'}"
    )

    result_rows = []
    query_score_rows = []

    for query_number, test in enumerate(
        TEST_QUERIES,
        start=1,
    ):
        query = test["query"]
        expected_title = test["expected_title"]
        category = test["category"]

        print(
            f"\n[{query_number:02d}/"
            f"{len(TEST_QUERIES)}] {query}"
        )

        # Alternate model order to reduce consistent
        # first-request timing bias.
        if query_number % 2 == 0:
            model_order = list(
                reversed(MODELS)
            )
        else:
            model_order = MODELS

        for config in model_order:
            query_vector, embedding_ms = (
                embed_query(
                    config,
                    query,
                    voyage_client,
                )
            )

            documents, search_ms = (
                run_vector_search(
                    collection,
                    config,
                    query_vector,
                )
            )

            expected_rank = find_expected_rank(
                documents,
                expected_title,
            )

            hit_at_1 = int(
                expected_rank is not None
                and expected_rank <= 1
            )

            hit_at_3 = int(
                expected_rank is not None
                and expected_rank <= 3
            )

            hit_at_5 = int(
                expected_rank is not None
                and expected_rank <= 5
            )

            reciprocal_rank = (
                1.0 / expected_rank
                if expected_rank is not None
                else 0.0
            )

            total_ms = (
                embedding_ms + search_ms
            )

            query_score_rows.append(
                {
                    "query": query,
                    "category": category,
                    "expected_title": expected_title,
                    "model": config["label"],
                    "expected_rank": (
                        expected_rank
                        if expected_rank is not None
                        else ""
                    ),
                    "hit_at_1": hit_at_1,
                    "hit_at_3": hit_at_3,
                    "hit_at_5": hit_at_5,
                    "reciprocal_rank": (
                        reciprocal_rank
                    ),
                    "embedding_ms": embedding_ms,
                    "search_ms": search_ms,
                    "total_ms": total_ms,
                }
            )

            for rank, document in enumerate(
                documents,
                start=1,
            ):
                title = str(
                    document.get("title", "")
                ).strip()

                result_rows.append(
                    {
                        "query": query,
                        "category": category,
                        "expected_title": (
                            expected_title
                        ),
                        "model": config["label"],
                        "rank": rank,
                        "is_expected": int(
                            title.casefold()
                            == expected_title.casefold()
                        ),
                        "title": title,
                        "section": document.get(
                            "section",
                            "",
                        ),
                        "content": document.get(
                            "content",
                            "",
                        ),
                        "chunkNumber": document.get(
                            "chunkNumber",
                            "",
                        ),
                        "score": document.get(
                            "score",
                            "",
                        ),
                        "embedding_ms": (
                            embedding_ms
                        ),
                        "search_ms": search_ms,
                        "total_ms": total_ms,
                    }
                )

            rank_display = (
                expected_rank
                if expected_rank is not None
                else "MISS"
            )

            top_title = (
                documents[0].get("title", "")
                if documents
                else "NO RESULTS"
            )

            print(
                f"  {config['label']:<24} "
                f"expected_rank={str(rank_display):<4} "
                f"top1={top_title} "
                f"embed={embedding_ms:.1f} ms "
                f"search={search_ms:.1f} ms"
            )

            if PRINT_TOP_RESULTS > 0:
                for rank, document in enumerate(
                    documents[
                        :PRINT_TOP_RESULTS
                    ],
                    start=1,
                ):
                    marker = (
                        "*"
                        if str(
                            document.get(
                                "title",
                                "",
                            )
                        ).casefold()
                        == expected_title.casefold()
                        else " "
                    )

                    print(
                        f"      {marker} "
                        f"{rank}. "
                        f"{document.get('title', '')} "
                        f"({document.get('score', 0):.4f})"
                    )

    results_frame = pd.DataFrame(
        result_rows
    )

    query_scores_frame = pd.DataFrame(
        query_score_rows
    )

    results_frame.to_csv(
        RESULTS_CSV,
        index=False,
        na_rep="",
    )

    query_scores_frame.to_csv(
        QUERY_SCORES_CSV,
        index=False,
        na_rep="",
    )

    summary_rows = []

    for config in MODELS:
        model_rows = query_scores_frame[
            query_scores_frame["model"]
            == config["label"]
        ]

        embedding_times = model_rows[
            "embedding_ms"
        ].tolist()

        search_times = model_rows[
            "search_ms"
        ].tolist()

        total_times = model_rows[
            "total_ms"
        ].tolist()

        summary_rows.append(
            {
                "model": config["label"],
                "queries": len(model_rows),
                "hit_at_1": model_rows[
                    "hit_at_1"
                ].mean(),
                "hit_at_3": model_rows[
                    "hit_at_3"
                ].mean(),
                "hit_at_5": model_rows[
                    "hit_at_5"
                ].mean(),
                "mrr": model_rows[
                    "reciprocal_rank"
                ].mean(),
                "embedding_median_ms": (
                    statistics.median(
                        embedding_times
                    )
                ),
                "embedding_p95_ms": (
                    percentile(
                        embedding_times,
                        0.95,
                    )
                ),
                "search_median_ms": (
                    statistics.median(
                        search_times
                    )
                ),
                "search_p95_ms": (
                    percentile(
                        search_times,
                        0.95,
                    )
                ),
                "total_median_ms": (
                    statistics.median(
                        total_times
                    )
                ),
                "total_p95_ms": (
                    percentile(
                        total_times,
                        0.95,
                    )
                ),
            }
        )

    summary_frame = pd.DataFrame(
        summary_rows
    )

    summary_frame.to_csv(
        SUMMARY_CSV,
        index=False,
        na_rep="",
    )

    print("\n================ SUMMARY ================")

    print(
        summary_frame[
            [
                "model",
                "queries",
                "hit_at_1",
                "hit_at_3",
                "hit_at_5",
                "mrr",
                "embedding_median_ms",
                "search_median_ms",
                "total_median_ms",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print("\nFiles written:")
    print(f"  {RESULTS_CSV}")
    print(f"  {QUERY_SCORES_CSV}")
    print(f"  {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
