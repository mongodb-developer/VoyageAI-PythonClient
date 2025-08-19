import textwrap
import voyageai

# Initialize client with your demo key directly
vo = voyageai.Client(api_key=" voyage key here ")

# --- Demo corpus: tech/productivity tools ---
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

def run_rerank(query, model="rerank-2.5", top_k=3):
    resp = vo.rerank(
        query=query,
        documents=documents,
        model=model,
        top_k=top_k
    )
    return resp.results

def show(results, title):
    print(f"\n=== {title} ===")
    for i, r in enumerate(results, 1):
        snippet = textwrap.shorten(r.document, width=100, placeholder="…")
        print(f"{i}. [score={r.relevance_score:.4f}] {snippet}")

# --- Run: rerank-2.5 baseline vs instruction ---
base_25 = run_rerank(no_instruction_query, model="rerank-2.5", top_k=3)
inst_25 = run_rerank(instruction_query, model="rerank-2.5", top_k=3)

show(base_25, "rerank-2.5  |  Baseline (no instruction)")
show(inst_25, "rerank-2.5  |  With instruction")

# --- Optional: compare the lighter model ---
base_lite = run_rerank(no_instruction_query, model="rerank-2.5-lite", top_k=3)
inst_lite = run_rerank(instruction_query, model="rerank-2.5-lite", top_k=3)

show(base_lite, "rerank-2.5-lite  |  Baseline (no instruction)")
show(inst_lite, "rerank-2.5-lite  |  With instruction")
