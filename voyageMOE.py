import pymongo
from voyageai import Client as VoyageClient
import openai
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich import box
import time

# --- Console ---
console = Console()

# --- Mongo ---
client = pymongo.MongoClient("")
db = client.voyagenew
collection = db.demo_rag

# --- APIs ---
voyage = VoyageClient(api_key="VOYAGE_KEY_HERE")
openai.api_key = "OPENAI_KEY_HERE"

# Expert styling
EXPERT_COLORS = {
    "nutrition": "green",
    "diabetes": "blue",
    "exercise": "yellow",
    "mental_health": "magenta",
}

EXPERT_EMOJI = {
    "nutrition": "🥗",
    "diabetes": "🩺",
    "exercise": "💪",
    "mental_health": "🧠",
}

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

    if "heart" in ql:
        return ["nutrition", "exercise"]

    if any(w in ql for w in ["food", "diet", "nutrition", "cholesterol"]): return ["nutrition"]
    if any(w in ql for w in ["diabetes", "glucose", "blood sugar", "a1c"]): return ["diabetes"]
    if any(w in ql for w in ["stress", "anxiety", "sleep"]): return ["mental_health"]
    if any(w in ql for w in ["exercise", "workout", "walking", "activity"]): return ["exercise"]

    return ["nutrition", "exercise"]

# --- 3) Expert answer ---
def expert_answer(expert, question, progress, task_id):
    progress.update(task_id, description=f"[{EXPERT_COLORS[expert]}]{EXPERT_EMOJI[expert]} Retrieving from {expert}...")
    docs = retrieve(question, category=expert, k=3)
    
    # Debug: check what category docs were actually retrieved
    retrieved_categories = [d.get('category', 'unknown') for d in docs]
    console.log(f"[dim]{expert} retrieved categories: {retrieved_categories}[/dim]")
    
    progress.update(task_id, advance=50, description=f"[{EXPERT_COLORS[expert]}]{EXPERT_EMOJI[expert]} Generating {expert} answer...")
    ctx = "\n".join([f"- {d['text']}" for d in docs]) or "No context."

    system = {
        "nutrition": "You are a nutrition expert. Focus ONLY on dietary and food-related advice. Do not discuss exercise.",
        "diabetes": "You are a diabetes educator. Give clear, patient-friendly guidance about blood sugar management and diabetes care.",
        "exercise": "You are a fitness expert. Focus ONLY on physical activity, workouts, and exercise recommendations. Do NOT mention food or diet - only discuss movement, cardio, strength training, and physical activity.",
        "mental_health": "You are a mental health coach. Focus ONLY on stress management, sleep, and mental wellness strategies. Do not discuss food or exercise.",
    }[expert]

    user_prompt = {
        "nutrition": f"Q: {question}\n\nContext:\n{ctx}\n\nProvide dietary and nutrition advice only:",
        "diabetes": f"Q: {question}\n\nContext:\n{ctx}\n\nProvide diabetes management advice:",
        "exercise": f"Q: {question}\n\nContext:\n{ctx}\n\nProvide ONLY exercise and physical activity recommendations (no food/diet advice):",
        "mental_health": f"Q: {question}\n\nContext:\n{ctx}\n\nProvide mental health and stress management advice:",
    }[expert]

    r = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=180
    )
    
    progress.update(task_id, advance=50, description=f"[{EXPERT_COLORS[expert]}]{EXPERT_EMOJI[expert]} {expert} complete")
    return docs, r.choices[0].message["content"].strip()

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

# --- Display retrieved docs with scores ---
def show_retrieval_results(expert, docs):
    color = EXPERT_COLORS[expert]
    emoji = EXPERT_EMOJI[expert]
    
    table = Table(title=f"{emoji} {expert.upper()} - Retrieved Documents", 
                  box=box.ROUNDED, 
                  title_style=f"bold {color}")
    table.add_column("Score", style=color, width=20)
    table.add_column("Text", style="white")
    
    for doc in docs:
        score = doc.get('score', 0)
        score_bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        table.add_row(f"{score_bar} {score:.3f}", doc['text'][:80] + "...")
    
    console.print(table)
    console.print()

# --- Main MoE pipeline ---
def run_moe_demo(question):
    console.clear()
    console.print(Panel.fit(
        f"[bold cyan]MongoDB Voyage-4-Large MoE Demo[/bold cyan]\n[yellow]Question:[/yellow] {question}",
        border_style="cyan"
    ))
    console.print()
    
    # Stage 1: Routing
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]🧭 Routing query to experts...", total=100)
        experts = route(question)
        progress.update(task, advance=100, description="[cyan]🧭 Routing complete")
        time.sleep(0.3)
    
    console.print(f"[bold]Selected Experts:[/bold] {', '.join([f'{EXPERT_EMOJI[e]} {e}' for e in experts])}")
    console.print()
    
    # Stage 2: Expert retrieval and answering
    answers = {}
    all_docs = {}
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        tasks = {e: progress.add_task(f"[{EXPERT_COLORS[e]}]{EXPERT_EMOJI[e]} {e}", total=100) for e in experts}
        
        for expert in experts:
            docs, answer = expert_answer(expert, question, progress, tasks[expert])
            answers[expert] = answer
            all_docs[expert] = docs
    
    console.print()
    
    # Show retrieval results
    for expert in experts:
        show_retrieval_results(expert, all_docs[expert])
    
    # Show individual expert answers
    for expert, answer in answers.items():
        color = EXPERT_COLORS[expert]
        emoji = EXPERT_EMOJI[expert]
        console.print(Panel(
            answer,
            title=f"{emoji} [bold {color}]{expert.upper()} Expert[/bold {color}]",
            border_style=color,
            box=box.ROUNDED
        ))
        console.print()
    
    # Stage 3: Aggregation
    if len(answers) > 1:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[magenta]🔮 Aggregating expert answers...", total=100)
            final = aggregate(question, answers)
            progress.update(task, advance=100, description="[magenta]🔮 Aggregation complete")
            time.sleep(0.3)
    else:
        final = list(answers.values())[0]
    
    console.print()
    console.print(Panel(
        f"[bold white]{final}[/bold white]",
        title="✨ [bold green]FINAL ANSWER[/bold green] ✨",
        border_style="green",
        box=box.DOUBLE
    ))

# --- Run ---
if __name__ == "__main__":
    q = "What are some healthy foods for heart health?"
    run_moe_demo(q)
