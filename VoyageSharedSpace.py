import pymongo
from voyageai import Client as VoyageClient
import openai
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.table import Table
from rich import box
import time

console = Console()

# --- Configuration ---
MONGO_URI = ""
VOYAGE_API_KEY = "VOYAGE_KEY_HERE"
OPENAI_API_KEY = "OPENAI_KEY_HERE"

# Document embedding model (used once, upfront)
DOCUMENT_MODEL = "voyage-4-large"

# Query embedding models (for comparison)
QUERY_MODELS = ["voyage-4-large", "voyage-4-lite", "voyage-4-nano"]

# Pricing (per 1M tokens) - approximate
MODEL_COSTS = {
    "voyage-4-large": 0.12,
    "voyage-4-lite": 0.06,
    "voyage-4-nano": 0.02,
}

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

# --- Setup ---
client = pymongo.MongoClient(MONGO_URI)
db = client.voyagenew
collection = db.demo_rag
voyage = VoyageClient(api_key=VOYAGE_API_KEY)
openai.api_key = OPENAI_API_KEY


def retrieve(query_embedding, category=None, k=3):
    """Retrieve docs using pre-computed query embedding."""
    stage = {
        "$vectorSearch": {
            "index": "vector_index",
            "path": "embedding",
            "queryVector": query_embedding,
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


def route(q):
    """Route query to appropriate experts."""
    ql = q.lower()
    if "heart" in ql:
        return ["nutrition", "exercise"]
    if any(w in ql for w in ["food", "diet", "nutrition", "cholesterol"]): 
        return ["nutrition"]
    if any(w in ql for w in ["diabetes", "glucose", "blood sugar", "a1c"]): 
        return ["diabetes"]
    if any(w in ql for w in ["stress", "anxiety", "sleep"]): 
        return ["mental_health"]
    if any(w in ql for w in ["exercise", "workout", "walking", "activity"]): 
        return ["exercise"]
    return ["nutrition", "exercise"]


def expert_answer(expert, question, query_embedding, progress, task_id):
    """Generate expert answer using pre-computed query embedding."""
    progress.update(task_id, description=f"[{EXPERT_COLORS[expert]}]{EXPERT_EMOJI[expert]} Retrieving from {expert}...")
    docs = retrieve(query_embedding, category=expert, k=3)
    
    progress.update(task_id, advance=50, description=f"[{EXPERT_COLORS[expert]}]{EXPERT_EMOJI[expert]} Generating {expert} answer...")
    ctx = "\n".join([f"- {d['text']}" for d in docs]) or "No context."

    system = {
        "nutrition": "You are a nutrition expert. Focus ONLY on dietary and food-related advice.",
        "diabetes": "You are a diabetes educator. Give clear, patient-friendly guidance.",
        "exercise": "You are a fitness expert. Focus ONLY on physical activity and exercise recommendations.",
        "mental_health": "You are a mental health coach. Focus ONLY on stress management and mental wellness.",
    }[expert]

    user_prompt = f"Q: {question}\n\nContext:\n{ctx}\n\nProvide concise advice:"

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


def aggregate(question, answers):
    """Combine expert answers into final response."""
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


def show_embedding_comparison(question, query_models):
    """Show asymmetric retrieval: compare different query models against large document embeddings."""
    console.print(Panel.fit(
        "[bold cyan]🔬 Voyage 4 Shared Embedding Space Demo[/bold cyan]\n"
        f"[yellow]Documents embedded with:[/yellow] {DOCUMENT_MODEL}\n"
        f"[yellow]Testing query models:[/yellow] {', '.join(query_models)}",
        border_style="cyan"
    ))
    console.print()

    # Create comparison table
    comparison_table = Table(
        title="⚡ Asymmetric Retrieval Performance", 
        box=box.ROUNDED,
        title_style="bold cyan"
    )
    comparison_table.add_column("Query Model", style="cyan", width=18)
    comparison_table.add_column("Embed Time", style="yellow", width=12)
    comparison_table.add_column("Top Score", style="green", width=12)
    comparison_table.add_column("Cost/1M", style="magenta", width=12)
    comparison_table.add_column("Compatible?", style="bold green", width=12)

    results = {}
    
    for model in query_models:
        console.print(f"[dim]Testing {model}...[/dim]")
        
        # Time the embedding
        start = time.time()
        qvec = voyage.embed([question], model=model).embeddings[0]
        embed_time = time.time() - start
        
        # Retrieve with this query embedding against large document embeddings
        docs = retrieve(qvec, k=3)
        top_score = docs[0]['score'] if docs else 0.0
        
        results[model] = {
            "embedding": qvec,
            "time": embed_time,
            "docs": docs,
            "top_score": top_score
        }
        
        # Add to table
        cost = f"${MODEL_COSTS[model]:.2f}"
        comparison_table.add_row(
            model.replace("voyage-4-", "v4-"),
            f"{embed_time*1000:.1f}ms",
            f"{top_score:.4f}",
            cost,
            "✅ Yes"
        )
    
    console.print(comparison_table)
    console.print()
    
    # Show that results are similar despite different models
    console.print("[bold]📊 Retrieved Documents Comparison:[/bold]\n")
    
    for model in query_models:
        docs = results[model]["docs"]
        color = "green" if model == DOCUMENT_MODEL else "cyan"
        
        table = Table(
            title=f"{model} results", 
            box=box.SIMPLE,
            title_style=color,
            show_header=False
        )
        table.add_column("", style=color, width=10)
        table.add_column("", style="white")
        
        for i, doc in enumerate(docs[:3], 1):
            score = doc.get('score', 0)
            table.add_row(f"#{i} ({score:.3f})", doc['text'][:60] + "...")
        
        console.print(table)
    
    console.print()
    
    # Show cost savings
    large_cost = MODEL_COSTS[DOCUMENT_MODEL]
    lite_cost = MODEL_COSTS["voyage-4-lite"]
    nano_cost = MODEL_COSTS["voyage-4-nano"]
    
    savings_lite = ((large_cost - lite_cost) / large_cost) * 100
    savings_nano = ((large_cost - nano_cost) / large_cost) * 100
    
    console.print(Panel(
        f"[bold green]💰 Cost Savings with Asymmetric Retrieval[/bold green]\n\n"
        f"Documents: [cyan]{DOCUMENT_MODEL}[/cyan] (embedded once)\n"
        f"Queries: [cyan]voyage-4-lite[/cyan] or [cyan]voyage-4-nano[/cyan] (per request)\n\n"
        f"Savings vs all-large:\n"
        f"  • voyage-4-lite: [yellow]{savings_lite:.0f}% cheaper[/yellow] per query\n"
        f"  • voyage-4-nano: [yellow]{savings_nano:.0f}% cheaper[/yellow] per query\n\n"
        f"Example: 1M queries/month\n"
        f"  • All large: [red]${large_cost*1000:.0f}/month[/red]\n"
        f"  • Asymmetric (nano): [green]${nano_cost*1000:.0f}/month[/green] [bold]({savings_nano:.0f}% savings!)[/bold]",
        border_style="green",
        box=box.ROUNDED
    ))
    
    return results


def run_asymmetric_moe_demo(question, query_model="voyage-4-lite"):
    """Run MoE demo with specified query model (asymmetric retrieval)."""
    console.print(Panel.fit(
        f"[bold cyan]🎯 MoE with Asymmetric Retrieval[/bold cyan]\n"
        f"[yellow]Documents:[/yellow] {DOCUMENT_MODEL} | [yellow]Queries:[/yellow] {query_model}\n"
        f"[yellow]Question:[/yellow] {question}",
        border_style="cyan"
    ))
    console.print()
    
    # Embed query with chosen model
    console.print(f"[dim]Embedding query with {query_model}...[/dim]")
    start = time.time()
    qvec = voyage.embed([question], model=query_model).embeddings[0]
    embed_time = time.time() - start
    console.print(f"[green]✓[/green] Query embedded in {embed_time*1000:.1f}ms\n")
    
    # Route to experts
    experts = route(question)
    console.print(f"[bold]Selected Experts:[/bold] {', '.join([f'{EXPERT_EMOJI[e]} {e}' for e in experts])}\n")
    
    # Get expert answers
    answers = {}
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        tasks = {e: progress.add_task(f"[{EXPERT_COLORS[e]}]{EXPERT_EMOJI[e]} {e}", total=100) for e in experts}
        
        for expert in experts:
            docs, answer = expert_answer(expert, question, qvec, progress, tasks[expert])
            answers[expert] = answer
    
    console.print()
    
    # Show expert answers
    for expert, answer in answers.items():
        color = EXPERT_COLORS[expert]
        emoji = EXPERT_EMOJI[expert]
        console.print(Panel(
            answer,
            title=f"{emoji} [bold {color}]{expert.upper()}[/bold {color}]",
            border_style=color,
            box=box.ROUNDED
        ))
        console.print()
    
    # Aggregate
    if len(answers) > 1:
        final = aggregate(question, answers)
    else:
        final = list(answers.values())[0]
    
    console.print(Panel(
        f"[bold white]{final}[/bold white]",
        title="✨ [bold green]FINAL ANSWER[/bold green] ✨",
        border_style="green",
        box=box.DOUBLE
    ))


if __name__ == "__main__":
    question = "What are some healthy foods for heart health?"
    
    # Demo 1: Show shared embedding space compatibility
    console.rule("[bold cyan]PART 1: Shared Embedding Space Demo[/bold cyan]")
    console.print()
    show_embedding_comparison(question, QUERY_MODELS)
    
    console.print("\n")
    console.rule("[bold cyan]PART 2: MoE with Asymmetric Retrieval[/bold cyan]")
    console.print()
    
    # Demo 2: Run MoE with asymmetric retrieval (lite model for queries)
    run_asymmetric_moe_demo(question, query_model="voyage-4-lite")
root@factory:/project/workspace# 
root@factory:/project/workspace# 
root@factory:/project/workspace# 
root@factory:/project/workspace# 
root@factory:/project/workspace# ls
voyage_asymmetric_moe_demo.py
root@factory:/project/workspace# cat voyage_asymmetric_moe_demo.py
import pymongo
from voyageai import Client as VoyageClient
import openai
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.table import Table
from rich import box
import time

console = Console()

# --- Configuration ---
MONGO_URI = ""
VOYAGE_API_KEY = "VOYAGE_KEY_HERE"
OPENAI_API_KEY = "OPENAI_KEY_HERE"

# Document embedding model (used once, upfront)
DOCUMENT_MODEL = "voyage-4-large"

# Query embedding models (for comparison)
QUERY_MODELS = ["voyage-4-large", "voyage-4-lite", "voyage-4-nano"]

# Pricing (per 1M tokens) - approximate
MODEL_COSTS = {
    "voyage-4-large": 0.12,
    "voyage-4-lite": 0.06,
    "voyage-4-nano": 0.02,
}

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

# --- Setup ---
client = pymongo.MongoClient(MONGO_URI)
db = client.voyagenew
collection = db.demo_rag
voyage = VoyageClient(api_key=VOYAGE_API_KEY)
openai.api_key = OPENAI_API_KEY


def expert_qvec(question, expert, model):
    """Generate expert-contextualized query embedding."""
    q = f"Question: {question}\nExpert domain: {expert}"
    return voyage.embed([q], model=model).embeddings[0]


def retrieve(query_embedding, category=None, k=3):
    """Retrieve docs using pre-computed query embedding."""
    stage = {
        "$vectorSearch": {
            "index": "vector_index",
            "path": "embedding",
            "queryVector": query_embedding,
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


def route(q):
    """Route query to appropriate experts."""
    ql = q.lower()
    if "heart" in ql:
        return ["nutrition", "exercise"]
    if any(w in ql for w in ["food", "diet", "nutrition", "cholesterol"]): 
        return ["nutrition"]
    if any(w in ql for w in ["diabetes", "glucose", "blood sugar", "a1c"]): 
        return ["diabetes"]
    if any(w in ql for w in ["stress", "anxiety", "sleep"]): 
        return ["mental_health"]
    if any(w in ql for w in ["exercise", "workout", "walking", "activity"]): 
        return ["exercise"]
    return ["nutrition", "exercise"]


def expert_answer(expert, question, query_model, progress, task_id):
    """Generate expert answer using expert-contextualized query embedding."""
    progress.update(task_id, description=f"[{EXPERT_COLORS[expert]}]{EXPERT_EMOJI[expert]} Embedding query for {expert}...")
    qvec = expert_qvec(question, expert, query_model)
    
    progress.update(task_id, description=f"[{EXPERT_COLORS[expert]}]{EXPERT_EMOJI[expert]} Retrieving from {expert}...")
    docs = retrieve(qvec, category=expert, k=3)
    
    progress.update(task_id, advance=50, description=f"[{EXPERT_COLORS[expert]}]{EXPERT_EMOJI[expert]} Generating {expert} answer...")
    ctx = "\n".join([f"- {d['text']}" for d in docs]) or "No context."

    system = {
        "nutrition": "You are a nutrition expert. Focus ONLY on dietary and food-related advice.",
        "diabetes": "You are a diabetes educator. Give clear, patient-friendly guidance.",
        "exercise": "You are a fitness expert. Focus ONLY on physical activity and exercise recommendations.",
        "mental_health": "You are a mental health coach. Focus ONLY on stress management and mental wellness.",
    }[expert]

    user_prompt = f"Q: {question}\n\nContext:\n{ctx}\n\nProvide concise advice:"

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


def aggregate(question, answers):
    """Combine expert answers into final response."""
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


def show_embedding_comparison(question, query_models):
    """Show asymmetric retrieval: compare different query models against large document embeddings."""
    console.print(Panel.fit(
        "[bold cyan]🔬 Voyage 4 Shared Embedding Space Demo[/bold cyan]\n"
        f"[yellow]Documents embedded with:[/yellow] {DOCUMENT_MODEL}\n"
        f"[yellow]Testing query models:[/yellow] {', '.join(query_models)}",
        border_style="cyan"
    ))
    console.print()

    # Create comparison table
    comparison_table = Table(
        title="⚡ Asymmetric Retrieval Performance", 
        box=box.ROUNDED,
        title_style="bold cyan"
    )
    comparison_table.add_column("Query Model", style="cyan", width=18)
    comparison_table.add_column("Embed Time", style="yellow", width=12)
    comparison_table.add_column("Top Score", style="green", width=12)
    comparison_table.add_column("Cost/1M", style="magenta", width=12)
    comparison_table.add_column("Compatible?", style="bold green", width=12)

    results = {}
    
    for model in query_models:
        console.print(f"[dim]Testing {model}...[/dim]")
        
        # Time the embedding
        start = time.time()
        qvec = voyage.embed([question], model=model).embeddings[0]
        embed_time = time.time() - start
        
        # Retrieve with this query embedding against large document embeddings
        docs = retrieve(qvec, k=3)
        top_score = docs[0]['score'] if docs else 0.0
        
        results[model] = {
            "embedding": qvec,
            "time": embed_time,
            "docs": docs,
            "top_score": top_score
        }
        
        # Add to table
        cost = f"${MODEL_COSTS[model]:.2f}"
        comparison_table.add_row(
            model.replace("voyage-4-", "v4-"),
            f"{embed_time*1000:.1f}ms",
            f"{top_score:.4f}",
            cost,
            "✅ Yes"
        )
    
    console.print(comparison_table)
    console.print()
    
    # Show that results are similar despite different models
    console.print("[bold]📊 Retrieved Documents Comparison:[/bold]\n")
    
    for model in query_models:
        docs = results[model]["docs"]
        color = "green" if model == DOCUMENT_MODEL else "cyan"
        
        table = Table(
            title=f"{model} results", 
            box=box.SIMPLE,
            title_style=color,
            show_header=False
        )
        table.add_column("", style=color, width=10)
        table.add_column("", style="white")
        
        for i, doc in enumerate(docs[:3], 1):
            score = doc.get('score', 0)
            table.add_row(f"#{i} ({score:.3f})", doc['text'][:60] + "...")
        
        console.print(table)
    
    console.print()
    
    # Show cost savings
    large_cost = MODEL_COSTS[DOCUMENT_MODEL]
    lite_cost = MODEL_COSTS["voyage-4-lite"]
    nano_cost = MODEL_COSTS["voyage-4-nano"]
    
    savings_lite = ((large_cost - lite_cost) / large_cost) * 100
    savings_nano = ((large_cost - nano_cost) / large_cost) * 100
    
    console.print(Panel(
        f"[bold green]💰 Cost Savings with Asymmetric Retrieval[/bold green]\n\n"
        f"Documents: [cyan]{DOCUMENT_MODEL}[/cyan] (embedded once)\n"
        f"Queries: [cyan]voyage-4-lite[/cyan] or [cyan]voyage-4-nano[/cyan] (per request)\n\n"
        f"Savings vs all-large:\n"
        f"  • voyage-4-lite: [yellow]{savings_lite:.0f}% cheaper[/yellow] per query\n"
        f"  • voyage-4-nano: [yellow]{savings_nano:.0f}% cheaper[/yellow] per query\n\n"
        f"Example: 1M queries/month\n"
        f"  • All large: [red]${large_cost*1000:.0f}/month[/red]\n"
        f"  • Asymmetric (nano): [green]${nano_cost*1000:.0f}/month[/green] [bold]({savings_nano:.0f}% savings!)[/bold]",
        border_style="green",
        box=box.ROUNDED
    ))
    
    return results


def run_asymmetric_moe_demo(question, query_model="voyage-4-lite"):
    """Run MoE demo with specified query model (asymmetric retrieval)."""
    console.print(Panel.fit(
        f"[bold cyan]🎯 MoE with Asymmetric Retrieval[/bold cyan]\n"
        f"[yellow]Documents:[/yellow] {DOCUMENT_MODEL} | [yellow]Queries:[/yellow] {query_model}\n"
        f"[yellow]Question:[/yellow] {question}",
        border_style="cyan"
    ))
    console.print()
    
    # Embed query with chosen model
    console.print(f"[dim]Embedding query with {query_model}...[/dim]")
    start = time.time()
    qvec = voyage.embed([question], model=query_model).embeddings[0]
    embed_time = time.time() - start
    console.print(f"[green]✓[/green] Query embedded in {embed_time*1000:.1f}ms\n")
    
    # Route to experts
    experts = route(question)
    console.print(f"[bold]Selected Experts:[/bold] {', '.join([f'{EXPERT_EMOJI[e]} {e}' for e in experts])}\n")
    
    # Get expert answers
    answers = {}
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        tasks = {e: progress.add_task(f"[{EXPERT_COLORS[e]}]{EXPERT_EMOJI[e]} {e}", total=100) for e in experts}
        
        for expert in experts:
            docs, answer = expert_answer(expert, question, query_model, progress, tasks[expert])
            answers[expert] = answer
    
    console.print()
    
    # Show expert answers
    for expert, answer in answers.items():
        color = EXPERT_COLORS[expert]
        emoji = EXPERT_EMOJI[expert]
        console.print(Panel(
            answer,
            title=f"{emoji} [bold {color}]{expert.upper()}[/bold {color}]",
            border_style=color,
            box=box.ROUNDED
        ))
        console.print()
    
    # Aggregate
    if len(answers) > 1:
        final = aggregate(question, answers)
    else:
        final = list(answers.values())[0]
    
    console.print(Panel(
        f"[bold white]{final}[/bold white]",
        title="✨ [bold green]FINAL ANSWER[/bold green] ✨",
        border_style="green",
        box=box.DOUBLE
    ))


if __name__ == "__main__":
    question = "What are some healthy foods for heart health?"
    
    # Demo 1: Show shared embedding space compatibility
    console.rule("[bold cyan]PART 1: Shared Embedding Space Demo[/bold cyan]")
    console.print()
    show_embedding_comparison(question, QUERY_MODELS)
    
    console.print("\n")
    console.rule("[bold cyan]PART 2: MoE with Asymmetric Retrieval[/bold cyan]")
    console.print()
    
    # Demo 2: Run MoE with asymmetric retrieval (lite model for queries)
    run_asymmetric_moe_demo(question, query_model="voyage-4-lite")
