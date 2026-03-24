import ast
import json
import math
import operator
import os
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

#Stores
chunk_store: dict = {}
session_store: dict = {}

#OpenAI
_openai_client: Optional[AsyncOpenAI] = None

def get_openai_client() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")
        _openai_client = AsyncOpenAI(api_key=api_key)
    return _openai_client

#RAG
def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    if not text or not text.strip():
        return []
    words = text.split()
    if len(words) <= chunk_size:
        return [text.strip()]
    step = chunk_size - overlap
    if step <= 0:
        raise ValueError(f"overlap ({overlap}) must be less than chunk_size ({chunk_size})")
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += step
    return chunks

async def embed_text(text: str, model: str = "text-embedding-3-small") -> list[float]:
    client = get_openai_client()
    response = await client.embeddings.create(input=text.replace("\n", " "), model=model)
    return response.data[0].embedding

def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a * a for a in vec_a))
    mag_b = math.sqrt(sum(b * b for b in vec_b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot_product / (mag_a * mag_b)

def cosine_similarity_search(query_embedding: list[float], store: dict, k: int = 3) -> list[tuple[str, float]]:
    scores = [(cid, cosine_similarity(query_embedding, data["embedding"])) for cid, data in store.items()]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]

RAG_SYSTEM_PROMPT = """You are a question-answering assistant.
Only answer based on the provided context passages below.
If the answer is not found in the context, say:
"I don't have enough information in the provided documents to answer that question."
Don't use any outside knowledge. Cite context as [chunk_id] inline where relevant."""

def build_grounded_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    lines = ["CONTEXT:"] + [f"[{c['chunk_id']}]: {c['text']}" for c in retrieved_chunks]
    return "\n".join(lines) + f"\n\nQUESTION: {question}"

async def generate_grounded_answer(question: str, retrieved_chunks: list[dict], history: list[dict], model: str = "gpt-4o-mini") -> str:
    client = get_openai_client()
    messages = [{"role": "system", "content": RAG_SYSTEM_PROMPT}] + history + [{"role": "user", "content": build_grounded_prompt(question, retrieved_chunks)}]
    response = await client.chat.completions.create(model=model, messages=messages, temperature=0.2)
    return response.choices[0].message.content

#Calculator Tool
SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

def safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in SAFE_OPERATORS:
        return SAFE_OPERATORS[type(node.op)](safe_eval(node.left), safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in SAFE_OPERATORS:
        return SAFE_OPERATORS[type(node.op)](safe_eval(node.operand))
    raise ValueError("Unsafe or unsupported expression")

def calculator_tool(expression: str) -> str:
    try:
        result = safe_eval(ast.parse(expression.strip(), mode="eval").body)
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        return str(result)
    except ZeroDivisionError:
        return "Error: division by zero"
    except Exception:
        return f"Error: invalid expression '{expression}'"

#Knowledge Base Tool
async def knowledge_base_tool(query: str, k: int = 3) -> str:
    if not chunk_store:
        return "No documents have been ingested into the knowledge base yet."
    top_results = cosine_similarity_search(await embed_text(query), chunk_store, k=k)
    if not top_results:
        return "No relevant chunks found."
    return "\n".join(f"[{cid}] (score={round(score, 4)}): {chunk_store[cid]['text']}" for cid, score in top_results)

#Tool Schemas
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate an arithmetic expression (+, -, *, /, **). Use for any math calculation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "e.g. '25 * 48' or '(10 + 5) / 3'"}
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "knowledge_base",
            "description": "Search ingested documents for relevant information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "k": {"type": "integer", "description": "Number of chunks to retrieve", "default": 3},
                },
                "required": ["query"],
            },
        },
    },
]

#Agent Loop
AGENT_SYSTEM_PROMPT = """You are a helpful assistant with access to tools.
Use the calculator tool for arithmetic. Use the knowledge_base tool for document questions.
For simple queries that need neither, answer directly."""

async def run_agent(query: str, history: list[dict], model: str = "gpt-4o-mini", max_steps: int = 5) -> tuple[str, list[dict]]:
    client = get_openai_client()
    steps = []
    messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}] + history + [{"role": "user", "content": query}]

    for _ in range(max_steps):
        response = await client.chat.completions.create(model=model, messages=messages, tools=TOOL_SCHEMAS, tool_choice="auto", temperature=0.2)
        choice = response.choices[0]
        messages.append(choice.message)

        if choice.finish_reason == "stop":
            return choice.message.content, steps

        if choice.finish_reason == "tool_calls":
            for tool_call in choice.message.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)

                if name == "calculator":
                    expr = args.get("expression", "")
                    output = calculator_tool(expr)
                    steps.append({"action": "calculator", "input": expr, "output": output})
                elif name == "knowledge_base":
                    kb_query = args.get("query", "")
                    output = await knowledge_base_tool(kb_query, k=args.get("k", 3))
                    steps.append({"action": "knowledge_base", "input": kb_query, "output": output})
                else:
                    output = f"Error: unknown tool '{name}'"
                    steps.append({"action": name, "input": str(args), "output": output})

                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": output})

    return "Agent reached maximum steps without a final answer.", steps

#Models
class IngestRequest(BaseModel):
    doc_id: str = Field(..., example="intro_cs_notes")
    text: str = Field(..., example="Long document text goes here...")

class IngestResponse(BaseModel):
    doc_id: str
    chunks_added: int

class SearchResult(BaseModel):
    chunk_id: str
    score: float
    text: str

class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]

class QARequest(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str = Field(..., example="What is recursion?")
    k: int = Field(default=4, ge=1, le=20)

class Citation(BaseModel):
    chunk_id: str
    score: float

class QAResponse(BaseModel):
    answer: str
    citations: list[Citation]
    turn_count: int

class AgentRequest(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query: str = Field(..., example="What is 25 * 48 and explain recursion?")

class AgentStep(BaseModel):
    action: str
    input: str
    output: str

class AgentResponse(BaseModel):
    answer: str
    steps: list[AgentStep]
    turn_count: int

#App & Endpoints
app = FastAPI(title="LLM Agent with Tool Use", version="2.0.0")

@app.get("/")
def root():
    return {"service": "LLM Agent with Tool Use", "endpoints": ["/ingest", "/search", "/qa", "/agent"], "chunks_stored": len(chunk_store), "sessions_active": len(session_store)}

@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(request: IngestRequest):
    chunks = chunk_text(request.text, chunk_size=300, overlap=50)
    for idx, text_content in enumerate(chunks):
        chunk_id = f"{request.doc_id}#{idx}"
        chunk_store[chunk_id] = {"doc_id": request.doc_id, "chunk_index": idx, "text": text_content, "embedding": await embed_text(text_content)}
    return IngestResponse(doc_id=request.doc_id, chunks_added=len(chunks))

@app.get("/search", response_model=SearchResponse)
async def search(query: str = Query(...), k: int = Query(default=3, ge=1, le=20)):
    if not chunk_store:
        raise HTTPException(status_code=404, detail="No documents ingested yet.")
    top_results = cosine_similarity_search(await embed_text(query), chunk_store, k=k)
    results = [SearchResult(chunk_id=cid, score=round(score, 4), text=chunk_store[cid]["text"]) for cid, score in top_results]
    return SearchResponse(query=query, results=results)

@app.post("/qa", response_model=QAResponse)
async def question_answer(request: QARequest):
    if not chunk_store:
        raise HTTPException(status_code=404, detail="No documents ingested yet.")
    top_results = cosine_similarity_search(await embed_text(request.question), chunk_store, k=request.k)
    retrieved_chunks = [{"chunk_id": cid, "score": score, "text": chunk_store[cid]["text"]} for cid, score in top_results]
    if request.session_id not in session_store:
        session_store[request.session_id] = []
    history = session_store[request.session_id]
    answer = await generate_grounded_answer(request.question, retrieved_chunks, history)
    history.append({"role": "user", "content": request.question})
    history.append({"role": "assistant", "content": answer})
    citations = [Citation(chunk_id=c["chunk_id"], score=round(c["score"], 4)) for c in retrieved_chunks]
    return QAResponse(answer=answer, citations=citations, turn_count=len(history) // 2)

@app.post("/agent", response_model=AgentResponse)
async def agent_endpoint(request: AgentRequest):
    if request.session_id not in session_store:
        session_store[request.session_id] = []
    history = session_store[request.session_id]
    answer, steps = await run_agent(query=request.query, history=history)
    history.append({"role": "user", "content": request.query})
    history.append({"role": "assistant", "content": answer})
    return AgentResponse(answer=answer, steps=[AgentStep(**s) for s in steps], turn_count=len(history) // 2)