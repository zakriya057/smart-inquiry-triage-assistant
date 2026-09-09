"""
Smart Inquiry Triage Assistant — LangGraph Triage Pipeline.

This module implements the core triage backend logic using a LangGraph StateGraph:
- State: TriageState (query, top_k, confidence_threshold, retrieved_cases, final_output)
- Node 1: retrieve_node (similarity search over ChromaDB historical cases)
- Node 2: reasoning_node (canonical taxonomy injection, Gemini reasoning, confidence scoring)
- Workflow: START -> retrieve_node -> reasoning_node -> END
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, TypedDict

from dotenv import load_dotenv
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Resolve project paths
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
TAXONOMY_PATH = BASE_DIR / "data" / "taxonomy.json"
CHROMA_DIR = BASE_DIR / "chroma_db" if (BASE_DIR / "chroma_db").exists() else BASE_DIR / "chroma"


def load_environment() -> str:
    """Load environment variables from .env file and return Google API key."""
    load_dotenv(dotenv_path=ENV_PATH)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key.strip() == "":
        logger.error(
            "GOOGLE_API_KEY is not set or empty in .env. "
            "Please provide a valid key before running triage."
        )
        sys.exit(1)
    return api_key


def get_embeddings_model(api_key: str) -> GoogleGenerativeAIEmbeddings:
    """
    Initialize GoogleGenerativeAIEmbeddings with fallback support.
    Attempts models/text-embedding-004 first, falls back to models/gemini-embedding-001.
    """
    model_name = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")
    try:
        embeddings_model = GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=api_key
        )
        embeddings_model.embed_query("probe")
        return embeddings_model
    except Exception as e:
        logger.warning(
            f"Embedding model '{model_name}' unavailable ({e}). "
            "Falling back to 'models/gemini-embedding-001'..."
        )
        return GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=api_key
        )


def get_vector_store(embeddings_model: GoogleGenerativeAIEmbeddings) -> Chroma:
    """Connect to the persistent local ChromaDB collection."""
    possible_dirs = [
        BASE_DIR / "chroma_db",
        BASE_DIR / "chroma",
        Path.home() / "chroma_db",
        Path.home() / "chroma",
    ]
    target_dir = None
    for p in possible_dirs:
        if p.exists():
            target_dir = p
            break

    if target_dir is None:
        target_dir = BASE_DIR / "chroma_db"

    client = chromadb.PersistentClient(path=str(target_dir))
    vector_store = Chroma(
        client=client,
        collection_name="triage_cases",
        embedding_function=embeddings_model
    )
    return vector_store


def load_taxonomy() -> dict:
    """Load canonical taxonomy definitions from data/taxonomy.json."""
    if not TAXONOMY_PATH.exists():
        logger.error(f"Taxonomy file not found at {TAXONOMY_PATH}")
        sys.exit(1)

    with open(TAXONOMY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_llm(api_key: str) -> ChatGoogleGenerativeAI:
    """
    Initialize ChatGoogleGenerativeAI with fallback support.
    Attempts gemini-1.5-flash (temperature=0) first, falls back to gemini-3.6-flash, then gemini-flash-latest.
    """
    preferred_model = os.getenv("LLM_MODEL", "gemini-1.5-flash")
    candidates = [preferred_model, "gemini-3.6-flash", "gemini-flash-latest"]
    seen = set()
    model_list = [m for m in candidates if not (m in seen or seen.add(m))]

    for model_name in model_list:
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=0,
                google_api_key=api_key
            )
            llm.invoke("probe")
            return llm
        except Exception as e:
            logger.warning(
                f"LLM model '{model_name}' unavailable ({e}). Trying next candidate..."
            )

    raise RuntimeError("All candidate LLM models failed to initialize.")


class TriageLLMResponse(BaseModel):
    """Pydantic schema enforcing structured triage output from the LLM."""
    category: str = Field(
        description="Canonical category predicted from taxonomy.json (e.g. service, ordering, billing, etc.)"
    )
    priority: str = Field(
        description="Inferred priority level strictly from retrieved cases: low, medium, or high"
    )
    routed_queue: str = Field(
        description="Target queue or operational team (e.g. Service Scheduling Team, Order Management Team)"
    )
    confidence: float = Field(
        description="Confidence score between 0.0 and 1.0 reflecting classification certainty, semantic alignment, and case consistency"
    )
    resolution_notes: str = Field(
        description="1-2 line summary note explaining recommended next steps or how similar cases were handled"
    )


def format_taxonomy_context(taxonomy_data: dict) -> str:
    """Format canonical categories and descriptions into textual context."""
    categories = taxonomy_data.get("categories", [])
    lines = []
    for cat in categories:
        name = cat.get("name", "")
        desc = cat.get("description", "")
        keywords = ", ".join(cat.get("keywords", []))
        lines.append(f"- Category: {name}\n  Description: {desc}\n  Keywords: {keywords}")
    return "\n\n".join(lines)


def format_past_cases_context(docs) -> str:
    """Format retrieved Top-K cases and their metadata into readable context."""
    if not docs:
        return "No historical cases retrieved."

    lines = []
    for i, doc in enumerate(docs, 1):
        case_id = doc.metadata.get("case_id", f"CASE-{i}")
        category = doc.metadata.get("category", "unknown")
        priority = doc.metadata.get("priority", "medium")
        routed_queue = doc.metadata.get("routed_queue", "General Support")
        lines.append(
            f"Case #{i} [ID: {case_id}]:\n"
            f"  Inquiry: \"{doc.page_content}\"\n"
            f"  Historical Category: {category}\n"
            f"  Historical Priority: {priority}\n"
            f"  Routed Queue: {routed_queue}"
        )
    return "\n\n".join(lines)


# =========================================================================
# LangGraph State & Node Definitions
# =========================================================================

class TriageState(TypedDict):
    """LangGraph state schema representing the triage workflow."""
    query: str
    top_k: int
    confidence_threshold: float
    retrieved_cases: list
    final_output: dict


def retrieve_node(state: TriageState) -> dict:
    """
    Node 1 (retrieve_node):
    Extract query and top_k from state, run ChromaDB similarity search,
    and return retrieved past cases to update the graph state.
    """
    query = state.get("query", "")
    top_k = state.get("top_k", 3)

    api_key = load_environment()
    embeddings_model = get_embeddings_model(api_key)
    vector_store = get_vector_store(embeddings_model)
    results = vector_store.similarity_search(query, k=top_k)

    return {"retrieved_cases": results}


def reasoning_node(state: TriageState) -> dict:
    """
    Node 2 (reasoning_node):
    Extract query, retrieved_cases, and confidence_threshold from state.
    Assemble taxonomy context and historical few-shot context, invoke Gemini
    with structured output, calculate escalation flag, and return final_output.
    """
    query = state.get("query", "")
    retrieved_docs = state.get("retrieved_cases", [])
    confidence_threshold = state.get("confidence_threshold", 0.5)

    api_key = load_environment()
    taxonomy_data = load_taxonomy()
    taxonomy_context = format_taxonomy_context(taxonomy_data)
    past_cases_context = format_past_cases_context(retrieved_docs)
    category_names = [c.get("name") for c in taxonomy_data.get("categories", [])]

    top_k = len(retrieved_docs) if retrieved_docs else 3

    system_prompt = f"""You are an expert customer inquiry triage AI assistant for an automotive company.
Your job is to analyze incoming customer inquiries, classify them accurately into the canonical categories defined in the taxonomy, infer their priority strictly from retrieved historical case metadata, route them to the appropriate support queue, estimate a confidence score, and draft concise resolution notes.

=== CANONICAL TAXONOMY ===
You must classify the inquiry into one of these official categories: {category_names}
Definitions and reference keywords:
{taxonomy_context}

=== RETRIEVED TOP-{top_k} HISTORICAL PRECEDENT CASES ===
Use these past cases and their metadata (category, priority, routed_queue) to guide your classification, priority assignment, and queue routing:
{past_cases_context}

=== TRIAGE RULES ===
1. Category: Choose the best matching category strictly from {category_names}.
2. Priority: Infer priority ('low', 'medium', 'high') strictly from the metadata of the retrieved Top-{top_k} cases. For vehicle safety-critical symptoms (e.g. brakes failing, smoke, loss of steering), prioritize 'high'.
3. Routed Queue: Select the appropriate operational queue/team matching historical precedent.
4. Confidence Score: Output an explicit float from 0.0 to 1.0 representing your classification confidence, considering semantic alignment, historical precedent consistency, and query clarity.
5. Resolution Notes: Provide a 1-2 line actionable summary on how to resolve the inquiry or next steps based on historical handling.
"""

    user_message = f"Customer Inquiry to triage:\n\"{query}\""



    llm = get_llm(api_key)
    structured_llm = llm.with_structured_output(TriageLLMResponse)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]

    response: TriageLLMResponse = structured_llm.invoke(messages)

    confidence_val = round(float(response.confidence), 2)
    escalated = bool(confidence_val < confidence_threshold)
    past_cases_texts = [doc.page_content for doc in retrieved_docs]

    category = response.category.lower().strip()
    priority = response.priority.lower().strip()
    routed_queue = response.routed_queue.strip()
    resolution_notes = response.resolution_notes.strip()

    # Formatted matching both strict PDF keys and frontend compatibility
    generated_json = {
        "query": query,
        "category": category,
        "priority": priority,
        "routed queue": routed_queue,
        "routed_queue": routed_queue,
        "confidence": confidence_val,
        "resolution notes": resolution_notes,
        "resolution_notes": resolution_notes,
        "retrieved past cases": past_cases_texts,
        "retrieved_past_cases": past_cases_texts,
        "escalated": escalated,
    }

    return {"final_output": generated_json}


# =========================================================================
# StateGraph Orchestration & Compilation
# =========================================================================

def build_triage_graph():
    """
    Initialize StateGraph, define nodes, wire edges:
    START -> retrieve_node -> reasoning_node -> END
    """
    workflow = StateGraph(TriageState)
    workflow.add_node("retrieve_node", retrieve_node)
    workflow.add_node("reasoning_node", reasoning_node)

    workflow.add_edge(START, "retrieve_node")
    workflow.add_edge("retrieve_node", "reasoning_node")
    workflow.add_edge("reasoning_node", END)

    return workflow.compile()


# Compile the singleton LangGraph application
triage_app = build_triage_graph()


def triage_inquiry(query: str, top_k: int = 3, confidence_threshold: float = 0.5) -> dict:
    """
    Execute end-to-end inquiry triage through the compiled LangGraph StateGraph.

    Args:
        query: Raw customer inquiry text.
        top_k: Number of historical cases to retrieve.
        confidence_threshold: Minimum confidence threshold before escalation.

    Returns:
        Structured output dictionary with keys matching PDF requirements:
        - "query"
        - "category"
        - "priority"
        - "routed queue"
        - "confidence"
        - "resolution notes"
        - "retrieved past cases"
        - "escalated"
    """
    initial_state: TriageState = {
        "query": query,
        "top_k": top_k,
        "confidence_threshold": confidence_threshold,
        "retrieved_cases": [],
        "final_output": {},
    }

    result = triage_app.invoke(initial_state)
    return result.get("final_output", {})


if __name__ == "__main__":
    # Terminal Validation Test
    print("\n" + "#" * 80)
    print("RUNNING LANGGRAPH TERMINAL VALIDATION TEST")
    print("#" * 80)

    test_query = "The brake fluid warning appeared and the pedal feels soft."
    test_top_k = 3
    test_confidence_threshold = 0.5

    print(f"Test Query: '{test_query}'")
    print(f"Top-K: {test_top_k} | Confidence Threshold: {test_confidence_threshold}\n")

    triage_result = triage_inquiry(
        query=test_query,
        top_k=test_top_k,
        confidence_threshold=test_confidence_threshold
    )

    print("\n" + "#" * 80)
    print("COMPLETE STRUCTURED OUTPUT DICTIONARY (FROM LANGGRAPH):")
    print("#" * 80)
    print(json.dumps(triage_result, indent=4))
    print("#" * 80 + "\n")
