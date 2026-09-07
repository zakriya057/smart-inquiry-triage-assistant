import os
import sys
import time
import logging
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
import chromadb
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Resolve project paths
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
DATA_PATH = BASE_DIR / "data" / "past_cases.csv"
CHROMA_DIR = BASE_DIR / "chroma_db"

def load_environment():
    """Load environment variables from .env file."""
    load_dotenv(dotenv_path=ENV_PATH)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key.strip() == "":
        logger.error(
            "GOOGLE_API_KEY is not set or is empty. "
            "Please add your key to .env before running ingestion."
        )
        sys.exit(1)
    logger.info("Google API key detected.")
    return api_key

def get_embeddings_model(api_key: str):
    """Initialize embedding model with fallback support."""
    model_name = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")
    logger.info(f"Initializing GoogleGenerativeAIEmbeddings ({model_name})...")
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

def embed_with_retry(embeddings_model, docs, max_retries: int = 5):
    """Embed documents with automatic backoff for rate limits (429)."""
    for attempt in range(1, max_retries + 1):
        try:
            return embeddings_model.embed_documents(docs)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                wait_time = 60
                logger.warning(
                    f"Rate limit encountered on attempt {attempt}/{max_retries}. "
                    f"Waiting {wait_time}s before retrying..."
                )
                time.sleep(wait_time)
            else:
                raise e
    raise RuntimeError("Exceeded maximum retries for embedding generation.")

def ingest_data(batch_size: int = 50):
    """Load past cases, generate embeddings, and store them into ChromaDB."""
    api_key = load_environment()

    if not DATA_PATH.exists():
        logger.error(f"Dataset not found at {DATA_PATH}")
        sys.exit(1)

    logger.info(f"Loading past cases from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    logger.info(f"Loaded {len(df)} records. Columns: {list(df.columns)}")

    required_cols = {"case_id", "inquiry_text", "category", "priority", "routed_queue"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        logger.error(f"Missing required columns in CSV: {missing_cols}")
        sys.exit(1)

    # Clean text and handle nulls
    df["inquiry_text"] = df["inquiry_text"].fillna("").astype(str).str.strip()
    df["category"] = df["category"].fillna("unknown").astype(str)
    df["priority"] = df["priority"].fillna("medium").astype(str)
    df["routed_queue"] = df["routed_queue"].fillna("unknown").astype(str)

    embeddings_model = get_embeddings_model(api_key)

    # Initialize persistent ChromaDB
    logger.info(f"Initializing ChromaDB PersistentClient at {CHROMA_DIR}...")
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma_client.get_or_create_collection(
        name="triage_cases",
        metadata={"hnsw:space": "cosine"}
    )

    # Check for already ingested IDs to allow resume
    existing_items = collection.get(include=[])
    existing_ids = set(existing_items["ids"]) if existing_items else set()
    logger.info(f"Found {len(existing_ids)} cases already in collection.")

    pending_df = df[~df["case_id"].isin(existing_ids)].reset_index(drop=True)
    if pending_df.empty:
        logger.info(f"All {len(df)} records are already ingested. Nothing to do.")
        return collection.count()

    total_pending = len(pending_df)
    logger.info(f"Ingesting remaining {total_pending} cases (batch size: {batch_size})...")

    for i in range(0, total_pending, batch_size):
        batch = pending_df.iloc[i : i + batch_size]
        
        batch_ids = batch["case_id"].tolist()
        batch_docs = batch["inquiry_text"].tolist()
        batch_metadatas = [
            {
                "case_id": row["case_id"],
                "category": row["category"],
                "priority": row["priority"],
                "routed_queue": row["routed_queue"],
            }
            for _, row in batch.iterrows()
        ]

        logger.info(
            f"Generating embeddings for batch {i // batch_size + 1} "
            f"({len(batch_docs)} records)..."
        )
        batch_embeddings = embed_with_retry(embeddings_model, batch_docs)

        # Upsert into ChromaDB
        collection.upsert(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_metadatas,
            embeddings=batch_embeddings
        )
        logger.info(f"Ingested {min(i + batch_size, total_pending)}/{total_pending} pending records.")

    total_count = collection.count()
    logger.info(f"Ingestion complete! Total cases currently in 'triage_cases' collection: {total_count}")
    return total_count

if __name__ == "__main__":
    ingest_data()
