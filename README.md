# Smart Inquiry Triage Assistant 📨

An AI-powered customer inquiry triage system that automatically **classifies** customer requests, infers operational **priority**, **routes** them to the correct department queue, estimates an explicit **confidence score**, and drafts concise **resolution notes** using semantic similarity over historical precedent cases (RAG).

---

## 📌 Project Overview

Customer support teams often handle high volumes of varied inquiries ranging from routine service requests to safety-critical defects. The **Smart Inquiry Triage Assistant** automates this workflow:

1. **Semantic Vector Retrieval (RAG)**: Uses Gemini embeddings to search a persistent ChromaDB vector store (`triage_cases`) containing 300+ historical precedents.
2. **Canonical Taxonomy Alignment**: Cross-references customer inquiries against official category definitions in `data/taxonomy.json`.
3. **Structured Reasoning**: Leverages Gemini with Pydantic structured output to classify the inquiry, infer priority from historical cases, assign the routed queue, and draft resolution notes.
4. **Confidence Scoring & Escalation**: Quantifies classification certainty ($0.0 - 1.0$) and automatically flags inquiries for human review if confidence falls below a configurable threshold.
5. **Interactive UI**: An intuitive Streamlit interface allowing support engineers to test inquiries and dynamically tune Top-K retrieval and confidence thresholds.

---

## 📂 Repository Structure

```text
├── app/
│   └── app.py              # Streamlit web application & chat interface
├── src/
│   ├── ingest.py           # Standalone ChromaDB data ingestion & embedding pipeline
│   └── main.py             # Core triage backend, vector retrieval, and LLM reasoning
├── data/
│   ├── past_cases.csv      # Historical inquiries dataset (300 cases with metadata)
│   └── taxonomy.json       # Canonical categories, definitions, and reference keywords
├── chroma_db/              # Persistent ChromaDB vector database
├── Dockerfile              # Container definition
├── docker-compose.yml      # Orchestration with automated database bootstrap
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## 🚀 Setup & Installation

### Option A: Docker (Highly Recommended)

Docker provides an isolated, zero-configuration environment.

> **Note on Automated ChromaDB Bootstrap:**
> The container includes a self-bootstrapping startup check: on the very first container build and launch, if the local `./chroma_db` directory does not already exist, the container automatically provisions it by running `python src/ingest.py` to embed all 300 historical cases before launching Streamlit. No manual database setup is required when using Docker.


#### 1. Prerequisites
- [Docker](https://docs.docker.com/get-docker/) & [Docker Compose](https://docs.docker.com/compose/) installed and running.

#### 2. Configure Environment Variables
Create a `.env` file in the root directory:
```bash
cp .env.example .env   # or create .env directly
```
Add your Google Gemini API key:
```dotenv
GOOGLE_API_KEY="your-gemini-api-key-here"
```

#### 3. Build & Run
Launch the application with Docker Compose:
```bash
docker compose up --build
```
*(Use `docker compose up -d` to run in the background).*

The application will be accessible at:
👉 **[http://localhost:8501](http://localhost:8501)**

#### 4. Stopping the Container
```bash
docker compose down
```

---

### Option B: Manual Setup (Local Python)

If you prefer to run directly on your host machine without Docker:

#### 1. Prerequisites
- Python 3.10 or higher.
- `pip` package manager.

#### 2. Clone & Create Virtual Environment
```bash
git clone https://github.com/zakriya057/smart-inquiry-triage-assistant.git
cd smart-inquiry-triage-assistant

python3 -m venv .venv
source .venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Configure Environment Variables
Create a `.env` file in the project root:
```dotenv
GOOGLE_API_KEY="your-gemini-api-key-here"
```

#### 5. Run Standalone Ingestion (Build Vector Store)
Ingest historical cases and generate embeddings into the local persistent ChromaDB:
```bash
python src/ingest.py
```
*(This creates and populates the `./chroma_db` directory).*

#### 6. Launch the Streamlit Frontend
```bash
streamlit run app/app.py
```

Open your browser and navigate to:
👉 **[http://localhost:8501](http://localhost:8501)**

---

## 🧪 Terminal Validation (Backend Test)

You can validate the backend triage pipeline directly in the terminal without opening the browser:
```bash
python src/main.py
```
*(Or inside Docker: `docker compose run --rm app python src/main.py`)*

This runs a sample test inquiry through retrieval, context assembly, and Gemini structured output, printing the complete response dictionary to the console.
