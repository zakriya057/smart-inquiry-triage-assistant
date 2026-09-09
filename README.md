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

---

## 🧠 Design Decisions & Architecture

To build a robust and adaptive triage system, I designed an architecture that decouples mathematical vector retrieval from semantic business reasoning. 

### 1. Embeddings & Vector Database
* **Embedding Model:** I utilized the **Gemini embedding model** to convert all historical customer inquiries from the CSV into vector representations.
* **Vector Store:** I chose **ChromaDB** as the vector database. The primary reason for this choice is that it is incredibly lightweight, easy to run locally or inside a Docker container, and perfectly suited for managing this scale of document retrieval without heavy infrastructure overhead.

### 2. Retrieval & Classification (Hybrid LLM Approach)
* **Top-K Retrieval (RAG):** When a new inquiry comes in, the system uses a Retrieval-Augmented Generation (RAG) pipeline to fetch the Top-K most semantically similar historical elements from ChromaDB.
* **LLM Reasoning:** Instead of relying on a simple K-Nearest Neighbors (KNN) algorithm for classification, I pass the retrieved Top-K elements alongside the official `taxonomy.json` directly into a prompt for the LLM. 
* **The Output:** The LLM acts as the core reasoning engine. It processes the context to generate the final classification, determine operational priority, route the ticket to the correct queue, draft resolution notes, and calculate a confidence score.

### 3. Sample LLM Prompt Structure
To achieve this, I guided the LLM with a strict system prompt that looks conceptually like this:

> **System Role:** You are an expert customer inquiry triage AI assistant. Your role is to classify inquiries, determine priority, route to the correct queue, calculate a confidence score, and draft resolution notes.
> 
> **Context 1 (Taxonomy):** 
> [Insert categories, definitions, and keywords from taxonomy.json]
> 
> **Context 2 (Top-K Elements):** 
> [Insert retrieved historical cases: queries, past priority, past routing]
> 
> **Instructions:** 
> 1. Classify the user query strictly using Context 1.
> 2. Determine priority and routing strictly based on precedent in Context 2.
> 3. Provide a confidence score (0.0 - 1.0) based on taxonomy alignment and Top-K consistency.
> 4. Return the output in the strict required JSON format.

### 4. Confidence Scoring & Escalation Strategy
Rather than relying on raw vector-distance metrics, I implemented a **Hybrid LLM Self-Report**. The confidence score (0.0 - 1.0) is evaluated dynamically by the LLM based on:
* **Taxonomy Alignment:** The precision of the incoming query's match to the canonical definitions.
* **Precedent Consistency:** The level of unanimous agreement among the retrieved Top-K cases regarding priority and routing.
* **Ambiguity Penalty:** Vague queries safely lower the score, automatically triggering the configurable `escalated` UI flag for human review.

### 5. Key Trade-offs
* **Advantage — Explainability & Flexibility:** Passing context to the LLM enables human-readable resolution notes and allows operations teams to update `taxonomy.json` without retraining models or rebuilding the vector database.
* **Trade-off — Latency & Cost:** Relying on Gemini for the final triage decision introduces API latency and ongoing token costs compared to executing a simple K-Nearest Neighbors (KNN) algorithm directly on the vector embeddings.
