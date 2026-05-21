# RepoInsight AI 🧠

RepoInsight AI is a highly performant, production-grade Retrieval-Augmented Generation (RAG) platform designed to ingest, index, and query entire Git codebases using advanced AI models. It acts as an intelligent assistant that instantly parses vast repositories, maps code structures semantically into a local Vector Database, and answers complex developer queries with precise file and line-number citations.

## 🚀 Features

- **Multi-Codebase Context Management**: Seamlessly ingest multiple repositories (like `uvicorn`, `fastapi`, or your own projects), switch between them on the fly, and securely delete context via an O(1) disk teardown.
- **Advanced RAG Pipeline**: Combines local SentenceTransformers embeddings (`all-MiniLM-L6-v2`) with ChromaDB for blazing-fast local vector search, passed to Groq's high-speed inference endpoints for instant semantic responses.
- **Robust Git Parsing Engine**: Safely clones repositories, aggressively filters binary and junk files via immutable rulesets, and implements syntax-aware text chunking to preserve code integrity.
- **Premium Glassmorphism UI**: A completely custom, zero-dependency, dark-mode frontend built with pure HTML, CSS, and Vanilla JavaScript, featuring smooth micro-animations, context switching, and inline code snippet visualizations.
- **Thread-Safe Backend architecture**: Built on FastAPI with asynchronous routing, thread-pooled disk I/O, strict Pydantic data validation, and custom structured JSON logging.

## 🛠️ Tech Stack

- **Backend**: FastAPI, Python 3.10+, Uvicorn, Pydantic
- **AI & RAG**: Groq API, ChromaDB (Local Vector Store), SentenceTransformers
- **Frontend**: Pure HTML5, CSS3 (Glassmorphism design), Vanilla JavaScript
- **Version Control Handling**: GitPython

## 💻 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/repoinsight-ai.git
   cd repoinsight-ai
   ```

2. **Set up a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables:**
   Create a `.env` file in the root directory and add your Groq API key:
   ```env
   GROQ_API_KEY=gsk_your_groq_api_key_here
   ```

5. **Run the Application:**
   ```bash
   python main.py
   ```
   *The server will start on `http://localhost:8000/`. The beautiful frontend dashboard will be served directly at the root endpoint.*

## 🔍 How it Works

1. **Ingest:** Enter a Git URL into the dashboard. The backend performs a shallow clone, strips out noise (`node_modules`, binaries, `.git`), and divides the remaining source files into overlapping semantic chunks.
2. **Embed:** Code chunks are passed through an embedding model and mapped into a high-dimensional vector space stored locally via ChromaDB.
3. **Query:** When you ask a question (e.g. *"Where is the core router logic?"*), your query is embedded, and ChromaDB retrieves the Top-K most relevant code snippets.
4. **Generate:** These snippets are injected into a highly defensive system prompt alongside your query and sent to the LLM to generate an accurate, grounded response with exact source code citations.

## 📝 License
This project is open-source and available under the MIT License.
