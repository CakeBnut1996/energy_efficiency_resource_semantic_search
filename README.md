## Energy Efficiency Resource Search 

This project is a Retrieval-Augmented Generation (RAG) application designed to help users quickly find and summarize technical energy efficiency resources. It uses a semantic search engine to navigate through a variety of documents including roadmaps, tip sheets, and toolkits.

### 🤖 What is RAG?
Retrieval-Augmented Generation (RAG) is a technique that combines the power of Large Language Models (LLMs) with your own private or specific data.

- Retrieval: When you ask a question, the system searches a vector database (ChromaDB) for the most relevant "chunks" of text from your documents.

- Augmentation: These snippets are then provided to the LLM as "context".

- Generation: The LLM uses this context to write a factually grounded answer, reducing "hallucinations" and ensuring the information is based on the actual resources provided.

### 🛠️ Tech Stack
- Frontend: Streamlit for an interactive, web-based UI.

- Vector Database: ChromaDB for high-speed semantic similarity search.

- LLMs: Integrated with Gemini (Google), GPT (OpenAI), Claude (Anthropic), and Llama (via Groq).

- Structured Output: Pydantic for structured data validation and schema management.

### 📂 Searchable Resources

The system indexes a wide range of industrial and energy-related documents:

- Energy Systems Source Books (e.g., Compressed Air, Steam, Pumps)

- Guidance Documents & Tip Sheets

- Manufacturing Studies

### ⚙️ Configuration & Adjustments

The app's behavior is fully controlled via `config.yaml`. You can adjust:

- Chunking: `chunk_size` and `chunk_overlap` for how documents are split.

- Retrieval: `num_docs` (how many documents to fetch) and `active_embedding` (which model to use for search).

- Models: Switch between different LLM providers (Gemini, OpenAI, Groq, Anthropic) by changing the `active_student`.

### 🧲 Output
![Semantic Search Demo.png](Semantic%20Search%20Demo.png)