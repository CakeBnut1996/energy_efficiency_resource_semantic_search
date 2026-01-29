import os
import re
import hashlib
import tiktoken
import fitz  # PyMuPDF
from pathlib import Path
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
from typing import List, Any
from io_utils.load_db import load_embedding_model, get_db_collection, get_or_create_collection


# --- Text Processing Functions (Same as before) ---

def extract_text_from_html(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.extract()
    return soup.get_text(separator="\n", strip=True)


def extract_text_from_pdf(path: str) -> str:
    """
    Opens a PDF file and extracts text from all pages.
    """
    if not os.path.exists(path):
        print(f"⚠️ Warning: File not found at {path}")
        return ""

    text_content = []

    try:
        # Open the PDF document
        with fitz.open(path) as doc:
            for page in doc:
                # Extract text from the current page
                page_text = page.get_text("text")
                text_content.append(page_text)

        # Join all pages with a newline separator
        return "\n".join(text_content).strip()

    except Exception as e:
        print(f"❌ Error reading PDF {path}: {e}")
        return ""

def clean_text(text: str) -> str:
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\f', '', text)
    return text.strip()


def filter_noise(text: str) -> str:
    lines = text.split("\n")
    clean_lines = []
    for ln in lines:
        s = ln.strip()
        if not s: continue
        if re.match(r'^\d+[\.\)]', s): continue
        if len(s) < 30 and s.isupper(): continue
        if "REFERENCES" in s.upper() or "TABLE" in s.upper(): continue
        clean_lines.append(ln)
    return "\n".join(clean_lines)


def _deduplicate_chunks(chunks: List[str]) -> List[str]:
    seen = set()
    unique = []
    for c in chunks:
        h = hashlib.md5(c.encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(c)
    return unique


def chunk_text(text: str, tokenizer_name: str = "cl100k_base", max_tokens: int = 256, overlap: int = 40) -> List[str]:
    enc = tiktoken.get_encoding(tokenizer_name)
    sentences = sent_tokenize(text)
    chunks = []
    current = []
    current_tokens = 0

    for sent in sentences:
        sent_len = len(enc.encode(sent))
        if sent_len > max_tokens: continue
        if current_tokens + sent_len > max_tokens:
            full_chunk = " ".join(current)
            chunks.append(full_chunk)
            overlap_txt = full_chunk[-overlap:] if len(full_chunk) > overlap else full_chunk
            current = [overlap_txt]
            current_tokens = len(enc.encode(overlap_txt))
        current.append(sent)
        current_tokens += sent_len

    if current: chunks.append(" ".join(current))
    return _deduplicate_chunks(chunks)


# --- Database Interaction ---

def embed_and_upsert(chunks: List[str], collection: Any, embedding_model: Any, model_name: str, source_filename: str):
    if not chunks: return

    # Prefix handling for E5 models
    doc_prefix = "passage: " if "e5" in model_name.lower() else ""
    texts_to_embed = [f"{doc_prefix}{c}" for c in chunks]

    embeddings = embedding_model.encode(texts_to_embed, convert_to_numpy=True)

    ids = [f"{source_filename}_{i}" for i in range(len(chunks))]
    metadatas = [{"dataset": source_filename}] * len(chunks)

    collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
    print(f"   ✅ Saved {len(chunks)} chunks.")


# --- 🚀 MASTER INGESTION FUNCTION ---

def run_ingestion(
        data_dir: str,
        db_path: str,
        collection_name: str,
        embedding_model_name: str,
        tokenizer_model: str = "cl100k_base",
        chunk_size: int = 256,
        chunk_overlap: int = 40
):
    # 1. Initialize Resources
    collection = get_or_create_collection(db_path, collection_name)
    model = load_embedding_model(embedding_model_name)

    # 2. Find Files (Recursive Search)
    if not os.path.exists(data_dir):
        print(f"❌ Error: Data directory '{data_dir}' not found.")
        return

    # Use Path.glob to find all files in all subfolders
    data_path = Path(data_dir)
    all_files = [f for f in data_path.rglob("*") if f.is_file()]
    print(f"\n🚀 Found {len(all_files)} files in subdirectories. Starting ingestion...\n")

    # 3. Process Loop
    for file_path in all_files:
        # Create a readable name: "Subfolder/Filename"
        # .relative_to(data_dir) keeps only the path after the main data folder
        relative_path = file_path.relative_to(data_path)
        readable_source_name = str(relative_path).replace("\\", "/")  # Use forward slashes for readability

        print(f"📄 Processing: {readable_source_name}")

        # Pipeline: Extract -> Clean -> Filter -> Chunk
        ext = file_path.suffix.lower()

        if ext == ".html":
            raw_text = extract_text_from_html(str(file_path))
        elif ext == ".pdf":
            raw_text = extract_text_from_pdf(str(file_path))
        else:
            continue  # Skip unsupported files

        clean_txt = clean_text(raw_text)
        filtered_txt = filter_noise(clean_txt)

        chunks = chunk_text(
            filtered_txt,
            tokenizer_name=tokenizer_model,
            max_tokens=chunk_size,
            overlap=chunk_overlap
        )

        # Database Upsert
        embed_and_upsert(
            chunks=chunks,
            collection=collection,
            embedding_model=model,
            model_name=embedding_model_name,
            source_filename=readable_source_name  # Now includes the subfolder path
        )

    count = collection.count()
    print(f"\n✅ Ingestion Complete! Collection '{collection_name}' now contains {count} chunks.")