#!/usr/bin/env python3
"""Index the O'Reilly CockroachDB Definitive Guide PDF."""

import hashlib
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

import openai
import pymupdf
import sqlite_vec

# --- Configuration ---
DB_PATH = Path(__file__).resolve().parent / "docs.db"
BOOK_PATH = Path(__file__).resolve().parent.parent / "oreilly-cockroachdb-the-definitive-guide.pdf"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
BATCH_SIZE = 100
VERSION_LABEL = "book"


def get_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: set OPENAI_API_KEY environment variable")
        sys.exit(1)
    return openai.OpenAI(api_key=api_key)


def init_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT NOT NULL,
            version TEXT,
            title TEXT,
            section_header TEXT,
            content TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            token_estimate INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_path);
        CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(content_hash);
        CREATE INDEX IF NOT EXISTS idx_chunks_version ON chunks(version);
    """)

    conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
            chunk_id INTEGER PRIMARY KEY,
            embedding FLOAT[{EMBEDDING_DIM}]
        );
    """)

    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            title, section_header, content,
            content='chunks',
            content_rowid='id'
        );
    """)

    return conn


def extract_chapters(pdf_path: Path) -> list[dict]:
    """Extract text from PDF, chunked by TOC entries."""
    doc = pymupdf.open(str(pdf_path))
    toc = doc.get_toc()  # [(level, title, page_number), ...]

    if not toc:
        print("No TOC found, falling back to page-based chunking")
        return _chunk_by_pages(doc)

    # Build sections from TOC: each section spans from its page to the next TOC entry's page
    sections = []
    for i, (level, title, start_page) in enumerate(toc):
        # Determine end page (next TOC entry's start, or end of doc)
        if i + 1 < len(toc):
            end_page = toc[i + 1][2]
        else:
            end_page = len(doc)

        # Skip very short sections (cover, copyright, etc.)
        if end_page - start_page < 1 and level <= 1:
            continue

        # Extract text from pages (pymupdf uses 0-indexed pages, TOC uses 1-indexed)
        text = ""
        for page_num in range(start_page - 1, min(end_page, len(doc))):
            page_text = doc[page_num].get_text()
            if page_text:
                text += page_text + "\n"

        text = text.strip()
        if not text or len(text) < 50:
            continue

        # Determine the parent chapter title for context
        chapter = title
        for j in range(i - 1, -1, -1):
            if toc[j][0] < level:
                chapter = toc[j][1]
                break

        sections.append({
            "level": level,
            "title": chapter if level > 1 else title,
            "section_header": title,
            "content": text,
            "start_page": start_page,
        })

    doc.close()
    return sections


def _chunk_by_pages(doc, pages_per_chunk=3):
    """Fallback: chunk by groups of pages."""
    sections = []
    for i in range(0, len(doc), pages_per_chunk):
        text = ""
        for j in range(i, min(i + pages_per_chunk, len(doc))):
            text += doc[j].get_text() + "\n"
        text = text.strip()
        if text and len(text) > 50:
            sections.append({
                "level": 1,
                "title": "CockroachDB: The Definitive Guide",
                "section_header": f"Pages {i+1}-{min(i+pages_per_chunk, len(doc))}",
                "content": text,
                "start_page": i + 1,
            })
    return sections


def make_chunks(sections: list[dict]) -> list[dict]:
    """Convert sections to embedding-ready chunks, splitting large ones."""
    chunks = []
    max_chars = 6000  # ~1500 tokens, keeps us well under embedding limits

    for section in sections:
        content = section["content"]
        title = section["title"]
        header = section["section_header"]

        # Split large sections into sub-chunks by paragraph
        if len(content) > max_chars:
            paragraphs = content.split("\n\n")
            current = ""
            part = 1
            for para in paragraphs:
                if len(current) + len(para) > max_chars and current:
                    chunks.append(_build_chunk(title, f"{header} (part {part})", current))
                    current = para
                    part += 1
                else:
                    current += "\n\n" + para if current else para
            if current.strip():
                chunks.append(_build_chunk(title, f"{header} (part {part})" if part > 1 else header, current))
        else:
            chunks.append(_build_chunk(title, header, content))

    return chunks


def _build_chunk(title: str, header: str, content: str) -> dict:
    embed_text = f"Book: CockroachDB Definitive Guide > {title} > {header}\n\n{content.strip()}"
    return {
        "file_path": "book:cockroachdb-definitive-guide",
        "version": VERSION_LABEL,
        "title": title,
        "section_header": header,
        "content": content.strip(),
        "embed_text": embed_text,
        "content_hash": hashlib.md5(embed_text.encode()).hexdigest(),
        "token_estimate": len(embed_text) // 4,
    }


def get_embeddings(client: openai.OpenAI, texts: list[str]) -> list[list[float]]:
    truncated = [t[:28000] for t in texts]
    try:
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=truncated)
        return [item.embedding for item in resp.data]
    except openai.BadRequestError:
        results = []
        for t in truncated:
            try:
                resp = client.embeddings.create(model=EMBEDDING_MODEL, input=[t[:20000]])
                results.append(resp.data[0].embedding)
            except openai.BadRequestError:
                resp = client.embeddings.create(model=EMBEDDING_MODEL, input=[t[:8000]])
                results.append(resp.data[0].embedding)
        return results


def main():
    if not BOOK_PATH.exists():
        print(f"Error: book not found at {BOOK_PATH}")
        sys.exit(1)

    client = get_client()
    conn = init_db(DB_PATH)

    existing_hashes = set(
        row[0] for row in conn.execute("SELECT content_hash FROM chunks").fetchall()
    )

    # Check if book is already indexed
    book_count = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE version = ?", (VERSION_LABEL,)
    ).fetchone()[0]
    print(f"Already indexed book chunks: {book_count}")

    print("Extracting text from PDF...")
    sections = extract_chapters(BOOK_PATH)
    print(f"Extracted {len(sections)} sections from TOC")

    print("Building chunks...")
    all_chunks = make_chunks(sections)
    print(f"Generated {len(all_chunks)} chunks")

    new_chunks = [c for c in all_chunks if c["content_hash"] not in existing_hashes]
    print(f"New chunks: {len(new_chunks)} (skipping {len(all_chunks) - len(new_chunks)} already indexed)")

    if not new_chunks:
        print("Nothing new to index!")
        conn.close()
        return

    total_tokens = sum(c["token_estimate"] for c in new_chunks)
    est_cost = total_tokens / 1_000_000 * 0.02
    print(f"Estimated tokens: {total_tokens:,} (~${est_cost:.3f})")

    total_batches = (len(new_chunks) + BATCH_SIZE - 1) // BATCH_SIZE
    for batch_idx in range(total_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(new_chunks))
        batch = new_chunks[start:end]

        print(f"  Batch {batch_idx + 1}/{total_batches} ({len(batch)} chunks)...", end=" ", flush=True)

        texts = [c["embed_text"] for c in batch]
        try:
            embeddings = get_embeddings(client, texts)
        except Exception as e:
            print(f"FAILED: {e}")
            time.sleep(5)
            try:
                embeddings = get_embeddings(client, texts)
            except Exception as e2:
                print(f"  Retry failed: {e2}, skipping")
                continue

        for chunk, emb in zip(batch, embeddings):
            cursor = conn.execute(
                """INSERT INTO chunks (file_path, version, title, section_header, content, content_hash, token_estimate)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (chunk["file_path"], chunk["version"], chunk["title"],
                 chunk["section_header"], chunk["content"], chunk["content_hash"],
                 chunk["token_estimate"])
            )
            chunk_id = cursor.lastrowid
            conn.execute(
                "INSERT INTO chunks_vec (chunk_id, embedding) VALUES (?, ?)",
                (chunk_id, sqlite_vec.serialize_float32(emb))
            )

        conn.commit()
        print("done")

        if batch_idx < total_batches - 1:
            time.sleep(0.2)

    print("Rebuilding FTS index...")
    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
    conn.commit()

    total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    book_total = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE version = ?", (VERSION_LABEL,)
    ).fetchone()[0]
    print(f"\nDone! Book chunks: {book_total}, Total chunks in database: {total}")
    conn.close()


if __name__ == "__main__":
    main()
