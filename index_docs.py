#!/usr/bin/env python3
"""Index CockroachDB docs into SQLite with vector embeddings via sqlite-vec."""

import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

import openai
import sqlite_vec

# --- Configuration ---
DOCS_BASE = Path(__file__).resolve().parent.parent / "docs" / "src" / "current"
COCKROACH_REPO = Path(__file__).resolve().parent.parent / "cockroach"
VERSIONS_TO_INDEX = ["v24.3", "v25.1", "v25.2", "v25.3", "v25.4", "v26.1", "v26.2"]
EXTRA_DIRS = ["cockroachcloud", "releases", "advisories", "molt"]
DB_PATH = Path(__file__).resolve().parent / "docs.db"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
BATCH_SIZE = 100  # OpenAI supports up to 2048 inputs per request


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

    # Create the vector table (sqlite-vec virtual table)
    conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
            chunk_id INTEGER PRIMARY KEY,
            embedding FLOAT[{EMBEDDING_DIM}]
        );
    """)

    # Create FTS5 full-text index for keyword search
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            title, section_header, content,
            content='chunks',
            content_rowid='id'
        );
    """)

    return conn


def parse_frontmatter(content: str) -> tuple[dict, str]:
    """Extract YAML frontmatter and return (metadata, body)."""
    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            fm_text = content[3:end].strip()
            meta = {}
            for line in fm_text.split("\n"):
                if ":" in line:
                    key, _, val = line.partition(":")
                    meta[key.strip()] = val.strip()
            body = content[end + 3:].strip()
            return meta, body
    return {}, content


def chunk_markdown(file_path: Path, version: str) -> list[dict]:
    """Split a markdown file into chunks by ## headers."""
    content = file_path.read_text(encoding="utf-8", errors="replace")
    meta, body = parse_frontmatter(content)
    title = meta.get("title", file_path.stem)

    # Strip Jekyll/Liquid template tags that add noise
    body = re.sub(r'\{%.*?%\}', '', body)
    body = re.sub(r'\{\{.*?\}\}', '', body)

    # Split on ## headers (keep the header with its section)
    sections = re.split(r'^(#{2,3}\s+.+)$', body, flags=re.MULTILINE)

    chunks = []
    current_header = title  # Use doc title as header for intro content
    current_text = ""

    for part in sections:
        part = part.strip()
        if not part:
            continue
        if re.match(r'^#{2,3}\s+', part):
            # Save previous chunk if it has content
            if current_text.strip():
                chunks.append(_make_chunk(
                    file_path, version, title, current_header, current_text
                ))
            current_header = re.sub(r'^#{2,3}\s+', '', part).strip()
            current_text = ""
        else:
            current_text += part + "\n"

    # Don't forget the last section
    if current_text.strip():
        chunks.append(_make_chunk(
            file_path, version, title, current_header, current_text
        ))

    # If the file is small and produced no chunks (no ## headers), treat the whole body as one chunk
    if not chunks and body.strip():
        chunks.append(_make_chunk(file_path, version, title, title, body))

    return chunks


def _make_chunk(file_path: Path, version: str, title: str, header: str, text: str) -> dict:
    """Create a chunk dict with metadata."""
    # Build a text representation for embedding that includes context
    try:
        rel_path = str(file_path.relative_to(DOCS_BASE))
    except ValueError:
        # File is outside DOCS_BASE (e.g. cockroach repo internals)
        try:
            rel_path = "cockroach/" + str(file_path.relative_to(COCKROACH_REPO))
        except ValueError:
            rel_path = str(file_path)
    embed_text = f"{title} > {header}\n\n{text.strip()}"

    # Rough token estimate (1 token ~ 4 chars for English)
    token_est = len(embed_text) // 4

    return {
        "file_path": rel_path,
        "version": version,
        "title": title,
        "section_header": header,
        "content": text.strip(),
        "embed_text": embed_text,
        "content_hash": hashlib.md5(embed_text.encode()).hexdigest(),
        "token_estimate": token_est,
    }


def get_embeddings(client: openai.OpenAI, texts: list[str]) -> list[list[float]]:
    """Get embeddings from OpenAI in a batch."""
    # 8191 token limit ≈ ~28000 chars conservatively
    truncated = [t[:28000] for t in texts]

    try:
        resp = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=truncated,
        )
        return [item.embedding for item in resp.data]
    except openai.BadRequestError:
        # A chunk in this batch is still too long; fall back to one-at-a-time
        results = []
        for t in truncated:
            try:
                resp = client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=[t[:20000]],  # more aggressive truncation for outliers
                )
                results.append(resp.data[0].embedding)
            except openai.BadRequestError:
                # Last resort: very short truncation
                resp = client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=[t[:8000]],
                )
                results.append(resp.data[0].embedding)
        return results


def collect_files() -> list[tuple[Path, str]]:
    """Collect all markdown files to index, returning (path, version) tuples."""
    files = []
    for version in VERSIONS_TO_INDEX:
        # Main version docs
        version_dir = DOCS_BASE / version
        if version_dir.is_dir():
            for md in version_dir.rglob("*.md"):
                files.append((md, version))
        # _includes for this version (release notes, feature highlights, partials)
        includes_dir = DOCS_BASE / "_includes" / version
        if includes_dir.is_dir():
            for f in includes_dir.rglob("*"):
                if f.suffix in (".md", ".html") and f.is_file():
                    files.append((f, version))

    for extra in EXTRA_DIRS:
        extra_dir = DOCS_BASE / extra
        if extra_dir.is_dir():
            for md in extra_dir.rglob("*.md"):
                files.append((md, extra))
        # _includes for extras (e.g. _includes/releases/, _includes/cockroachcloud/)
        includes_dir = DOCS_BASE / "_includes" / extra
        if includes_dir.is_dir():
            for f in includes_dir.rglob("*"):
                if f.suffix in (".md", ".html") and f.is_file():
                    files.append((f, extra))

    # Also index _includes/common (shared across versions)
    common_dir = DOCS_BASE / "_includes" / "common"
    if common_dir.is_dir():
        for f in common_dir.rglob("*"):
            if f.suffix in (".md", ".html") and f.is_file():
                files.append((f, "common"))

    # Index cockroach repo internals: tech notes, RFCs, READMEs
    if COCKROACH_REPO.is_dir():
        for subdir, version_label in [
            ("docs/tech-notes", "tech-notes"),
            ("docs/RFCS", "rfcs"),
        ]:
            d = COCKROACH_REPO / subdir
            if d.is_dir():
                for f in d.rglob("*.md"):
                    files.append((f, version_label))

        # READMEs throughout the codebase (skip vendor/node_modules)
        for f in COCKROACH_REPO.rglob("README.md"):
            if "vendor" not in f.parts and "node_modules" not in f.parts:
                files.append((f, "source-readme"))

    return files


def main():
    client = get_client()
    conn = init_db(DB_PATH)

    # Check what's already indexed
    existing_hashes = set(
        row[0] for row in conn.execute("SELECT content_hash FROM chunks").fetchall()
    )

    print("Collecting files...")
    files = collect_files()
    print(f"Found {len(files)} markdown files")

    # Chunk all files
    all_chunks = []
    for file_path, version in files:
        try:
            chunks = chunk_markdown(file_path, version)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"  Warning: failed to chunk {file_path}: {e}")

    print(f"Generated {len(all_chunks)} chunks")

    # Filter out already-indexed chunks
    new_chunks = [c for c in all_chunks if c["content_hash"] not in existing_hashes]
    print(f"New chunks to embed: {len(new_chunks)} (skipping {len(all_chunks) - len(new_chunks)} already indexed)")

    if not new_chunks:
        print("Nothing new to index!")
        conn.close()
        return

    # Estimate cost
    total_tokens = sum(c["token_estimate"] for c in new_chunks)
    est_cost = total_tokens / 1_000_000 * 0.02  # $0.02 per 1M tokens
    print(f"Estimated tokens: {total_tokens:,} (~${est_cost:.3f})")

    # Embed and insert in batches
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
            # Wait and retry once
            time.sleep(5)
            try:
                embeddings = get_embeddings(client, texts)
            except Exception as e2:
                print(f"  Retry also failed: {e2}, skipping batch")
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

            # Insert the embedding vector
            conn.execute(
                "INSERT INTO chunks_vec (chunk_id, embedding) VALUES (?, ?)",
                (chunk_id, sqlite_vec.serialize_float32(emb))
            )

            # Insert into FTS index
            conn.execute(
                "INSERT INTO chunks_fts (rowid, title, section_header, content) VALUES (?, ?, ?, ?)",
                (chunk_id, chunk["title"], chunk["section_header"], chunk["content"])
            )

        conn.commit()
        print("done")

        # Small delay to be nice to the API
        if batch_idx < total_batches - 1:
            time.sleep(0.2)

    # Rebuild FTS index to sync with external content table
    print("Rebuilding FTS index...")
    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
    conn.commit()

    # Print stats
    total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    print(f"\nDone! Total chunks in database: {total}")
    print(f"Database: {DB_PATH}")
    conn.close()


if __name__ == "__main__":
    main()
