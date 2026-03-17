#!/usr/bin/env python3
"""Scrape and index CockroachDB engineering + product blog posts."""

import hashlib
import os
import re
import sqlite3
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

import httpx
import openai
import sqlite_vec
from bs4 import BeautifulSoup

# --- Configuration ---
DB_PATH = Path(__file__).resolve().parent / "docs.db"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
BATCH_SIZE = 100
BASE_URL = "https://www.cockroachlabs.com"
CATEGORIES = {
    "ai": "/blogs/ai/",
    "engineering": "/blogs/engineering/",
    "product": "/blogs/product/",
}
REQUEST_DELAY = 0.5  # seconds between requests, be polite


def get_openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: set OPENAI_API_KEY environment variable")
        sys.exit(1)
    return openai.OpenAI(api_key=api_key)


def get_http_client():
    return httpx.Client(
        headers={"User-Agent": "CockroachDB-Docs-Indexer/1.0"},
        follow_redirects=True,
        timeout=30.0,
    )


def init_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    # Ensure tables exist (same schema as index_docs.py)
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


def discover_post_urls(http: httpx.Client, category: str, category_path: str) -> list[str]:
    """Crawl category listing pages to find all blog post URLs."""
    urls = []
    page = 1

    while True:
        list_url = f"{BASE_URL}{category_path}?page={page}"
        print(f"  Fetching {category} page {page}...", end=" ", flush=True)

        try:
            resp = http.get(list_url)
            resp.raise_for_status()
        except Exception as e:
            print(f"failed: {e}")
            break

        soup = BeautifulSoup(resp.text, "lxml")

        # Find blog post links - they're in /blog/ paths
        found = set()
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith("/blog/") and href != "/blog/" and href.count("/") >= 2:
                # Normalize: ensure trailing slash
                if not href.endswith("/"):
                    href += "/"
                found.add(href)

        if not found:
            print("no posts found, stopping")
            break

        new_count = len(found - set(urls))
        urls.extend(u for u in found if u not in urls)
        print(f"{len(found)} links ({new_count} new)")

        # Check if there's a next page
        pagination_text = soup.get_text()
        match = re.search(r"Page\s+\d+\s+of\s+(\d+)", pagination_text)
        if match:
            total_pages = int(match.group(1))
            if page >= total_pages:
                break
        else:
            # No pagination info found, stop after this page
            if new_count == 0:
                break

        page += 1
        time.sleep(REQUEST_DELAY)

    return urls


def fetch_and_parse_post(http: httpx.Client, url_path: str) -> dict | None:
    """Fetch a blog post and extract its content."""
    full_url = f"{BASE_URL}{url_path}"

    try:
        resp = http.get(full_url)
        resp.raise_for_status()
    except Exception as e:
        print(f"    Failed to fetch {url_path}: {e}")
        return None

    soup = BeautifulSoup(resp.text, "lxml")

    # Extract title
    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else url_path.split("/")[-2]

    # Extract date
    date = ""
    date_match = re.search(r"Published on\s+(\w+ \d+, \d{4})", soup.get_text())
    if date_match:
        date = date_match.group(1)

    # Extract article body - look for the main content area
    # The blog uses class "blog-template" or we can find the article content
    # by looking for the main content div with paragraphs and headers
    article = None

    # Try common content containers
    for selector in [
        soup.find("article"),
        soup.find("div", class_="blog-template"),
        soup.find("div", class_="blog-content"),
        soup.find("main"),
    ]:
        if selector:
            article = selector
            break

    if not article:
        # Fallback: find the largest div with the most <p> tags
        divs = soup.find_all("div")
        best = None
        best_count = 0
        for div in divs:
            p_count = len(div.find_all("p", recursive=False))
            if p_count > best_count:
                best_count = p_count
                best = div
        article = best

    if not article:
        return None

    # Remove script, style, nav, footer elements
    for tag in article.find_all(["script", "style", "nav", "footer", "noscript"]):
        tag.decompose()

    # Convert to text preserving structure
    sections = []
    current_header = title
    current_text = []

    for element in article.descendants:
        if element.name in ("h2", "h3"):
            # Save previous section
            text = "\n".join(current_text).strip()
            if text:
                sections.append((current_header, text))
            current_header = element.get_text(strip=True)
            current_text = []
        elif element.name in ("p", "li", "pre", "code", "blockquote", "td"):
            text = element.get_text(strip=True)
            if text and len(text) > 10:
                current_text.append(text)

    # Don't forget the last section
    text = "\n".join(current_text).strip()
    if text:
        sections.append((current_header, text))

    if not sections:
        return None

    return {
        "url_path": url_path,
        "full_url": full_url,
        "title": title,
        "date": date,
        "sections": sections,
    }


def chunk_blog_post(post: dict) -> list[dict]:
    """Convert a parsed blog post into chunks for embedding."""
    chunks = []

    for header, content in post["sections"]:
        embed_text = f"Blog: {post['title']} > {header}\n\n{content}"
        token_est = len(embed_text) // 4

        chunks.append({
            "file_path": f"blog:{post['url_path']}",
            "version": "blog",
            "title": post["title"],
            "section_header": header,
            "content": content,
            "embed_text": embed_text,
            "content_hash": hashlib.md5(embed_text.encode()).hexdigest(),
            "token_estimate": token_est,
        })

    return chunks


def get_embeddings(client: openai.OpenAI, texts: list[str]) -> list[list[float]]:
    """Get embeddings from OpenAI in a batch."""
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
    oai = get_openai_client()
    http = get_http_client()
    conn = init_db(DB_PATH)

    # Check already-indexed blog posts
    existing_hashes = set(
        row[0] for row in conn.execute("SELECT content_hash FROM chunks").fetchall()
    )
    existing_blogs = set(
        row[0] for row in conn.execute(
            "SELECT DISTINCT file_path FROM chunks WHERE version = 'blog'"
        ).fetchall()
    )
    print(f"Already indexed: {len(existing_blogs)} blog posts")

    # Discover all post URLs
    all_urls = []
    for category, path in CATEGORIES.items():
        print(f"\nDiscovering {category} posts...")
        urls = discover_post_urls(http, category, path)
        all_urls.extend(urls)
        print(f"  Found {len(urls)} {category} posts")

    # Deduplicate (some posts may appear in multiple categories)
    all_urls = list(dict.fromkeys(all_urls))
    print(f"\nTotal unique post URLs: {len(all_urls)}")

    # Skip already-fetched posts
    new_urls = [u for u in all_urls if f"blog:{u}" not in existing_blogs]
    print(f"New posts to fetch: {len(new_urls)} (skipping {len(all_urls) - len(new_urls)} already indexed)")

    if not new_urls:
        print("Nothing new to index!")
        conn.close()
        return

    # Fetch and parse posts
    all_chunks = []
    for i, url_path in enumerate(new_urls):
        print(f"  [{i+1}/{len(new_urls)}] Fetching {url_path}...", end=" ", flush=True)

        post = fetch_and_parse_post(http, url_path)
        if post:
            chunks = chunk_blog_post(post)
            # Filter out already-indexed chunks
            new_chunks = [c for c in chunks if c["content_hash"] not in existing_hashes]
            all_chunks.extend(new_chunks)
            print(f"{len(chunks)} chunks ({len(new_chunks)} new)")
        else:
            print("failed to parse")

        time.sleep(REQUEST_DELAY)

    print(f"\nTotal new chunks to embed: {len(all_chunks)}")

    if not all_chunks:
        print("No new chunks to embed!")
        conn.close()
        return

    # Estimate cost
    total_tokens = sum(c["token_estimate"] for c in all_chunks)
    est_cost = total_tokens / 1_000_000 * 0.02
    print(f"Estimated tokens: {total_tokens:,} (~${est_cost:.3f})")

    # Embed and insert in batches
    total_batches = (len(all_chunks) + BATCH_SIZE - 1) // BATCH_SIZE
    for batch_idx in range(total_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(all_chunks))
        batch = all_chunks[start:end]

        print(f"  Batch {batch_idx + 1}/{total_batches} ({len(batch)} chunks)...", end=" ", flush=True)

        texts = [c["embed_text"] for c in batch]
        try:
            embeddings = get_embeddings(oai, texts)
        except Exception as e:
            print(f"FAILED: {e}")
            time.sleep(5)
            try:
                embeddings = get_embeddings(oai, texts)
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
            conn.execute(
                "INSERT INTO chunks_vec (chunk_id, embedding) VALUES (?, ?)",
                (chunk_id, sqlite_vec.serialize_float32(emb))
            )

        conn.commit()
        print("done")

        if batch_idx < total_batches - 1:
            time.sleep(0.2)

    # Rebuild FTS
    print("Rebuilding FTS index...")
    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
    conn.commit()

    total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    blog_count = conn.execute("SELECT COUNT(*) FROM chunks WHERE version = 'blog'").fetchone()[0]
    print(f"\nDone! Blog chunks: {blog_count}, Total chunks in database: {total}")
    print(f"Database: {DB_PATH}")

    http.close()
    conn.close()


if __name__ == "__main__":
    main()
