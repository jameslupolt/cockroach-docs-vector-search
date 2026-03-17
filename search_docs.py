#!/usr/bin/env python3
"""Search CockroachDB docs using hybrid vector + full-text search with RRF."""

import os
import sqlite3
import sys
from pathlib import Path

import openai
import sqlite_vec

DB_PATH = Path(__file__).resolve().parent / "docs.db"
EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_LIMIT = 10
RRF_K = 60  # standard RRF constant — controls how much rank matters vs. being present at all
CANDIDATE_POOL = 40  # fetch this many from each source before merging


def get_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: set OPENAI_API_KEY environment variable")
        sys.exit(1)
    return openai.OpenAI(api_key=api_key)


def open_db() -> sqlite3.Connection:
    if not DB_PATH.exists():
        print(f"Error: database not found at {DB_PATH}")
        print("Run index_docs.py first.")
        sys.exit(1)

    conn = sqlite3.connect(str(DB_PATH))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def embed_query(client: openai.OpenAI, query: str) -> list[float]:
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
    return resp.data[0].embedding


def vector_search(conn: sqlite3.Connection, query_embedding, pool_size: int, version: str | None = None) -> list[tuple]:
    """Return (chunk_id, distance) tuples ranked by vector similarity."""
    serialized = sqlite_vec.serialize_float32(query_embedding)
    if version:
        rows = conn.execute("""
            SELECT v.chunk_id, v.distance
            FROM chunks_vec v
            JOIN chunks c ON c.id = v.chunk_id
            WHERE v.embedding MATCH ? AND k = ?
              AND c.version = ?
            ORDER BY v.distance
        """, (serialized, pool_size * 2, version)).fetchall()[:pool_size]
    else:
        rows = conn.execute("""
            SELECT v.chunk_id, v.distance
            FROM chunks_vec v
            WHERE v.embedding MATCH ? AND k = ?
            ORDER BY v.distance
        """, (serialized, pool_size)).fetchall()
    return rows


def fts_search(conn: sqlite3.Connection, query: str, pool_size: int, version: str | None = None) -> list[tuple]:
    """Return (chunk_id, fts_rank) tuples ranked by FTS5 BM25."""
    # Build FTS5 query: quote each word to avoid syntax issues, OR them together
    words = query.split()
    # Use implicit AND (FTS5 default) for multi-word queries, with individual words
    fts_query = " OR ".join(f'"{w}"' for w in words if len(w) > 1)
    if not fts_query:
        return []

    try:
        if version:
            rows = conn.execute("""
                SELECT f.rowid, f.rank
                FROM chunks_fts f
                JOIN chunks c ON c.id = f.rowid
                WHERE chunks_fts MATCH ? AND c.version = ?
                ORDER BY f.rank
                LIMIT ?
            """, (fts_query, version, pool_size)).fetchall()
        else:
            rows = conn.execute("""
                SELECT rowid, rank
                FROM chunks_fts
                WHERE chunks_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (fts_query, pool_size)).fetchall()
        return rows
    except sqlite3.OperationalError:
        # FTS query syntax error — fall back to empty
        return []


def hybrid_search(query: str, limit: int = DEFAULT_LIMIT, version: str | None = None) -> list[dict]:
    """Hybrid search combining vector similarity and FTS5 with reciprocal rank fusion."""
    client = get_client()
    conn = open_db()

    query_embedding = embed_query(client, query)

    # Get candidates from both sources
    vec_results = vector_search(conn, query_embedding, CANDIDATE_POOL, version)
    fts_results = fts_search(conn, query, CANDIDATE_POOL, version)

    # Build RRF scores: score = sum(1 / (k + rank)) across sources
    rrf_scores: dict[int, float] = {}

    for rank, (chunk_id, _distance) in enumerate(vec_results):
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (RRF_K + rank + 1)

    for rank, (chunk_id, _fts_rank) in enumerate(fts_results):
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (RRF_K + rank + 1)

    # Sort by RRF score descending
    all_ranked = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

    if not all_ranked:
        conn.close()
        return []

    # Fetch full chunk data for top candidates
    top_candidates = all_ranked[:limit * 4]
    placeholders = ",".join("?" * len(top_candidates))
    rows = conn.execute(f"""
        SELECT id, file_path, version, title, section_header, content
        FROM chunks WHERE id IN ({placeholders})
    """, top_candidates).fetchall()

    chunk_map = {row[0]: row for row in rows}

    # Diversify: limit results per source type (blog, release notes, docs)
    # This prevents patch release notes from flooding all result slots
    max_per_source = max(limit // 2, 3)
    source_counts: dict[str, int] = {}  # "blog", "releases", "docs"
    ranked_ids = []
    for cid in all_ranked:
        if cid not in chunk_map:
            continue
        file_path = chunk_map[cid][1]
        if file_path.startswith("blog:"):
            src = "blog"
        elif "_includes/releases" in file_path or file_path.startswith("releases/"):
            src = "releases"
        else:
            src = "docs"
        if source_counts.get(src, 0) < max_per_source:
            ranked_ids.append(cid)
            source_counts[src] = source_counts.get(src, 0) + 1
        if len(ranked_ids) >= limit:
            break
    results = []
    for cid in ranked_ids:
        if cid not in chunk_map:
            continue
        row = chunk_map[cid]
        # Determine which sources contributed
        in_vec = any(c == cid for c, _ in vec_results)
        in_fts = any(c == cid for c, _ in fts_results)
        source = "vec+fts" if (in_vec and in_fts) else ("vec" if in_vec else "fts")

        results.append({
            "id": row[0],
            "file_path": row[1],
            "version": row[2],
            "title": row[3],
            "section_header": row[4],
            "content": row[5],
            "rrf_score": rrf_scores[cid],
            "source": source,
        })

    conn.close()
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Search CockroachDB docs (hybrid vector + FTS)")
    parser.add_argument("query", nargs="+", help="Search query")
    parser.add_argument("-n", "--limit", type=int, default=DEFAULT_LIMIT, help="Number of results")
    parser.add_argument("-v", "--version", default=None, help="Filter by version")
    parser.add_argument("--full", action="store_true", help="Show full chunk content")
    args = parser.parse_args()

    query = " ".join(args.query)
    print(f"Searching for: {query}\n")

    results = hybrid_search(query, limit=args.limit, version=args.version)

    if not results:
        print("No results found.")
        return

    for i, r in enumerate(results, 1):
        print(f"{'─' * 80}")
        print(f"  [{i}] {r['title']} > {r['section_header']}")
        print(f"      File: {r['file_path']}  |  Score: {r['rrf_score']:.4f}  |  Via: {r['source']}")
        print()
        content = r["content"]
        if not args.full and len(content) > 500:
            content = content[:500] + "..."
        for line in content.split("\n"):
            print(f"      {line}")
        print()

    print(f"{'─' * 80}")
    print(f"{len(results)} results returned")


if __name__ == "__main__":
    main()
