#!/usr/bin/env python3
"""MCP server exposing CockroachDB docs hybrid search (vector + FTS with RRF)."""

import os
import sqlite3
from pathlib import Path

import openai
import sqlite_vec
from mcp.server.fastmcp import FastMCP

DB_PATH = Path(__file__).resolve().parent / "docs.db"
EMBEDDING_MODEL = "text-embedding-3-small"
RRF_K = 60
CANDIDATE_POOL = 40

mcp = FastMCP("cockroachdb-docs-jlupolt")

_client = None
_conn = None


def get_client():
    global _client
    if _client is None:
        _client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def get_conn():
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(str(DB_PATH))
        _conn.enable_load_extension(True)
        sqlite_vec.load(_conn)
        _conn.enable_load_extension(False)
    return _conn


def _vector_search(conn, query_embedding, pool_size, version=None):
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


def _fts_search(conn, query, pool_size, version=None):
    words = query.split()
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
        return []


@mcp.tool()
def search_cockroachdb_docs(query: str, limit: int = 10, version: str | None = None) -> str:
    """Search CockroachDB documentation using hybrid vector + full-text search.

    Combines semantic similarity (OpenAI embeddings) with keyword matching (FTS5)
    using reciprocal rank fusion for best results on any query type.

    Use this to find docs about CockroachDB features, architecture, SQL syntax,
    cluster operations, performance tuning, troubleshooting, release notes, and more.
    Indexed versions: v24.3 through v26.2, plus CockroachCloud, releases, advisories, and MOLT.

    Args:
        query: Natural language search query (e.g. "how does lease transfer work",
               "performance improvements in v25.2", "changefeed syntax")
        limit: Number of results to return (default 10)
        version: Optional version filter (e.g. "v26.1", "cloud")
    """
    client = get_client()
    conn = get_conn()

    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
    query_embedding = resp.data[0].embedding

    vec_results = _vector_search(conn, query_embedding, CANDIDATE_POOL, version)
    fts_results = _fts_search(conn, query, CANDIDATE_POOL, version)

    # Reciprocal rank fusion
    rrf_scores: dict[int, float] = {}
    for rank, (chunk_id, _) in enumerate(vec_results):
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (RRF_K + rank + 1)
    for rank, (chunk_id, _) in enumerate(fts_results):
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (RRF_K + rank + 1)

    all_ranked = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

    if not all_ranked:
        return "No results found."

    # Fetch top candidates
    top_candidates = all_ranked[:limit * 4]
    placeholders = ",".join("?" * len(top_candidates))
    rows = conn.execute(f"""
        SELECT id, file_path, version, title, section_header, content
        FROM chunks WHERE id IN ({placeholders})
    """, top_candidates).fetchall()

    chunk_map = {row[0]: row for row in rows}
    vec_ids = {c for c, _ in vec_results}
    fts_ids = {c for c, _ in fts_results}

    # Diversify: limit results per source type so blogs/docs/release notes all surface
    max_per_source = max(limit // 2, 3)
    source_counts: dict[str, int] = {}
    ranked_ids = []
    for cid in all_ranked:
        if cid not in chunk_map:
            continue
        fp = chunk_map[cid][1]
        src = "blog" if fp.startswith("blog:") else ("releases" if "_includes/releases" in fp or fp.startswith("releases/") else "docs")
        if source_counts.get(src, 0) < max_per_source:
            ranked_ids.append(cid)
            source_counts[src] = source_counts.get(src, 0) + 1
        if len(ranked_ids) >= limit:
            break

    results = []
    for i, cid in enumerate(ranked_ids, 1):
        row = chunk_map[cid]
        source = "vec+fts" if (cid in vec_ids and cid in fts_ids) else ("vec" if cid in vec_ids else "fts")
        content = row[5]
        if len(content) > 1500:
            content = content[:1500] + "\n[...truncated]"
        results.append(
            f"[{i}] {row[3]} > {row[4]}\n"
            f"    File: {row[1]} | Version: {row[2]} | Score: {rrf_scores[cid]:.4f} | Via: {source}\n\n"
            f"{content}"
        )

    return "\n\n---\n\n".join(results)


@mcp.tool()
def get_doc_chunk_by_file(file_path: str) -> str:
    """Retrieve all indexed chunks from a specific documentation file.

    Use this to read the full indexed content of a doc page after finding it via search.

    Args:
        file_path: The relative file path as returned by search (e.g. "v26.1/admission-control.md")
    """
    conn = get_conn()
    rows = conn.execute(
        "SELECT section_header, content FROM chunks WHERE file_path = ? ORDER BY id",
        (file_path,)
    ).fetchall()

    if not rows:
        return f"No chunks found for file: {file_path}"

    parts = []
    for header, content in rows:
        parts.append(f"## {header}\n\n{content}")

    return "\n\n".join(parts)


if __name__ == "__main__":
    mcp.run()
