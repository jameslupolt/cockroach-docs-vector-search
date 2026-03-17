# docs-vector-search

Local hybrid search (vector + full-text) over CockroachDB documentation, blog posts, internal design docs, and the O'Reilly book. Built with SQLite, [sqlite-vec](https://github.com/asg017/sqlite-vec), FTS5, and OpenAI embeddings. Exposes an MCP server for use with Claude Code, Cursor, VS Code, etc.

## What's indexed

| Source | Chunks | Description |
|---|---|---|
| Docs (v24.3-v26.2) | ~15,200 | Official versioned docs + `_includes` (release notes, feature highlights) |
| Blog posts | ~2,300 | 327 posts from engineering, product, and AI categories |
| RFCs | ~2,200 | Internal design documents from `cockroach/docs/RFCS/` |
| O'Reilly book | ~500 | *CockroachDB: The Definitive Guide*, 2nd edition |
| Source READMEs | ~440 | Package-level overviews from `cockroach/` repo |
| Tech notes | ~380 | Internal technical deep-dives from `cockroach/docs/tech-notes/` |
| **Total** | **~21,700** | **~210 MB database, ~$0.12 total embedding cost** |

## How it works

**Indexing:** Markdown/HTML files are chunked by `##`/`###` headers. Blog posts are scraped and split by `<h2>`/`<h3>` tags. The PDF book is split by TOC entries. Each chunk is embedded with OpenAI `text-embedding-3-small` (1536 dimensions) and stored in a single SQLite database alongside an FTS5 full-text index.

**Search:** Every query runs two searches in parallel:
1. **Vector similarity** via sqlite-vec -- finds semantically related content even when keywords don't match
2. **FTS5 keyword search** via SQLite -- finds exact term matches (version numbers, error codes, function names)

Results are merged using **Reciprocal Rank Fusion (RRF)** -- a chunk that ranks high in both searches gets boosted to the top. A **source diversity** step ensures results include a mix of blogs, docs, release notes, and internal docs rather than 10 copies of the same page.

**Deduplication:** Content is hashed before insertion. Identical content across doc versions is stored once. Re-running indexers only embeds new/changed content.

## Prerequisites

- Python 3.12+ (tested with 3.14)
- [uv](https://docs.astral.sh/uv/) package manager
- An OpenAI API key (for embeddings -- ~$0.001 per query, ~$0.12 to build the full index)

For indexing source-repo internals (RFCs, tech notes, READMEs):
- Clone of [cockroachdb/cockroach](https://github.com/cockroachdb/cockroach) at `../cockroach`

For indexing the docs site:
- Clone of [cockroachdb/docs](https://github.com/cockroachdb/docs) at `../docs`

For indexing the book:
- PDF of *CockroachDB: The Definitive Guide* at `../oreilly-cockroachdb-the-definitive-guide.pdf`

For indexing internal Confluence pages:
- `CONFLUENCE_URL`, `CONFLUENCE_USERNAME`, and `CONFLUENCE_API_TOKEN` environment variables

## Setup

```bash
cd docs-vector-search
export OPENAI_API_KEY="sk-..."
```
### Build the index from scratch

```bash
# Index official docs (v24.3-v26.2), _includes, tech notes, RFCs, READMEs
uv run python index_docs.py

# Scrape and index engineering + product + AI blog posts
uv run python index_blog.py

# Index the O'Reilly book (if you have the PDF)
uv run python index_book.py
```

Each indexer is incremental -- re-run anytime to pick up new content. Already-indexed chunks are skipped via content hashing. The Confluence indexer re-indexes all configured pages on each run to pick up edits.

### Update the index

Just re-run the indexers. They'll detect and embed only new/changed content:

```bash
uv run python index_docs.py        # picks up new doc versions, release notes
uv run python index_blog.py        # picks up new blog posts
uv run python index_confluence.py  # re-indexes configured Confluence pages
```

## Usage

### CLI search

```bash
# Basic search
uv run python search_docs.py "how does admission control work"

# More results
uv run python search_docs.py "changefeed duplicate messages" -n 15

# Filter by version
uv run python search_docs.py "lease transfer" -v v26.1

# Show full chunk content
uv run python search_docs.py "transaction retry error 40001" --full
```

### MCP server (for Claude Code, Cursor, etc.)

Add to your `.mcp.json`:

```json
{
  "mcpServers": {
    "cockroachdb-docs": {
      "type": "stdio",
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/path/to/docs-vector-search",
        "python",
        "mcp_server.py"
      ],
      "env": {
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

The MCP server exposes two tools:

- **`search_cockroachdb_docs(query, limit?, version?)`** -- hybrid search across all indexed sources
- **`get_doc_chunk_by_file(file_path)`** -- retrieve all chunks from a specific file (for deep-diving after search)

## Project structure

```
docs-vector-search/
  index_docs.py        # Index official docs, _includes, tech notes, RFCs, READMEs
  index_blog.py        # Scrape and index cockroachlabs.com/blog
  index_book.py        # Index the O'Reilly book PDF
  index_confluence.py  # Index internal Confluence pages by ID
  search_docs.py       # CLI search tool (hybrid vector + FTS with RRF)
  mcp_server.py        # MCP server wrapping search for AI assistants
  docs.db              # SQLite database (not checked in -- build with indexers)
  pyproject.toml       # Dependencies managed by uv
```

## Architecture decisions

**Why sqlite-vec + FTS5 instead of a vector database?**
Single file, no server, no infrastructure. The entire index is a 208 MB SQLite file. Queries take ~1.5 seconds (dominated by the OpenAI embedding API call, not the local search).

**Why hybrid search?**
Vector search alone struggles with version-specific queries ("what changed in v25.2") because embeddings don't capture version numbers well. FTS alone misses conceptual matches ("why is my cluster slow" won't match "admission control"). RRF fusion gives you the best of both.

**Why source diversity?**
Without it, release notes (which all contain similar boilerplate) flood the results. Capping results per source type (docs, blog, releases) ensures you see a mix of perspectives for any query.

**Why OpenAI embeddings instead of local?**
`text-embedding-3-small` is cheap ($0.02/M tokens), fast, and high quality. The full index costs ~$0.12 to build. Each query costs ~$0.001. A local model (e.g., via Ollama) would eliminate this cost but is slower and lower quality for technical documentation.

**Why retrieval-only instead of full RAG?**
RAG (Retrieval-Augmented Generation) has two steps: retrieve relevant chunks, then pass them to an LLM to synthesize an answer. This project only does the first step -- it returns raw source chunks with metadata. When used as an MCP tool, the AI assistant you're already talking to *is* the generation step, so adding another LLM layer in the middle would just add latency, cost, and a place where information gets lost or distorted.

This turns out to be better than full RAG for investigative work:

- **Transparency.** You see exactly which sources were found and can judge their relevance yourself, rather than trusting a black-box synthesis.
- **Lower hallucination risk.** In full RAG, an intermediate LLM can misinterpret chunks before you ever see them. Here, the LLM in your conversation has direct access to the sources and you can verify its reasoning.
- **Flexibility.** The LLM can decide to search again with a different query, pull a full file with `get_doc_chunk_by_file`, or combine results from multiple searches. A RAG pipeline gives you one shot.
- **Cheaper.** One embedding call per query (~$0.001) vs embedding + LLM generation.

Full RAG services like Kapa are better for "give me a quick answer" use cases where you don't need to see the sources. For "why did performance change after an upgrade" or "how does this internal system actually work," retrieval-only with an AI assistant is the stronger approach.

## Adapting for other projects

The approach generalizes to any documentation corpus:

1. **Swap the source paths** in `index_docs.py` to point at your docs
2. **Adjust chunking** in `chunk_markdown()` if your docs use different header levels
3. **Modify blog categories** in `index_blog.py` for your site's structure
4. **Add Confluence page IDs** to `index_confluence.py` for internal wikis
5. **Update the MCP tool descriptions** in `mcp_server.py` so AI assistants know what's indexed

The core search infrastructure (sqlite-vec, FTS5, RRF fusion, source diversity) is source-agnostic.
