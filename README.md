# TLDrRacket

![TLDrRacket — a reading companion for CS4400 / CS5400](docs/assets/readme-header.png)

**English** · [简体中文](README.zh-CN.md) · [日本語](README.ja.md)

A small reading companion for Northeastern's [CS4400 / CS5400: Principles of Programming Languages](https://pl.barzilay.org/). Ask a natural-language question, get an answer grounded in the indexed readings, and open up to **five relevant reading locations**. Fewer matches means fewer sources; no matches produces an explicit abstention and no generation call.

The application uses Python, Flask, local Chroma, Gemini, and the existing HTML/CSS/JavaScript frontend. There is no frontend build step, separate database service, agent framework, or queue. It is also intended as an approachable example of building a RAG application.

## Quick start

Use Python **3.12**. Commands below run from the repository root.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set `GOOGLE_API_KEY`. It uses standard `KEY=value` entries; a legacy file containing only a bare API key must be converted. Existing process environment variables take precedence. Never put the key in frontend code.

```sh
python index.py build
python webserver.py
```

Open [localhost:5000](http://127.0.0.1:5000). The first command builds the persisted index using the embedding API; the server does not build it automatically. The page and `/healthz` can load without an index, but questions require a configured key and a ready index. Use the [Linux VPS guide](docs/deployment.md) for Gunicorn, systemd, Nginx, and HTTPS.

## How the RAG works

```text
PDF/text → page extraction → overlapping chunks → embeddings → local Chroma
Question → one vector search → distance filter → page deduplication → up to 5 excerpts
                                                                    ↓
                                             Gemini answer + the same reading locations
```

1. `index.py` reads PDFs and UTF-8 `.txt` files under `DATA_PATH`. PDF pages are split into 800-character chunks with 80 characters of overlap. Empty pages are skipped; scanned PDFs need OCR before ingestion.
2. Chroma stores embeddings and metadata: relative filename, zero-based PDF page, chunk number, and a stable ID including a content hash. The collection explicitly uses **cosine distance**.
3. `rag.py` retrieves up to 20 candidate chunks once. It filters distances, takes the best chunk per file/page, and caps the selection at five. These are five **locations**, not necessarily five different documents.
4. Only the selected chunks enter the generation prompt. The same chunks supply source links and previews. The model is instructed to cite `[1]`, `[2]`, etc., and admit insufficient evidence.
5. With no accepted chunks, the server returns `status: "no_match"`, an explanation, and `sources: []`, without calling the generation model. The query embedding still requires the embedding API.

Source responses include the full retrieved excerpt as `content`, preserving its line breaks and indentation, alongside the compact `preview`. For `.txt` notes such as `data/5400Notes.txt`, `format` is `markdown`; PDF excerpts use `text`. The **Read excerpt** section shows the original passage with headings, lists, tables and code blocks when present. Markdown excerpts open by default and can be collapsed; PDF text excerpts can be expanded. This is the same retrieved passage used for the answer, not the whole file or an extra model-generated summary. Excerpts may begin or end within a Markdown construct because the index stores bounded chunks.

The UI immediately shows a waiting animation and elapsed time, blocks duplicate submissions, and restores the input after success, failure, or its 90-second timeout. PDF links use **PDF page numbers starting at 1**, which may differ from printed page labels. Text sources open the raw text file; the expandable excerpt provides formatted reading inside the chat. Generated Markdown is sanitized, and user questions are rendered as text.

## Configuration

See [`.env.example`](.env.example). Paths can be absolute; relative paths resolve from the process working directory.

| Variable | Default | Purpose |
| --- | --- | --- |
| `GOOGLE_API_KEY` | required | Server-side Gemini Developer API key |
| `CHAT_MODEL` | `gemini-3.5-flash-lite` | Answer generation model |
| `EMBEDDING_MODEL` | `models/gemini-embedding-001` | Embedding model, shared by indexing and queries |
| `EMBEDDING_DIMENSIONS` | `768` | Embedding vector dimensions |
| `DATA_PATH` | repository `data/` | PDF and text readings |
| `CHROMA_PATH` | repository `chroma/` | Persistent local index |
| `MAX_DISTANCE` | `0.35` | **Provisional** maximum accepted cosine distance |
| `CANDIDATE_COUNT` | `20` | Candidate chunks before filtering/deduplication; 5–100 |
| `PROVIDER_TIMEOUT` | `30` | Provider transport timeout in seconds; maximum 45 |
| `MAX_MESSAGE_LENGTH` | `2000` | Maximum non-empty question length |

Defaults are configured for the Gemini Developer API, not Vertex AI. The old Gemini 1.5 Flash and `text-embedding-004` identifiers in the historical notebooks are retired; see Google's [release notes](https://ai.google.dev/gemini-api/docs/changelog) and [deprecation schedule](https://ai.google.dev/gemini-api/docs/deprecations). The text embedding default remains supported and is the integration tested here. Changing embedding models, dimensions, or chunk settings requires an explicit rebuild; changing only the chat model or cutoff does not.

### Tune relevance instead of filling slots

**Lower cosine distance means a closer match.** `0.35` is a starting value, not a measured confidence level or a universally correct cutoff. It has not been calibrated against real Gemini embeddings for this course corpus in this checkout.

Inspect candidates without paying for answer generation:

```sh
python index.py probe "How do closures capture an environment?"
python index.py probe "What is the weather tomorrow?"
python index.py calibrate docs/retrieval-eval.json
```

`probe` prints candidate distances, previews, and whether each passes the cutoff; page deduplication and the five-location cap happen afterwards. `calibrate` embeds each labeled question once, then sweeps thresholds from 0.05 to 0.80. It reports correct/missed course questions, wrong-source answers, and unrelated questions that would be accepted. No answer generation occurs.

Review and extend the starter labels in `docs/retrieval-eval.json` for your actual readings, including valid alternative sources such as textbooks. Prefer a cutoff that avoids unrelated and wrong-source accepts while still finding course questions. Check it on separate held-out questions, including Chinese questions if relevant, then set `MAX_DISTANCE` and restart the service. This helper checks whether at least one expected source was retrieved; it does not grade every displayed source or the generated answer.

### Update materials or migrate an index

With the web service stopped:

```sh
python index.py build
```

Unchanged chunks are reused; changed/new chunks are embedded; stale chunks from removed or edited readings are deleted. Treat the entire `DATA_PATH` directory as the intended corpus: removing files there removes their indexed chunks on the next build.

Old notebook indices and changed embedding settings require migration. Back up the index first, then run:

```sh
python index.py build --rebuild
```

This replaces only the application's `tldrracket` collection. When upgrading from an index built before Markdown whitespace preservation was added, run `build --rebuild` once so indented code remains intact. An interrupted/failed write leaves a `.building` marker that prevents serving a partial index; see the [recovery procedure](docs/deployment.md#recover-an-interrupted-build). Restart the server after every successful update. Keep data, index and model configuration together when restoring a backup.

## HTTP API

`POST /api/ask` with `Content-Type: application/json`:

```json
{"message": "What is lexical scope?"}
```

A successful response has `response`, `status` (`ok` or `no_match`), and `sources`. Each source has `id`, `file`, `page` (1-based for PDF, `null` for text), `url`, `preview`, `content` (the complete retrieved chunk), `format` (`markdown` for `.txt`, `text` for PDF), and `distance`. Distances are diagnostics, not confidence scores.

| Endpoint / status | Meaning |
| --- | --- |
| `GET /healthz` | Process is alive; makes no model calls |
| `GET /readyz` | Key configured, compatible non-empty index opens; makes no model calls |
| `GET /documents/<filename>` | Serve a PDF/text reading inside `DATA_PATH` |
| `400`, `413` | Invalid question or oversized request body |
| `429` | Two questions already running in this process, or proxy rate limit |
| `502`, `504` | Provider failure, empty answer, or detected provider timeout |
| `503` | Missing, incompatible, empty, or updating index/configuration |

Errors use `{"error": "..."}`. API responses are not cached. Nginx may return its own error page; the UI handles non-JSON errors too. Readiness does not verify the remote API key, quota or model availability. A browser timeout stops waiting, but does not guarantee cancellation of a provider request already running on the server.

## Files and learning path

- `index.py`: document loading, chunking, incremental indexing, probes and cutoff sweep.
- `rag.py`: shared configuration, index compatibility, retrieval, sources, and generation.
- `webserver.py`: HTTP validation, lazy initialization, concurrency bound, and safe document serving.
- `static/index.html`, `static/chat.js`, `static/style.css`: existing frontend with waiting feedback and source links.
- `deploy/`: production process and reverse proxy configuration examples.
- `TLDrRacket.ipynb`, `evaluation.ipynb`: historical experiments, preserved for reference. They are not the supported runtime and need dependency/model updates before reuse. The old evaluation used source IDs as contexts; meaningful faithfulness evaluation should use actual excerpt text.

## Verification

Runtime tests use Python's standard library; no test framework or API key is required:

```sh
python -m unittest discover -s tests -v
```

Tests include real temporary Chroma collections with deterministic **local test vectors**, alongside isolated provider substitutes. They cover cosine filtering, 0–5 results, source/context consistency, deduplication, index updates/migrations, request validation, failures, and path traversal. They do not prove real-model answer quality.

Optional browser checks require Playwright only in your development environment. Start `python webserver.py` in another terminal, then:

```sh
python -m pip install playwright
python -m playwright install chromium
python tests/browser_smoke.py
```

These checks use intercepted responses to verify waiting, duplicate submissions, 0/1/5 sources, timeout/error recovery, Markdown answers/source excerpts, sanitization, and mobile behavior.

## Scope and limitations

This is a single-node RAG example, with up to two concurrent questions and a persistent embedded database. Retrieval is approximate; a distance cutoff can still accept unrelated passages or reject useful ones. The prompt encourages grounded answers and numbered citations, but does not formally verify claims or citations. There is no browsing, reranking model, chat memory, OCR service, or automated corpus synchronization. Reading coverage reflects the files you index, not necessarily the current course website. The document route exposes the readings in `DATA_PATH` to site visitors, so use a corpus you intend to make available through this tool.

JavaScript dependencies are vendored and versioned in `static/vendor/`, with their licenses and origins in [THIRD_PARTY.md](THIRD_PARTY.md). The application code is [MIT licensed](LICENSE); included course/readings assets retain their respective authorship and terms.
