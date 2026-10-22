# Lightweight RAG Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline, task by task. The user approved the design; implementation proceeds in this chat.

**Goal:** Add honest waiting feedback, zero-to-five relevant reading locations, reproducible documentation and Linux VPS deployment without a new application language.

**Architecture:** Flask serves the existing frontend, API and safe document routes. One shared Python module owns configuration, cosine retrieval, source selection and generation; one CLI owns indexing and threshold inspection. The approved design is `docs/implementation-proposal.md`.

**Tech Stack:** Python, Flask, LangChain Chroma and Gemini integrations, existing HTML/CSS/JavaScript, Gunicorn, systemd, Nginx.

## 1. Retrieval and HTTP behavior

- [x] Add `tests/test_rag.py` and `tests/test_webserver.py`, using unittest and injected provider substitutes. Cover empty/unrelated retrieval, distance boundary, page deduplication, maximum five locations, shared answer context, input validation, provider errors, safe PDF serving and readiness.
- [x] Run `python -m unittest discover -s tests -v`; confirm missing behavior fails before implementation.
- [x] Add `rag.py` with `Settings`, configuration validation, index compatibility checks, cosine filtering and `RagService.ask`. Replace `webserver.py` with `create_app` and a lazy service factory so importing the app never invokes paid APIs.
- [x] Run the same tests and verify all pass using an isolated virtual environment.

## 2. Reproducible indexing

- [x] Add tests for stable chunk IDs, metadata compatibility and rebuilding constraints.
- [x] Implement `index.py`: read PDF and text pages, split 800/80, use deterministic IDs including content hashes, synchronize additions/deletions in batches, and record model/distance/chunk settings in a manifest. Require explicit rebuild for incompatible settings; never rebuild at server startup.
- [x] Add `probe` to inspect candidate distances without generation and `calibrate` to compare labeled queries against candidate distance thresholds. Keep the initial threshold explicitly provisional until checked against the real corpus.
- [x] Verify with a temporary persistent Chroma collection and deterministic local embeddings, plus CLI help and unit tests.

## 3. Frontend feedback and sources

- [x] Extract chat code to `static/chat.js`. Add native loading status, elapsed time, duplicate-submit guard, fetch timeout and error recovery; keep theme behavior.
- [x] Render user text with textContent and sanitize generated Markdown before insertion. Render sources as escaped file/page links and previews. Label no-match results clearly.
- [x] Verify in a browser with intercepted responses: loading, duplicate submission, zero/one/five sources, error recovery, timeout, source links, mobile layout and untrusted content. No JS test framework or frontend build dependency is added.

## 4. Documentation and VPS artifacts

- [x] Add `requirements.txt`, `.env.example`, README and `docs/deployment.md`, explaining the RAG flow, source semantics, API, indexing, threshold tuning and model migrations. Keep actual secrets and Chroma files out of Git.
- [x] Add `deploy/gunicorn.conf.py`, `deploy/tldrracket.service` and `deploy/nginx.conf`. Bind Gunicorn to localhost, run as a dedicated account, preserve `/var/lib/tldrracket`, set startup/read/request timeouts, and include health checks, HTTPS, upgrades and rollback commands.
- [x] Run dependency checks, compilation, full tests, real local Chroma integration and browser verification. Review the final diff for design coverage.
- [x] Honor the user’s subsequent scope clarification: deliver the VPS deployment files only, without connecting to a server. Record that real Gemini calibration and remote smoke checks have not been run.

## Verification record

- 27 unittest cases pass, including real temporary Chroma indexing/search and an actual constructed Gemini transport request with a mocked response.
- Browser checks pass for loading/elapsed state, duplicate protection, 0/1/5 sources, error/timeout recovery, sanitized content, mobile viewport and reduced motion.
- Gunicorn starts from the production configuration; page/health return 200, and missing-index readiness/questions return 503.
- Dependency compatibility, Python compilation, CLI help, Gunicorn configuration and Git whitespace checks pass.
- No real Gemini call or semantic threshold calibration has been performed: this checkout has no configured key. The cutoff remains explicitly provisional.
- VPS configuration is provided for target-machine validation; no remote deployment was requested after scope clarification.
