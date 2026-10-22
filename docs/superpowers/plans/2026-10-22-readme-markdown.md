# Multilingual README and Markdown source excerpts

The user approved the existing design and requested two concrete refinements: English-default documentation in English/Chinese/Japanese with the supplied header image, and formatted excerpts from the Markdown notes stored as .txt.

- [x] Add the supplied image unchanged at docs/assets/readme-header.png; embed it in README.md, README.zh-CN.md and README.ja.md with cross-language navigation. Translate the full existing README, preserving executable commands and API identifiers.
- [x] Add regression tests for exact source content, Markdown/text format tags, and preservation of leading indentation in text chunks. Confirm they fail first.
- [x] Keep the plain preview for compatibility; return full selected content and format. Render expandable excerpts using the existing Showdown + DOMPurify pipeline. Keep PDF extraction as plain text and preserve existing retrieval/citation limits.
- [x] Preserve text indentation during indexing, record the index format change, and document the required rebuild. No new runtime library or application language.
- [x] Verify headings/lists/indented and fenced code/tables in a real browser, including malicious source HTML and missing-library fallbacks. Run the existing backend/browser checks.
- [x] Refresh the explicitly labeled local demo to show actual Markdown from data/5400Notes.txt, and provide local README previews without publishing or pushing the repository.

## Verification

- 31 backend tests pass, including verbatim source content, format tags, preserved code indentation, and legacy-index rejection.
- Browser checks pass for Markdown headings/lists/fenced code/tables, source sanitization, plain PDF rendering, missing-sanitizer fallback, loading/error states and mobile behavior. Markdown excerpts open by default without a duplicated raw preview.
- Live demo renders an actual 702-character chunk from data/5400Notes.txt, including its indented Racket code. Demo answers remain explicitly labeled fixed samples with no model or vector search.
- English, Chinese and Japanese README links and shell/JSON examples match; their banner is byte-identical to the supplied image. All three documents and images render in the local browser preview.
- Python compilation, JavaScript syntax and Git whitespace checks pass. Read-only code review found no remaining actionable issues.
- No new runtime dependency or application language was added, and no GitHub push or VPS connection was performed.
