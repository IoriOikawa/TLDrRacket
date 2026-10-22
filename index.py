"""Build the persistent reading index, inspect matches, or calibrate its cutoff."""

import argparse
import hashlib
import json
from dataclasses import replace

from rag import (CHUNK_OVERLAP, CHUNK_SIZE, COLLECTION, IndexNotReady, RagService,
                 Settings, check_manifest, create_service, get_embeddings,
                 open_store, write_manifest)


def read_chunks(settings):
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from pypdf import PdfReader

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    # Indentation is syntax in the Markdown notes (including indented code).
    markdown_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, strip_whitespace=False)
    chunks = []
    if not settings.data_path.is_dir():
        raise ValueError('DATA_PATH must be an existing directory containing readings.')
    for path in sorted(settings.data_path.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in {'.pdf', '.txt'}:
            continue
        if not path.resolve().is_relative_to(settings.data_path):
            raise ValueError(f'Reading symlink points outside DATA_PATH: {path.name}')
        filename = path.relative_to(settings.data_path).as_posix()
        if path.suffix.lower() == '.pdf':
            with path.open('rb') as source:
                pages = [(number, page.extract_text() or '') for number, page in enumerate(PdfReader(source).pages)]
        else:
            pages = [(0, path.read_text(encoding='utf-8'))]
        active_splitter = splitter if path.suffix.lower() == '.pdf' else markdown_splitter
        for page, text in pages:
            for number, content in enumerate(active_splitter.split_text(text)):
                if not content.strip():
                    continue
                digest = hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
                identifier = f'{filename}:{page}:{number}:{digest}'
                chunks.append(Document(page_content=content, metadata=dict(
                    source=filename, page=page, chunk=number, id=identifier)))
    if not chunks:
        raise ValueError('No readable text found. Check DATA_PATH; scanned PDFs need OCR before indexing.')
    return chunks


def build_index(settings, *, rebuild=False, embeddings=None):
    """Synchronize documents. Run with the web service stopped; failures stay unready."""
    directory = settings.chroma_path
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / '.building'
    if marker.exists():
        raise IndexNotReady('An index build is active or was interrupted. See docs/deployment.md before removing .building.')
    if not rebuild:
        if (directory / 'manifest.json').exists():
            check_manifest(settings)
        elif any(directory.iterdir()):
            raise IndexNotReady('An unmanaged index exists. Back it up and use build --rebuild to migrate.')
    # Read the whole corpus before changing any existing collection.
    chunks = read_chunks(settings)
    embedding_function = embeddings if embeddings is not None else get_embeddings(settings)
    with marker.open('x', encoding='utf-8') as lock:
        lock.write('Index build in progress. Do not serve this directory until it completes.\n')
    # On failure keep the marker so a partial index cannot pass readiness checks.
    if rebuild:
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        from chromadb.errors import NotFoundError
        client = chromadb.PersistentClient(path=str(directory), settings=ChromaSettings(anonymized_telemetry=False))
        try:
            client.delete_collection(COLLECTION)
        except NotFoundError:
            pass
    db = open_store(settings, create=True, embeddings=embedding_function)
    existing = set(db.get(include=[])['ids'])
    current = {chunk.metadata['id'] for chunk in chunks}
    additions = [chunk for chunk in chunks if chunk.metadata['id'] not in existing]
    for start in range(0, len(additions), 64):
        batch = additions[start:start + 64]
        db.add_documents(batch, ids=[chunk.metadata['id'] for chunk in batch])
        print(f'Embedded {min(start + 64, len(additions))}/{len(additions)} new chunks', flush=True)
    stale = sorted(existing - current)
    for start in range(0, len(stale), 64):
        db.delete(ids=stale[start:start + 64])
    write_manifest(settings, len(current))
    marker.unlink()
    print(f'Index ready: {len(current)} chunks; {len(additions)} added, {len(stale)} removed.', flush=True)


def calibrate(settings, samples):
    """Show a threshold sweep; labels test abstention and expected source files."""
    if not isinstance(samples, list) or not samples:
        raise ValueError('Calibration data must be a non-empty JSON array.')
    for sample in samples:
        if (not isinstance(sample, dict) or not isinstance(sample.get('question'), str)
                or not sample['question'].strip() or not isinstance(sample.get('relevant_sources'), list)
                or any(not isinstance(name, str) for name in sample['relevant_sources'])):
            raise ValueError('Each sample needs a question string and relevant_sources array (empty for unrelated questions).')
    if not any(sample['relevant_sources'] for sample in samples) or not any(not sample['relevant_sources'] for sample in samples):
        raise ValueError('Calibration needs both course questions and unrelated questions.')
    service = create_service(settings)
    service.ready()
    retrieved = [(sample, service.candidates(sample['question'])) for sample in samples]
    print('distance  correct_positive  missed_positive  wrong_source  unrelated_accepted')
    for step in range(1, 17):
        threshold = step / 20
        selector = RagService(replace(settings, max_distance=threshold), service.db)
        correct = missed = wrong = false_accepts = 0
        for sample, candidates in retrieved:
            selected = selector.select(candidates)
            files = {document.metadata['source'] for document, _ in selected}
            expected = set(sample['relevant_sources'])
            if not expected:
                false_accepts += bool(files)
            elif files & expected:
                correct += 1
            elif files:
                wrong += 1
            else:
                missed += 1
        print(f'{threshold:8.2f}  {correct:16d}  {missed:15d}  {wrong:12d}  {false_accepts:18d}')
    print('Choose MAX_DISTANCE using held-out questions, prioritizing few unrelated/wrong-source accepts.')
    print('This sweep does not measure answer accuracy or every individual source\'s relevance.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    build = commands.add_parser('build', help='Build or synchronize the readings (uses the embedding API).')
    build.add_argument('--rebuild', action='store_true', help='Replace the tldrracket collection; back it up and stop the service first.')
    probe = commands.add_parser('probe', help='Show distances and sources without generating an answer.')
    probe.add_argument('question')
    calibration = commands.add_parser('calibrate', help='Sweep cosine cutoffs over labeled questions without generation.')
    calibration.add_argument('samples', type=argparse.FileType('r', encoding='utf-8'))
    args = parser.parse_args()
    try:
        settings = Settings.from_env()
        if args.command == 'build':
            build_index(settings, rebuild=args.rebuild)
        elif args.command == 'probe':
            service = create_service(settings)
            service.ready()
            candidates = service.candidates(args.question)
            print(json.dumps([dict(service.source(document, distance), passes_cutoff=distance <= settings.max_distance)
                              for document, distance in candidates], ensure_ascii=False, indent=2))
        else:
            with args.samples as source:
                calibrate(settings, json.load(source))
    except (IndexNotReady, ValueError, OSError) as exc:
        parser.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
