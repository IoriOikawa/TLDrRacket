import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from langchain_core.embeddings import Embeddings

import rag
try:
    import index
except ModuleNotFoundError:
    index = None


class LocalEmbeddings(Embeddings):
    """Deterministic vectors: tests exercise real Chroma without a network call."""
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        return [1.0, 0.0, 0.0] if 'scope' in text.lower() else [0.0, 1.0, 0.0]


class IndexTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(index, 'The reproducible index CLI is missing')
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / 'data'
        self.data.mkdir()
        (self.data / 'notes.txt').write_text('Lexical scope uses the definition environment.', encoding='utf-8')
        self.settings = rag.Settings(data_path=self.data, chroma_path=self.root / 'chroma', embedding_dimensions=3)

    def test_chunks_have_stable_ids_and_changed_content_changes_ids(self):
        first = index.read_chunks(self.settings)
        second = index.read_chunks(self.settings)
        self.assertEqual(first[0].metadata['id'], second[0].metadata['id'])
        self.assertEqual(first[0].metadata['source'], 'notes.txt')
        (self.data / 'notes.txt').write_text('Dynamic scope is different.', encoding='utf-8')
        changed = index.read_chunks(self.settings)
        self.assertNotEqual(first[0].metadata['id'], changed[0].metadata['id'])


    def test_markdown_indented_code_keeps_leading_whitespace(self):
        text = '    (define (square x)\n      (* x x))\n'
        (self.data / 'notes.txt').write_text(text, encoding='utf-8')
        chunks = index.read_chunks(self.settings)
        self.assertEqual(chunks[0].page_content, text)

    def test_build_is_idempotent_and_removes_stale_chunks(self):
        embeddings = LocalEmbeddings()
        index.build_index(self.settings, embeddings=embeddings)
        db = rag.open_store(self.settings, embeddings=embeddings)
        self.assertEqual(db._collection.count(), 1)
        index.build_index(self.settings, embeddings=embeddings)
        self.assertEqual(db._collection.count(), 1)
        (self.data / 'notes.txt').write_text('Lazy evaluation delays computation.', encoding='utf-8')
        index.build_index(self.settings, embeddings=embeddings)
        result = db.get()
        self.assertEqual(len(result['ids']), 1)
        self.assertIn('Lazy evaluation', result['documents'][0])
        rag.check_manifest(self.settings)

    def test_real_cosine_search_returns_match_and_rejects_unrelated(self):
        index.build_index(self.settings, embeddings=LocalEmbeddings())
        db = rag.open_store(self.settings, embeddings=LocalEmbeddings())
        service = rag.RagService(self.settings, db)
        service.ready()
        self.assertEqual(len(service.select(service.candidates('scope'))), 1)
        self.assertEqual(service.select(service.candidates('weather')), [])

    def test_incompatible_index_requires_explicit_rebuild(self):
        index.build_index(self.settings, embeddings=LocalEmbeddings())
        changed = replace(self.settings, embedding_model='new-model')
        with self.assertRaises(rag.IndexNotReady):
            index.build_index(changed, embeddings=LocalEmbeddings())
        index.build_index(changed, rebuild=True, embeddings=LocalEmbeddings())
        rag.check_manifest(changed)

    def test_failed_build_is_not_ready(self):
        with patch.object(index, 'open_store', side_effect=RuntimeError('provider failed')):
            with self.assertRaises(RuntimeError):
                index.build_index(self.settings, embeddings=LocalEmbeddings())
        with self.assertRaises(rag.IndexNotReady):
            rag.check_manifest(self.settings)


    def test_calibration_rejects_incomplete_labels_without_embedding(self):
        bad_samples = ([], [{'question': 'scope', 'relevant_sources': ['notes.txt']}],
                       [{'question': 'weather', 'relevant_sources': []}],
                       [{'question': 'scope', 'relevant_sources': 'notes.txt'}])
        with patch.object(index, 'create_service') as factory:
            for samples in bad_samples:
                with self.subTest(samples=samples), self.assertRaises(ValueError):
                    index.calibrate(self.settings, samples)
            factory.assert_not_called()

    def test_empty_corpus_does_not_destroy_existing_index(self):
        index.build_index(self.settings, embeddings=LocalEmbeddings())
        (self.data / 'notes.txt').unlink()
        with self.assertRaises(ValueError):
            index.build_index(self.settings, rebuild=True, embeddings=LocalEmbeddings())
        rag.check_manifest(self.settings)


if __name__ == '__main__':
    unittest.main()
