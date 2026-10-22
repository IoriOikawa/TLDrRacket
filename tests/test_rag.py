import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

try:
    import rag
except ModuleNotFoundError:
    rag = None


def doc(name='notes.pdf', page=0, text='Lexical scope uses the definition environment.'):
    return SimpleNamespace(page_content=text, metadata={'source': name, 'page': page, 'id': f'{name}:{page}:0'})


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(rag, 'The shared RAG module has not been implemented')
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.settings = rag.Settings(data_path=Path(self.tmp.name), chroma_path=Path(self.tmp.name) / 'index')
        self.db = Mock()
        self.db._collection.count.return_value = 10
        self.model = Mock()
        self.model.invoke.return_value = SimpleNamespace(content='A grounded answer [1].')
        self.service = rag.RagService(self.settings, self.db, self.model)

    def test_unrelated_results_produce_empty_sources_without_generation(self):
        self.db.similarity_search_with_score.return_value = [(doc(), 0.95)]
        result = self.service.ask('What is the weather?')
        self.assertEqual(result['sources'], [])
        self.assertEqual(result['status'], 'no_match')
        self.model.invoke.assert_not_called()

    def test_empty_database_does_not_embed_or_generate(self):
        self.db._collection.count.return_value = 0
        with self.assertRaises(rag.IndexNotReady):
            self.service.ask('What is scope?')
        self.db.similarity_search_with_score.assert_not_called()
        self.model.invoke.assert_not_called()

    def test_boundary_distance_is_included_and_larger_distance_is_not(self):
        self.db.similarity_search_with_score.return_value = [
            (doc(page=0), self.settings.max_distance),
            (doc(page=1), self.settings.max_distance + 0.001),
            (doc(page=2), float('nan')),
        ]
        result = self.service.ask('What is scope?')
        self.assertEqual(len(result['sources']), 1)
        self.assertEqual(result['sources'][0]['page'], 1)

    def test_one_search_supplies_both_prompt_and_sources(self):
        self.db.similarity_search_with_score.return_value = [(doc(text='UNIQUE_CONTEXT'), 0.1)]
        result = self.service.ask('What is scope?')
        self.db.similarity_search_with_score.assert_called_once()
        prompt = str(self.model.invoke.call_args)
        self.assertIn('UNIQUE_CONTEXT', prompt)
        self.assertEqual(result['sources'][0]['preview'], 'UNIQUE_CONTEXT')
        self.assertIn('/documents/notes.pdf#page=1', result['sources'][0]['url'])

    def test_duplicate_pages_keep_best_chunk_and_cap_five(self):
        self.db.similarity_search_with_score.return_value = [
            (doc(page=0, text='worse'), 0.2), (doc(page=0, text='best'), 0.05),
            *[(doc(page=i), 0.1) for i in range(1, 9)],
        ]
        result = self.service.ask('What is scope?')
        self.assertEqual(len(result['sources']), 5)
        self.assertEqual(result['sources'][0]['preview'], 'best')
        self.assertNotIn('worse', str(self.model.invoke.call_args))

    def test_unsafe_source_metadata_never_becomes_a_link_or_context(self):
        self.db.similarity_search_with_score.return_value = [(doc('../private.pdf'), 0.1)]
        self.assertEqual(self.service.ask('scope')['sources'], [])
        self.model.invoke.assert_not_called()


    def test_markdown_note_content_is_returned_verbatim(self):
        markdown = "# Closures\n\n- Captured environment\n\n    (lambda (x)\n      (+ x 1))\n"
        self.db.similarity_search_with_score.return_value = [(doc('5400Notes.txt', text=markdown), 0.1)]
        source = self.service.ask('What is a closure?')['sources'][0]
        self.assertEqual(source.get('content'), markdown)
        self.assertEqual(source.get('format'), 'markdown')
        self.assertIsNone(source['page'])

    def test_pdf_excerpts_stay_plain_text(self):
        text = '# This is extracted PDF text\n    indentation is preserved'
        self.db.similarity_search_with_score.return_value = [(doc(text=text), 0.1)]
        source = self.service.ask('scope')['sources'][0]
        self.assertEqual(source.get('content'), text)
        self.assertEqual(source.get('format'), 'text')

    def test_empty_model_response_is_an_error(self):
        self.db.similarity_search_with_score.return_value = [(doc(), 0.1)]
        self.model.invoke.return_value = SimpleNamespace(content='')
        with self.assertRaises(rag.GenerationFailed):
            self.service.ask('scope')


class SettingsTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(rag, 'The shared RAG module has not been implemented')

    def test_invalid_thresholds_fail_at_startup(self):
        for value in (-1, 3, float('nan')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                rag.Settings(max_distance=value)

    def test_index_manifest_rejects_changed_embedding_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = rag.Settings(chroma_path=Path(tmp))
            rag.write_manifest(settings, chunk_count=1)
            rag.check_manifest(settings)
            from dataclasses import replace
            with self.assertRaises(rag.IndexNotReady):
                rag.check_manifest(replace(settings, embedding_model='different-model'))


    def test_legacy_whitespace_stripped_index_requires_migration(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            settings = rag.Settings(chroma_path=Path(tmp))
            rag.write_manifest(settings, chunk_count=1)
            path = Path(tmp) / 'manifest.json'
            manifest = json.loads(path.read_text())
            manifest['schema_version'] = 1
            path.write_text(json.dumps(manifest))
            with self.assertRaises(rag.IndexNotReady):
                rag.check_manifest(settings)

    def test_missing_or_building_index_is_not_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = rag.Settings(chroma_path=Path(tmp))
            with self.assertRaises(rag.IndexNotReady):
                rag.check_manifest(settings)
            rag.write_manifest(settings, chunk_count=1)
            (Path(tmp) / '.building').touch()
            with self.assertRaises(rag.IndexNotReady):
                rag.check_manifest(settings)


class ProviderTimeoutTests(unittest.TestCase):
    def test_embedding_transport_receives_a_real_finite_timeout(self):
        import httpx
        with patch.dict('os.environ', {'GOOGLE_API_KEY': 'test-key-for-offline-transport'}):
            embeddings = rag.get_embeddings(rag.Settings(provider_timeout=7))
        response = httpx.Response(200, json={'embeddings': [{'values': [1.0, 0.0]}]})
        with patch.object(embeddings.client._api_client._httpx_client, 'send', return_value=response) as send:
            embeddings.embed_query('scope')
        actual = send.call_args.args[0].extensions['timeout']
        self.assertEqual(actual['connect'], 7.0)
        self.assertEqual(actual['read'], 7.0)
        embeddings.client.close()


if __name__ == '__main__':
    unittest.main()
