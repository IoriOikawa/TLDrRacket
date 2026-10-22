import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import rag
try:
    import webserver
except (FileNotFoundError, ModuleNotFoundError):
    webserver = None


class WebTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(hasattr(webserver, 'create_app'), 'The testable Flask application factory is missing')
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'notes.pdf').write_bytes(b'%PDF-1.4\n')
        self.settings = rag.Settings(data_path=self.root, chroma_path=self.root / 'index')
        self.service = Mock()
        self.service.ask.return_value = dict(response='Grounded answer', sources=[], status='no_match')
        self.factory = Mock(return_value=self.service)
        self.app = webserver.create_app(self.settings, self.factory)
        self.app.config['TESTING'] = True
        self.app.logger.disabled = True
        self.client = self.app.test_client()

    def test_home_and_health_do_not_initialize_provider(self):
        with self.client.get('/') as response:
            self.assertEqual(response.status_code, 200)
        self.assertEqual(self.client.get('/healthz').status_code, 200)
        self.factory.assert_not_called()

    def test_invalid_inputs_never_reach_provider(self):
        for payload in (None, [], {'message': None}, {'message': 123}, {'message': '  '}, {'message': 'a' * 2001}):
            with self.subTest(payload=str(payload)[:50]):
                self.assertEqual(self.client.post('/api/ask', json=payload).status_code, 400)
        self.assertEqual(self.client.post('/api/ask', data='{bad', content_type='application/json').status_code, 400)
        self.factory.assert_not_called()

    def test_answer_is_returned_and_service_is_reused(self):
        for _ in range(2):
            response = self.client.post('/api/ask', json={'message': '  scope  '})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json['status'], 'no_match')
        self.factory.assert_called_once_with(self.settings)
        self.service.ask.assert_called_with('scope')
        self.assertIn('no-store', response.headers['Cache-Control'])

    def test_provider_failure_does_not_leak_details_and_next_request_works(self):
        self.service.ask.side_effect = RuntimeError('secret-provider-details')
        response = self.client.post('/api/ask', json={'message': 'scope'})
        self.assertEqual(response.status_code, 502)
        self.assertNotIn('secret-provider-details', response.get_data(as_text=True))
        self.service.ask.side_effect = None
        self.assertEqual(self.client.post('/api/ask', json={'message': 'scope'}).status_code, 200)


    def test_concurrency_bound_rejects_third_question_and_releases_slots(self):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Barrier, Event
        entered, release = Barrier(3), Event()
        def blocking(_message):
            entered.wait(timeout=5)
            if not release.wait(timeout=5):
                raise TimeoutError()
            return dict(response='answer', sources=[], status='ok')
        def question():
            with self.app.test_client() as client:
                return client.post('/api/ask', json={'message': 'scope'}).status_code
        self.service.ask.side_effect = blocking
        with ThreadPoolExecutor(max_workers=2) as executor:
            calls = [executor.submit(question) for _ in range(2)]
            try:
                entered.wait(timeout=5)
                self.assertEqual(self.client.post('/api/ask', json={'message': 'scope'}).status_code, 429)
                self.assertEqual(self.client.get('/healthz').status_code, 200)
            finally:
                release.set()
            self.assertEqual([future.result(timeout=5) for future in calls], [200, 200])
        self.service.ask.side_effect = None
        self.assertEqual(question(), 200)

    def test_oversized_body_is_rejected_before_provider_initialization(self):
        response = self.client.post('/api/ask', json={'message': 'x' * 20000})
        self.assertEqual(response.status_code, 413)
        self.factory.assert_not_called()

    def test_provider_timeout_is_reported(self):
        self.service.ask.side_effect = TimeoutError()
        self.assertEqual(self.client.post('/api/ask', json={'message': 'scope'}).status_code, 504)

    def test_not_ready_is_reported_without_initializing_on_health(self):
        self.factory.side_effect = rag.IndexNotReady('Run python index.py build first.')
        self.assertEqual(self.client.get('/readyz').status_code, 503)
        response = self.client.post('/api/ask', json={'message': 'scope'})
        self.assertEqual(response.status_code, 503)
        self.assertIn('index', response.json['error'])

    def test_documents_serve_pdf_but_reject_traversal_and_symlinks(self):
        with self.client.get('/documents/notes.pdf') as response:
            self.assertEqual(response.status_code, 200)
        for url in ('/documents/../webserver.py', '/documents/%2e%2e/webserver.py', '/documents/no.pdf', '/documents/manifest.json'):
            self.assertEqual(self.client.get(url).status_code, 404)
        with tempfile.TemporaryDirectory() as outside:
            target = Path(outside) / 'private.pdf'
            target.write_bytes(b'secret')
            (self.root / 'symlink.pdf').symlink_to(target)
            self.assertEqual(self.client.get('/documents/symlink.pdf').status_code, 404)


if __name__ == '__main__':
    unittest.main()
