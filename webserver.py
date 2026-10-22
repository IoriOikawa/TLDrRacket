"""Flask HTTP entry point. Importing this module makes no provider requests."""

import logging
from threading import BoundedSemaphore, Lock

import httpx
from flask import Flask, abort, jsonify, request, send_file
from werkzeug.exceptions import RequestEntityTooLarge

from rag import ROOT, GenerationFailed, IndexNotReady, Settings, create_service


def create_app(settings=None, service_factory=create_service):
    settings = settings or Settings.from_env()
    app = Flask(__name__, static_url_path='', static_folder=str(ROOT / 'static'))
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024
    service = None
    service_lock = Lock()
    slots = BoundedSemaphore(2)

    def get_service():
        nonlocal service
        with service_lock:
            if service is None:
                service = service_factory(settings)
        return service

    @app.after_request
    def response_headers(response):
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'same-origin'
        if request.path.startswith('/api/') or request.path in {'/healthz', '/readyz'}:
            response.headers['Cache-Control'] = 'no-store'
        return response

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_error):
        return jsonify(error='The request body is too large.'), 413

    @app.get('/')
    def index():
        return app.send_static_file('index.html')

    @app.get('/healthz')
    def health():
        return jsonify(status='ok')

    @app.get('/readyz')
    def ready():
        try:
            get_service().ready()
            return jsonify(status='ready')
        except IndexNotReady as exc:
            return jsonify(status='not_ready', error=str(exc)), 503
        except Exception:
            app.logger.exception('Index readiness check failed')
            return jsonify(status='not_ready', error='The reading index could not be opened.'), 503

    @app.get('/documents/<path:filename>')
    def document(filename):
        path = (settings.data_path / filename).resolve()
        if (not path.is_relative_to(settings.data_path)
                or path.suffix.lower() not in {'.pdf', '.txt'} or not path.is_file()):
            abort(404)
        return send_file(path, conditional=True)

    @app.post('/api/ask')
    def ask():
        data = request.get_json(silent=True)
        message = data.get('message') if isinstance(data, dict) else None
        if not isinstance(message, str) or not message.strip():
            return jsonify(error='Please provide a non-empty message string.'), 400
        message = message.strip()
        if len(message) > settings.max_message_length:
            return jsonify(error=f'Messages must be at most {settings.max_message_length} characters.'), 400
        if not slots.acquire(blocking=False):
            return jsonify(error='The assistant is busy. Please try again shortly.'), 429
        try:
            current = get_service()
            current.ready()
            return jsonify(current.ask(message))
        except IndexNotReady as exc:
            return jsonify(error=str(exc)), 503
        except GenerationFailed as exc:
            return jsonify(error=str(exc)), 502
        except Exception as exc:
            app.logger.exception('Question processing failed')
            # Integration libraries can wrap transport timeouts in provider errors.
            cause, visited = exc, set()
            while cause is not None and id(cause) not in visited:
                visited.add(id(cause))
                if isinstance(cause, (TimeoutError, httpx.TimeoutException)):
                    return jsonify(error='The model request timed out. Please try again.'), 504
                cause = cause.__cause__ or cause.__context__
            return jsonify(error='The assistant could not complete this request. Please try again.'), 502
        finally:
            slots.release()

    return app


app = create_app()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(host='127.0.0.1', port=5000)
