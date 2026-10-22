"""Small single-process deployment for a persistent local Chroma index."""
bind = '127.0.0.1:8000'
workers = 1
worker_class = 'gthread'
threads = 4
# Two /api/ask calls may run at once; spare threads can answer health checks.
timeout = 120
graceful_timeout = 120
keepalive = 5
accesslog = '-'
errorlog = '-'
capture_output = True
