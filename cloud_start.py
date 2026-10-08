"""Start MLflow behind a public read-only proxy on Render."""
import os
import subprocess
import sys
from pathlib import Path

port = int(os.environ.get('PORT', '10000'))
public_host = os.environ.get('RENDER_EXTERNAL_HOSTNAME', 'localhost')
config = '''events { worker_connections 128; }
http {
  access_log /dev/stdout;
  error_log /dev/stderr;
  server {
    listen PORT_NUMBER;
    location = /healthz { proxy_pass http://127.0.0.1:5000/health; }
    location ~ ^/api/2\\.0/mlflow/(experiments|runs|registered-models|model-versions|logged-models)/search$ {
      limit_except GET POST { deny all; }
      proxy_pass http://127.0.0.1:5000;
      proxy_set_header Host $host;
      proxy_set_header X-Forwarded-Proto $scheme;
    }
    location / {
      limit_except GET { deny all; }
      proxy_pass http://127.0.0.1:5000;
      proxy_set_header Host $host;
      proxy_set_header X-Forwarded-Proto $scheme;
    }
  }
}
'''.replace('PORT_NUMBER',str(port))
Path('/tmp/day3-nginx.conf').write_text(config)
mlflow_process = subprocess.Popen([
    sys.executable,'-m','mlflow','server',
    '--backend-store-uri','sqlite:////app/seed/mlflow.db',
    '--host','127.0.0.1','--port','5000','--workers','1',
    '--allowed-hosts',f'{public_host},localhost,127.0.0.1',
    '--cors-allowed-origins',f'https://{public_host}',
])
try:
    subprocess.run(['nginx','-c','/tmp/day3-nginx.conf','-g','daemon off;'],check=True)
finally:
    mlflow_process.terminate()
