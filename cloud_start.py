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
# A single preloaded Flask worker avoids background-job processes and duplicate imports.
server_env = os.environ.copy()
server_env.update({
    "_MLFLOW_SERVER_FILE_STORE": "sqlite:////app/seed/mlflow.db",
    "_MLFLOW_SERVER_SERVE_ARTIFACTS": "true",
    "_MLFLOW_SERVER_ARTIFACT_DESTINATION": "/app/seed/artifacts",
    "MLFLOW_SERVER_ALLOWED_HOSTS": f"{public_host},localhost,127.0.0.1",
    "MLFLOW_SERVER_CORS_ALLOWED_ORIGINS": f"https://{public_host}",
    "MLFLOW_SERVER_ENABLE_JOB_EXECUTION": "false",
    "MLFLOW_SERVER_JOB_ENABLE_PERIODIC_TASKS": "false",
    "MLFLOW_DISABLE_AGENT_HINT": "1",
})
mlflow_process = subprocess.Popen([
    sys.executable, "-m", "gunicorn", "--workers", "1", "--threads", "2",
    "--preload", "--timeout", "120", "--bind", "127.0.0.1:5000",
    "mlflow.server:app",
], env=server_env)

try:
    subprocess.run(['nginx','-c','/tmp/day3-nginx.conf','-g','daemon off;'],check=True)
finally:
    mlflow_process.terminate()
