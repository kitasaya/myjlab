# Pythonの公式イメージをベースにする
FROM python:3.9-slim-buster

# 作業ディレクトリを設定
WORKDIR /app

# 必要なファイルをコンテナにコピー
COPY main.py .
COPY frontend/ ./frontend/

# 必要なシステムパッケージとPythonパッケージをインストール
# build-essential と pip install を同じ RUN 命令で実行することで、
# PATH の問題や依存関係の整合性を保ちやすくなります。
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir fastapi "uvicorn[standard]" python-multipart gunicorn

# アプリケーションがリッスンするポートを公開
EXPOSE 8000

# Docker ComposeでこのCMDを上書きするので、ここでは単体実行用として残しておく
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "main:app"]