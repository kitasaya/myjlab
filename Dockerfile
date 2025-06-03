# Pythonの公式イメージをベースイメージとして使用します。
# slim-busterは、軽量でDebianベースのOSです。
FROM python:3.9-slim-buster

# コンテナ内の作業ディレクトリを設定します。
# 以降のコマンドは、このディレクトリを基準に実行されます。
WORKDIR /app

# ホストのrequirements.txtファイルをコンテナの/appディレクトリにコピーします。
COPY requirements.txt .

# コピーしたrequirements.txtを使ってPythonの依存関係をインストールします。
# --no-cache-dir: キャッシュを使用しないことでイメージサイズを削減します。
# --upgrade pip: pipを最新版にアップグレードします。
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ホストのfrontendディレクトリの内容をコンテナの/app/frontendにコピーします。
COPY frontend/ ./frontend/

# ホストのmain.pyファイルをコンテナの/appディレクトリにコピーします。
COPY main.py .

# アプリケーションがリッスンするポートをDockerに通知します。
# これはドキュメント目的であり、ネットワークルールを強制するものではありません。
EXPOSE 8000

# コンテナが起動したときに実行されるコマンドを定義します。
# uvicornを使ってmain.pyのFastAPIアプリケーションを起動します。
# --host 0.0.0.0: 外部からのアクセスを許可するために必要です。
# --port 8000: アプリケーションがリッスンするポートです。
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]