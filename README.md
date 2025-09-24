# Flask API Project

これは、Poetryで管理された最小構成のFlask APIバックエンドです。

## 特徴

-   **パッケージ管理:** `Poetry`
-   **フレームワーク:** `Flask`
-   **コードフォーマッタ:** `Black`
-   **リンター:** `Flake8`
-   **設定管理:** 環境変数 (`.env`)

---

## 開発環境の推奨設定 (VS Code)
このプロジェクトは、**ファイルを保存するだけ(Ctrl+S)で、自動的にコードが整形される**ように設定されています。手動でフォーマットコマンドを実行する必要はありません。

この機能を有効にするため、[Visual Studio Code](https://code.visualstudio.com/)に以下の拡張機能をインストールしてください。

[Python (ms-python.python)](https://marketplace.visualstudio.com/items?itemName=ms-python.python) - Microsoft公式

[Ruff (charliermarsh.ruff)](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) - Ruff公式

拡張機能をインストール後、このプロジェクトを開くと、同梱されている`.vscode/settings.json`が自動的に読み込まれ、保存時の自動整形が有効になります。

## セットアップ (Setup)

### 前提条件

-   Python 3.12以上
-   [Poetry](https://python-poetry.org/docs/#installation)

### 手順

1.  **リポジトリをクローン**
    ```bash
    git clone <your-repository-url>
    cd tourists
    ```

2.  **`.env`ファイルを作成**
    `.env.example`をコピーして`.env`ファイルを作成します。`SECRET_KEY`は必要に応じて変更してください。
    ```bash
    cp .env.example .env
    ```

3.  **依存関係をインストール**
    Poetryを使って、`pyproject.toml`に記述されたパッケージをインストールします。
    ```bash
    poetry install
    ```

---

## 実行 (Running the Application)

1.  **仮想環境を有効化**
    Poetryが作成した仮想環境に入ります。
    ```bash
    poetry shell
    ```

2.  **開発サーバーを起動**
    FlaskのCLIを使って開発サーバーを起動します。`.env`ファイルが自動で読み込まれます。
    ```bash
    flask run
    ```

    サーバーが `http://127.0.0.1:5000` で起動します。

### APIエンドポイントの確認

-   **ヘルスチェック:**
    ```bash
    curl [http://127.0.0.1:5000/](http://127.0.0.1:5000/)
    # Expected output: {"status":"ok"}
    ```
-   **アイテムリスト取得(例):**
    ```bash
    curl [http://127.0.0.1:5000/api/items](http://127.0.0.1:5000/api/items)
    # Expected output: [{"id":1,"name":"Item 1"},{"id":2,"name":"Item 2"}]
    ```

---