# uv インストール手順書（Windows）

## 概要
uvは高速なPythonパッケージマネージャーです。この手順書では、Windows環境でのインストールと環境変数の設定方法を説明します。

## 前提条件
- Windows 10/11
- PowerShellまたはコマンドプロンプトが使用可能

## インストール手順

### 1. uvのインストール

PowerShellを開いて、以下のコマンドを実行します：

```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

**実行結果の例：**
```
Downloading uv 0.9.5 (x86_64-pc-windows-msvc)
Installing to C:\Users\<ユーザー名>\.local\bin
  uv.exe
  uvx.exe
  uvw.exe
everything's installed!
```

### 2. インストール確認

インストールされたuvのバージョンを確認します（フルパスで実行）：

```powershell
C:\Users\<ユーザー名>\.local\bin\uv.exe --version
```

**期待される出力：**
```
uv 0.9.5 (d5f39331a 2025-10-21)
```

## 環境変数の設定

### 方法1: PowerShellで自動設定（推奨）

PowerShellで以下のコマンドを実行して、PATHに永続的に追加します：

```powershell
[Environment]::SetEnvironmentVariable('Path', [Environment]::GetEnvironmentVariable('Path', 'User') + ';C:\Users\<ユーザー名>\.local\bin', 'User')
```

**注意:** `<ユーザー名>` は実際のユーザー名に置き換えてください。

### 方法2: GUIで手動設定

1. Windowsキー + Rを押して「ファイル名を指定して実行」を開く
2. `sysdm.cpl` と入力してEnter
3. 「詳細設定」タブを選択
4. 「環境変数」ボタンをクリック
5. 「ユーザー環境変数」セクションで「Path」を選択
6. 「編集」ボタンをクリック
7. 「新規」をクリック
8. 以下のパスを追加：
   ```
   C:\Users\<ユーザー名>\.local\bin
   ```
9. 「OK」をクリックしてすべてのダイアログを閉じる

### 3. 設定の反映

**重要:** 環境変数の変更を反映させるため、以下のいずれかを実行してください：

- **新しいPowerShell/コマンドプロンプトウィンドウを開く**（推奨）
- または、コンピューターを再起動する

### 4. 動作確認

新しいターミナルを開いて、以下のコマンドを実行します：

```powershell
uv --version
```

**期待される出力：**
```
uv 0.9.5 (d5f39331a 2025-10-21)
```

パスなしで `uv` コマンドが実行できれば、環境変数の設定は成功です。

## トラブルシューティング

### uvコマンドが見つからない場合

1. 新しいターミナルを開いているか確認
2. PATHが正しく設定されているか確認：
   ```powershell
   [Environment]::GetEnvironmentVariable('Path', 'User')
   ```
3. `C:\Users\<ユーザー名>\.local\bin` が含まれているか確認

### 一時的にPATHを追加する場合

現在のセッションのみで使用する場合（再起動すると消える）：

**PowerShell:**
```powershell
$env:Path = "C:\Users\<ユーザー名>\.local\bin;$env:Path"
```

**コマンドプロンプト:**
```cmd
set Path=C:\Users\<ユーザー名>\.local\bin;%Path%
```

## 基本的な使い方

```powershell
# ヘルプを表示
uv --help

# Pythonパッケージのインストール
uv pip install <パッケージ名>

# 仮想環境の作成
uv venv

# uvのバージョン確認
uv --version
```

## 参考リンク

- 公式サイト: https://github.com/astral-sh/uv
- ドキュメント: https://docs.astral.sh/uv/

## 更新履歴

- 2025-10-23: 初版作成（uv 0.9.5対応）
