# LND Routing Dashboard

LND Routing Dashboardは、LNDのfowarding履歴を解析し、チャンネルの状況を分析するためのダッシュボードです。
このプロジェクトは、Gradioを使用してインタラクティブなユーザーインターフェースを提供します。

## 機能

- ノード情報の表示(Kazumyon nodeの情報を表示しています。)
- ルーティングフローリストの表示
- ルーティングの可視化
- AIを使用したチャンネル分析

## インストール

1. このリポジトリをクローンします。

    git clone https://github.com/yourusername/LND_Routing.git
    cd LND_Routing

2. 仮想環境を作成し、必要なパッケージをインストールします。
   Poetry(1.5.1)での環境を使用しています。

    poetry install

## 使用方法

1. .envに"OPENAI_API_KEY"を設定(https://platform.openai.com/docs/overview)
2. main.pyを実行してダッシュボードを起動します。

    poetry run python main.py

3. ウェブブラウザで表示されるダッシュボードを使用して、LNDのfowarding履歴を解析します。

   ・　”Routing flow list"タブではLNDで取得したrouting forwarding dataを読み込みます。
   　　プルダウンlistでデータ(Kazumyon nodeのデータ)を選択するか、自身でLNDから取得したデータをアップロードします。

       Forwarding data取得方法（例）
         lncli fwdinghistory --start_time "-1w" --max_events 2000 >> fwd_history20250117.json
       channl balance data取得方法（例）
         lncli listchannels >> listchannels20250117.json

       自身のデータをプルダウンlistに付け加えるにはデータを./dataにおきます。
       Forwarding dataのファイルは”fwd_history”を名前に含まれるようにしてください。
       Forwarding dataに"listchannel_file"をキーとしてchannl balance dataデータのファイルをデータとして付け加えてください。

   　  正常にデータが読み込まれると各ノードからの入出力ノードのリストが表示されます
       データが読み込まれた状態で”Routing visualization”でRoutingが可視化できるようになります。
       データが読み込まれた状態で”AI analysis”でAIでRouting状況を分析できるようになります。
