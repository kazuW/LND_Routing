from openai import OpenAI
from pydantic import BaseModel

import networkx as nx
import matplotlib.pyplot as plt
import scipy as sp

from dotenv import load_dotenv
import os
import json
import gradio as gr
import pandas as pd
import sys
from datetime import datetime, timezone, timedelta

load_dotenv()

"""
class Channel(BaseModel):
    alias: str
    ave_fee: float
    sigma: float

class ChannelDetail(BaseModel):
    peer_alias: str
    input_peer: list[Channel]
    output_peer: list[Channel]

class ChannelList(BaseModel):
    channel_peer: list[ChannelDetail]
"""

class channel:
    def __init__(self, alias: str, ave_fee: float, ave_min: float, ave_max: float, amt_sat: float, count: int):
        self.alias = alias
        self.ave_fee = ave_fee
        self.ave_min = ave_min
        self.ave_max = ave_max
        self.amt_sat = amt_sat
        self.count = count

    def __str__(self):
        return json.dumps(self.__dict__, ensure_ascii=False)

    def to_dict(self):
        return self.__dict__

class channel_detail:
    def __init__(self, peer_alias: str, input_peer: list[channel], output_peer: list[channel], input_amt_sat: float, output_amt_sat: float, capacity: int, local_balance_ratio: float):
        self.peer_alias = peer_alias
        self.input_peer = input_peer
        self.output_peer = output_peer
        self.input_amt_sat = input_amt_sat
        self.output_amt_sat = output_amt_sat
        self.capacity = capacity
        self.local_balance_ratio = local_balance_ratio

    def __str__(self):
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def to_dict(self):
        return {
            "peer_alias": self.peer_alias,
            "input_peer": [peer.to_dict() for peer in self.input_peer],
            "output_peer": [peer.to_dict() for peer in self.output_peer],
            "input_amt_sat": self.input_amt_sat,
            "output_amt_sat": self.output_amt_sat,
            "capacity": self.capacity,
            "local_balance_ratio": self.local_balance_ratio
        }

class ChannelList:
    def __init__(self, channel_peer: list[channel_detail], analysis_result: str, start_time: str, end_time: str, channel_num: int):
        self.channel_peer = channel_peer
        self.start_time = start_time
        self.end_time = end_time
        self.analysis_result = analysis_result
        self.channel_num = channel_num

    def __str__(self):
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def to_dict(self):
        return {
            "channel_peer": [peer.to_dict() for peer in self.channel_peer],
            "analysis_result": self.analysis_result,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "channel_num": self.channel_num
        }

RE_DATA_GLOBAL = None
CHAN_DATA_GLOBAL = None

# Unixタイムスタンプを日本時間に変換する関数
def convert_to_jst(unix_timestamp):
    # Unixタイムスタンプをdatetimeオブジェクトに変換
    dt_utc = datetime.fromtimestamp(int(unix_timestamp), tz=timezone.utc)
    # 日本時間（UTC+9）に変換
    dt_jst = dt_utc.astimezone(timezone(timedelta(hours=9)))
    return dt_jst.strftime('%Y-%m-%d %Y-%m-%d %H:%M:%S')


# ディレクトリ内のファイルを抽出
data_dir = "./data"
files = [f for f in os.listdir(data_dir) if f.startswith("fwd_history")]

# fowarding履歴のJSONファイルを読み込む
def read_fwd_data(fwd_data_file):
    with open(fwd_data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        fwd_data = data["forwarding_events"]
        channel_file = data["listchannel_file"]
        #print("fowarding Data数 : ",len(fwd_data))

    # fowarding履歴のalias名がない場合、chan_id_in, chan_id_outに置き換える
    for i in range(0, len(fwd_data)):
        if "edge not found" in fwd_data[i]["peer_alias_in"]:
            #print("edge not found input")
            fwd_data[i]["peer_alias_in"] = "n" + fwd_data[i]["chan_id_in"]
        if "edge not found" in fwd_data[i]["peer_alias_out"]:
            #print("edge not found output")
            fwd_data[i]["peer_alias_out"] = "n" + fwd_data[i]["chan_id_out"]
            #print(fwd_data[i]["peer_alias_in"])
    
    return fwd_data, channel_file

# fowarding履歴のJSONファイルをクライアントから読み込む
def read_fwd_data_ext(fwd_data_file):
    with open(fwd_data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        fwd_data = data["forwarding_events"]
        #channel_file = data["listchannel_file"]
        #print("fowarding Data数 : ",len(fwd_data))

    # fowarding履歴のalias名がない場合、chan_id_in, chan_id_outに置き換える
    for i in range(0, len(fwd_data)):
        if "edge not found" in fwd_data[i]["peer_alias_in"]:
            #print("edge not found input")
            fwd_data[i]["peer_alias_in"] = "n" + fwd_data[i]["chan_id_in"]
        if "edge not found" in fwd_data[i]["peer_alias_out"]:
            #print("edge not found output")
            fwd_data[i]["peer_alias_out"] = "n" + fwd_data[i]["chan_id_out"]
            #print(fwd_data[i]["peer_alias_in"])
    
    return fwd_data

# 接続してるchanel状態のJSONファイルを読み込む
def read_channel_data(chan_data_file):
    with open(chan_data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        channel_data = data["channels"]
   
    return channel_data

# データを再構築する
def reStructure_data(fwd_data):
    data = []
    for i in range(0, len(fwd_data)):
        data.append(fwd_data[i]["peer_alias_in"])
        data.append(fwd_data[i]["peer_alias_out"])
    # 重複を削除
    data = list(set(data))

    re_data = ChannelList(channel_peer=[], analysis_result="", start_time="", end_time="", channel_num=0)
    re_data.start_time = convert_to_jst(fwd_data[0]["timestamp"])
    re_data.end_time = convert_to_jst(fwd_data[-1]["timestamp"])

    # チャンネルのpeer_aliasを取得
    for i in range(0, len(data)):
        re_data.channel_peer.append(channel_detail(data[i], [], [], 0, 0, 0, 0.0))
        for j in range(0, len(fwd_data)):
            if fwd_data[j]["peer_alias_in"] == data[i] and fwd_data[j]["peer_alias_out"] not in [peer.alias for peer in re_data.channel_peer[i].output_peer]:
                re_data.channel_peer[i].output_peer.append(channel(fwd_data[j]["peer_alias_out"], 0, float('inf'), float('-inf'), 0, 0))
            if fwd_data[j]["peer_alias_out"] == data[i] and fwd_data[j]["peer_alias_in"] not in [peer.alias for peer in re_data.channel_peer[i].input_peer]:
                re_data.channel_peer[i].input_peer.append(channel(fwd_data[j]["peer_alias_in"], 0, float('inf'), float('-inf'), 0, 0))
 
    # チャンネルのpeer_aliasの追加情報を取得
    for i in range(0, len(re_data.channel_peer)):
        for j in range(0, len(fwd_data)):
            if fwd_data[j]["peer_alias_in"] == re_data.channel_peer[i].peer_alias:
                for k in range(0, len(re_data.channel_peer[i].output_peer)):
                    if fwd_data[j]["peer_alias_out"] == re_data.channel_peer[i].output_peer[k].alias:
                        count = re_data.channel_peer[i].output_peer[k].count + 1
                        before_fee = re_data.channel_peer[i].output_peer[k].ave_fee
                        ave_fee = float(fwd_data[j]["fee"])
                        re_data.channel_peer[i].output_peer[k].ave_fee = (before_fee * (count - 1) + ave_fee) / count
                        re_data.channel_peer[i].output_peer[k].count = count
                        re_data.channel_peer[i].output_peer[k].amt_sat += float(fwd_data[j]["amt_in"])
                        if count == 1:
                            re_data.channel_peer[i].output_peer[k].ave_min = ave_fee
                        else:
                            re_data.channel_peer[i].output_peer[k].ave_min = min(re_data.channel_peer[i].output_peer[k].ave_min, float(fwd_data[j]["fee"]))
                        re_data.channel_peer[i].output_peer[k].ave_max = max(re_data.channel_peer[i].output_peer[k].ave_max, float(fwd_data[j]["fee"]))      
            if fwd_data[j]["peer_alias_out"] == re_data.channel_peer[i].peer_alias:
                for k in range(0, len(re_data.channel_peer[i].input_peer)):
                    if fwd_data[j]["peer_alias_in"] == re_data.channel_peer[i].input_peer[k].alias:
                        re_data.channel_peer[i].input_peer[k].count += 1
                        re_data.channel_peer[i].input_peer[k].amt_sat += float(fwd_data[j]["amt_out"])

    for i in range(0, len(re_data.channel_peer)):
        re_data.channel_peer[i].input_amt_sat = sum([peer.amt_sat for peer in re_data.channel_peer[i].input_peer])
        re_data.channel_peer[i].output_amt_sat = sum([peer.amt_sat for peer in re_data.channel_peer[i].output_peer])

    re_data.channel_num = len(re_data.channel_peer)

    return re_data

def print_data(re_data):

    print("##################################################")
    for i in range(0, len(re_data.channel_peer)):
        print(re_data.channel_peer[i].peer_alias, re_data.channel_peer[i].capacity, re_data.channel_peer[i].local_balance_ratio, re_data.channel_peer[i].input_amt_sat, re_data.channel_peer[i].output_amt_sat)
        for j in range(0, len(re_data.channel_peer[i].input_peer)):
            print("  input_peer: ", re_data.channel_peer[i].input_peer[j].alias, re_data.channel_peer[i].input_peer[j].ave_fee, re_data.channel_peer[i].input_peer[j].amt_sat)
        for j in range(0, len(re_data.channel_peer[i].output_peer)):
            print("  output_peer: ", re_data.channel_peer[i].output_peer[j].alias, re_data.channel_peer[i].output_peer[j].ave_fee, re_data.channel_peer[i].output_peer[j].amt_sat)

    print("##################################################")
    print("start_time: ", re_data.start_time)
    print("end_time: ", re_data.end_time)
    print("channel number: ", re_data.channel_num)

# データにチャンネル情報を追加する
def add_other_data(re_data, channel_data):
    for i in range(0, len(re_data.channel_peer)):
        for j in range(0, len(channel_data)):
            if re_data.channel_peer[i].peer_alias == channel_data[j]["peer_alias"] or re_data.channel_peer[i].peer_alias[:10] == channel_data[j]["remote_pubkey"][:10]:
                re_data.channel_peer[i].capacity += int(channel_data[j]["capacity"])
                re_data.channel_peer[i].local_balance_ratio = float(channel_data[j]["local_balance"]) / float(channel_data[j]["capacity"]) * 100
                re_data.channel_peer[i].local_balance_ratio = round(re_data.channel_peer[i].local_balance_ratio, 2)
    return re_data

# 表示用のデータを整形する関数
def format_data(data):
    rows = []
    for channel in data['channel_peer']:
        input_peers = ""
        output_peers = ""
        input_amt_sats = 0
        output_amt_sat = 0
        for input_peer in channel['input_peer']:
            ratio = round(input_peer['amt_sat'] / channel['input_amt_sat'] * 100, 2)
            ave_fee = round(input_peer['ave_fee'], 2)
            input_peers += input_peer['alias'] + ":(" + str(ave_fee) + ")(" + str(ratio) + "%)<br>"
        for output_peer in channel['output_peer']:
            ratio = round(output_peer['amt_sat'] / channel['output_amt_sat'] * 100, 2)
            ave_fee = round(output_peer['ave_fee'], 2)
            output_peers += output_peer['alias'] + ":(" + str(ave_fee) + ")(" + str(ratio) + "%)<br>"
        
        row = (
                channel['peer_alias'].replace('|', '\\|'),
                channel['capacity'],
                channel['local_balance_ratio'],
                input_peers.replace('|', '\\|'),
                channel['input_amt_sat'],
                output_peers.replace('|', '\\|'),
                channel['output_amt_sat']
            )
        rows.append(row)
    return rows

def format_data_as_markdown(data):
    formatted_data = format_data(data)
    #print(formatted_data)
    headers = ["Peer Alias", "Capacity", "Local Balance Ratio(%)", "Input Peer Alias:(ave fee)(share)", "Input amt sat", "Output Peer Alias:(ave fee)(share)", "Output amt sat"]
    markdown = "| " + " | ".join(headers) + " |\n"
    markdown += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    for row in formatted_data:
        markdown += "| " + " | ".join(map(str, row)) + " |\n"
    return markdown

# 表示用のデータをDataFrameに変換する関数
def display_table(data, sort_by=None, reverse_sort=False):
    if sort_by:
        data['channel_peer'].sort(key=lambda x: x[sort_by], reverse=reverse_sort)
    markdown = format_data_as_markdown(data)
    return gr.Markdown(markdown)

# Gradioで表を表示する
def update_table(file_name, sort_by, reverse_sort, use_client_files, fwd_data_file, chan_data_file):
    re_data, chan_data = load_file(data_dir, file_name, use_client_files, fwd_data_file, chan_data_file)
    data = re_data.to_dict()
    data_period = f"<h3>・Data period: {re_data.start_time} ~ {re_data.end_time}</h3>"
    routing_peer_num = f"<h3>・Routing peer number: {len(re_data.channel_peer)}/{len(chan_data)}</h3>"
    table_markdown = display_table(data, sort_by, reverse_sort)
    return data_period, routing_peer_num, table_markdown

# ファイルを読み込む関数
def load_file(data_dir, file_name, use_client_files, fwd_data_file=None, chan_data_file=None):
    global RE_DATA_GLOBAL, CHAN_DATA_GLOBAL
    if use_client_files:
        selected_fwd_data_file = fwd_data_file.name
        selected_chan_data_file = chan_data_file.name
        fwd_data = read_fwd_data_ext(fwd_data_file.name)
        re_data = reStructure_data(fwd_data)
        chan_data = read_channel_data(chan_data_file.name)
        re_data = add_other_data(re_data, chan_data)
        RE_DATA_GLOBAL = re_data
        CHAN_DATA_GLOBAL = chan_data
    else:
        file_path = os.path.join(data_dir, file_name)
        fwd_data, channel_data_file = read_fwd_data(file_path)
        re_data = reStructure_data(fwd_data)
        file_path = os.path.join(data_dir, channel_data_file)
        chan_data = read_channel_data(file_path)
        re_data = add_other_data(re_data, chan_data)
        RE_DATA_GLOBAL = re_data
        CHAN_DATA_GLOBAL = chan_data
    return re_data, chan_data

def generate_image(min_transaction_amount, calc_amountxsat):
    # RE_DATA_GLOBALを更新
    # グラフの作成
    G = nx.DiGraph()  # 有向グラフを作成

    transactions = []
    transaction = ()

    # データを追加（送信元、受信先、取引金額）
    for i in range(0, len(RE_DATA_GLOBAL.channel_peer)):
        for j in range(0, len(RE_DATA_GLOBAL.channel_peer[i].output_peer)):
            if RE_DATA_GLOBAL.channel_peer[i].output_peer[j].amt_sat < min_transaction_amount:
                continue
            if calc_amountxsat:
                amount = RE_DATA_GLOBAL.channel_peer[i].output_peer[j].amt_sat * RE_DATA_GLOBAL.channel_peer[i].output_peer[j].ave_fee /1000000
            else:
                amount = RE_DATA_GLOBAL.channel_peer[i].output_peer[j].amt_sat

            transaction = (RE_DATA_GLOBAL.channel_peer[i].peer_alias, RE_DATA_GLOBAL.channel_peer[i].output_peer[j].alias, amount)
            transactions.append(transaction)

    data_num_node_text = f"Number of data: {len(transactions)}"
    #print("data数=", len(transactions))

    # ノードとエッジを追加
    for sender, receiver, amount in transactions:
        G.add_edge(sender, receiver, weight=amount)

    # 可視化
    #pos = nx.spring_layout(G)  # ノードの配置を決定
    #plt.figure(figsize=(5, 5))
    #pos = nx.spring_layout(G, k=0.5, iterations=200)
    pos = nx.kamada_kawai_layout(G)
    plt.figure(figsize=(16, 16))  # 画像サイズを大きくする


    # ノードを描画
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=3000, edgecolors='black')
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, edge_color='gray', connectionstyle='arc3,rad=0.2')
    nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')

    # エッジを描画
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=10)


    plt.title("Transaction Network", fontsize=20)
    plt.axis('off')  # 軸を非表示にする

    # 画像をファイルに保存
    image_path = "graph.png"
    plt.savefig(image_path)
    plt.close()

    return image_path, data_num_node_text

# メイン関数
def main():
    global RE_DATA_GLOBAL, CHAN_DATA_GLOBAL
    #node informationのデータ
    node_name = f"<h3>Node name: " + "Kazumyon</h3>"
    node_pubkey = f"<h3>Node pubkey: " + "039cdd937f8d83fb2f78c8d7ddc92ae28c9dbb5c4827181cfc80df60dee1b7bf19</h3>"
    node_info = f"""<h3>Node information: </h3>
      <h3>--- Min channel size >= 2,000,000sat </h3>
      <h3>--- Peerswap is available (LBTC only) </h3><br>
      <h3>--- Twitter : @KazumyonL </h3>
      <h3>--- Telegram : @Kazumyon </h3>
    """
     # Gradioを利用してデータを表示
    with gr.Blocks(css=".scrollable { overflow-x: auto; white-space: nowrap; }") as iface:
        with gr.Tabs():
            with gr.TabItem("Node information"):
                gr.Markdown("# Node Information")  # 見出しを追加
                gr.Markdown(node_name)
                gr.Markdown(node_pubkey, elem_classes="scrollable")
                gr.Markdown(node_info)

            with gr.TabItem("Routing flow list"):
                gr.Markdown("# Routing Flow List")  # 見出しを追加
                file_dropdown = gr.Dropdown(choices=files, label="Select file")
                use_client_files = gr.Checkbox(label="Use client files")
                fwd_data_file = gr.File(label="Select fwd_data file", visible=False)
                chan_data_file = gr.File(label="Select chan_data file", visible=False)
                sort_by = gr.Dropdown(choices=["peer_alias", "capacity", "local_balance_ratio", "input_amt_sat", "output_amt_sat"], label="Sort by")
                reverse_sort = gr.Checkbox(label="Reverse Sort")
                data_period_markdown = gr.Markdown()  # データ期間を表示
                routing_peer_num_markdown = gr.Markdown()  # ルーティングピア数を表示
                table = gr.Markdown(elem_classes="scrollable")  # 初期状態で表を表示

                def toggle_file_inputs(use_client_files):
                    return gr.update(visible=use_client_files), gr.update(visible=use_client_files)

                use_client_files.change(fn=toggle_file_inputs, inputs=use_client_files, outputs=[fwd_data_file, chan_data_file])
                file_dropdown.change(fn=update_table, inputs=[file_dropdown, sort_by, reverse_sort, use_client_files, fwd_data_file, chan_data_file], outputs=[data_period_markdown, routing_peer_num_markdown, table])
                sort_by.change(fn=update_table, inputs=[file_dropdown, sort_by, reverse_sort, use_client_files, fwd_data_file, chan_data_file], outputs=[data_period_markdown, routing_peer_num_markdown, table])
                reverse_sort.change(fn=update_table, inputs=[file_dropdown, sort_by, reverse_sort, use_client_files, fwd_data_file, chan_data_file], outputs=[data_period_markdown, routing_peer_num_markdown, table])

                gr.Column([file_dropdown, use_client_files, fwd_data_file, chan_data_file, sort_by, reverse_sort])
                gr.Column([data_period_markdown])
                gr.Column([routing_peer_num_markdown])
                gr.Column([table])

            with gr.TabItem("Routing visualization"):
                gr.Markdown("# Routing visualization")
                min_transaction_amount = gr.Number(label="Min. transaction amount", value=100000, precision=0)
                calc_amountxsat = gr.Checkbox(label="Calculate amount x fee")
                update_button = gr.Button("Update DATA & visualize")
                data_num_node_output = gr.Markdown()
                image_output = gr.Image()
                update_button.click(fn=generate_image, inputs=[min_transaction_amount,calc_amountxsat], outputs=[image_output, data_num_node_output])

    iface.launch()
    
    sys.exit()
    #print("---------------------------------")
    #with open('re_data_str.txt', 'w', encoding='utf-8') as file:
    #    file.write(re_data_str)
    #print("-------------json data file write!! --------------------")
    #with open('re_data_json.txt', 'w', encoding='utf-8') as file:
    #    file.write(re_data_json_p)
    #print("---------------------------------")

    #sys.exit()
    # OpenAI APIを利用して、fowarding履歴の解析を行う
    client = OpenAI()

    response = client.beta.chat.completions.parse(
        #model="gpt-4o-2024-08-06",
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": f""" \
                "channel_peer"は"channel_detail"のListです。\
                "channel_detail"にはnode名"peer_alias"とnodeに対する入力node名一覧"input_peer"出力node名一覧"output_peer"があります。 \
                入力node一覧にはそれぞれのnodeの総流入量"amt_sat"が示してあります。 \
                出力node一覧にはそれぞれのnodeの総流出量"amt_sat"が示してあります。\
                入力node一覧にはそれぞれのnodeの総流入量時の平均手数料"fee_ave"が示してあります。\
                出力node一覧にはそれぞれのnodeの総流出量時の平均手数料"fee_ave"が示してあります。 \
                "channel_detail"にはすべての入力nodeからの総流入量"input_amt_sat"が示してあります。\
                "channel_detail"にはすべての出力nodeからの総流出量"output_amt_sat"が示してあります。\
                "channel_detail"にはnode名"peer_alias"の現在の総容量に対するLocal balanceの割合が"local_balance_ratio"として示してあります。\

                以上を前提として以下のjson形式の入力をもとに次の考察をしてください。\n\n{re_data_json} \
                また、このデータをきれいにhtml形式で表示するためのコードを生成してください。
                日本語に翻訳してください。

                1.各チャンネルの出力nodeの平均手数料が高く総流出量が多いチャンネルはありますか?
                    （このようなチャンネル群はより多くの手数料を稼ぎます。）
                2.各チャンネルの総流出量や総流入量が偏って多くLocal balanceの割合が偏ってしまうチャンネルはありますか?
                    (このようなチャンネルは片方に資金が貯まりやすく双方向に動きにくいnodeです。)
                3.各チャンネルのLocal balanceの割合が30%-70%に近く総流出量と総流入量が多いnodeはありますか?
                    （このようなチャンネルは双方向に流れることにより多くの資金を動かしており、より多くの手数料を稼げるため良いnodeといえます。）

                最後に各チャンネルの特性に基づいて、手数料の稼ぎやすさや資金の流れの偏りを分析してください。
                """,
            }
        ],
        temperature=0, 
        #response_format=ChannelList,
    )

    #print("##################################################")
    channel_reasoning = response.choices[0].message.content
    #channel_reasoning = response.choices[0].message.parsed
    print(channel_reasoning)  # デバッグ: channel_reasoning を表示
    #print("##################################################")

    with open('response.txt', 'w', encoding='utf-8') as file:
        file.write(channel_reasoning)


if __name__ == '__main__':
    main()
    #iface.launch()
