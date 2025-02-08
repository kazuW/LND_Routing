import networkx as nx
import matplotlib.pyplot as plt
import scipy as sp
import gradio as gr
import pandas as pd
import os
import json
import sys
from dotenv import load_dotenv

from datetime import datetime, timezone, timedelta
from ecdsa import SECP256k1, ellipticcurve, numbertheory
from ecdsa.ellipticcurve import Point
from ecdsa.util import number_to_string
import hashlib
import base64


from ai_engine import chat

load_dotenv()

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
RE_DATA_JSON = None
CHAN_DATA_JSON = None

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

def get_latest_listchannel_file(data_dir):
    listchannel_files = [f for f in os.listdir(data_dir) if 'listchannel' in f]
    if not listchannel_files:
        return None
    latest_file = max(listchannel_files, key=lambda f: os.path.getmtime(os.path.join(data_dir, f)))
    return os.path.join(data_dir, latest_file)

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
    global RE_DATA_GLOBAL, CHAN_DATA_GLOBAL, RE_DATA_JSON, CHAN_DATA_JSON_P
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
    
            # re_dataを文字列に変換
    RE_DATA_JSON_P = json.dumps(re_data.to_dict(), ensure_ascii=False, indent=2)
    RE_DATA_JSON = json.dumps(re_data.to_dict(), ensure_ascii=False, separators=(',', ':'))

    #print("---------------------------------")
    #with open('re_data_str.txt', 'w', encoding='utf-8') as file:
    #    file.write(re_data_str)
    #print("-------------json data file write!! --------------------")
    #with open('re_data_json.txt', 'w', encoding='utf-8') as file:
    #    file.write(RE_DATA_JSON_P)
    #print("-------------write file (re_data_json_p)------------")

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

def zbase32_decode(zbase32_encoded):
    """
    zbase32エンコードされた文字列をデコードする
    BOLT仕様に準拠したデコード
    """
    # 入力値の検証
    if not zbase32_encoded or not isinstance(zbase32_encoded, str):
        #print("無効な入力: 空または文字列ではありません")
        return None

    # zbase32のアルファベット (BOLT仕様)
    ZBASE32_ALPHABET = "ybndrfg8ejkmcpqxot1uwisza345h769"
    base32_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"
    
    # デバッグ情報の出力
    #print(f"入力されたzbase32: {zbase32_encoded}")
    
    # 小文字に変換
    zbase32_encoded = zbase32_encoded.lower()
    #print(f"小文字変換後: {zbase32_encoded}")
    
    # 変換テーブルの作成と変換
    translation_table = str.maketrans(ZBASE32_ALPHABET, base32_alphabet)
    base32_encoded = zbase32_encoded.translate(translation_table)
    #print(f"Base32変換後: {base32_encoded}")
    
    # パディングの追加
    padding_length = (8 - len(base32_encoded) % 8) % 8
    padded_base32 = base32_encoded + "=" * padding_length
    #print(f"パディング追加後: {padded_base32}")
    
    try:
        # デコード実行
        decoded = base64.b32decode(padded_base32)
        #print(f"デコード結果(16進数): {decoded.hex()}")
        return decoded
    except Exception as e:
        print(f"デコードエラー: {e}")
        return None

def recover_public_key_from_signature(signature, message_hash):
    """Recover public key from the signature and message hash."""
    curve = SECP256k1.curve
    order = SECP256k1.order
    G = SECP256k1.generator

    # Extract recovery code from the signature
    sig_recovery_code = signature[0]
    min_valid_code = 27
    max_valid_code = 34
    compact_sig_magic_offset = 27
    compact_sig_comp_pub_key = 4

    if sig_recovery_code < min_valid_code or sig_recovery_code > max_valid_code:
        raise ValueError(f"Invalid signature: public key recovery code {sig_recovery_code} is not in the valid range [{min_valid_code}, {max_valid_code}]")

    sig_recovery_code -= compact_sig_magic_offset
    was_compressed = (sig_recovery_code & compact_sig_comp_pub_key) != 0
    pub_key_recovery_code = sig_recovery_code & 3

    r = int.from_bytes(signature[1:33], byteorder='big')
    s = int.from_bytes(signature[33:], byteorder='big')

    # Calculate the x coordinate of the R point
    x = r + (pub_key_recovery_code // 2) * order

    # Calculate the y coordinate of the R point
    alpha = (pow(x, 3, curve.p()) + 7) % curve.p()
    beta = numbertheory.square_root_mod_prime(alpha, curve.p())
    y = beta if (beta % 2) == (pub_key_recovery_code % 2) else curve.p() - beta

    # Create the R point
    R = Point(curve, x, y)

    # Calculate the public key
    r_inv = pow(r, -1, order)
    e = int.from_bytes(message_hash, byteorder='big')
    sR = R * s
    eG = G * e
    Q = (sR + (-eG)) * r_inv

    # Return the public key in compressed format
    prefix = b'\x02' if Q.y() % 2 == 0 else b'\x03'
    return prefix + number_to_string(Q.x(), order), pub_key_recovery_code

def verify_message(message, signature, pubkey_bytes):
    """署名を検証する"""
    msg_bytes = message.encode('utf-8')
    formatted_msg = (
        b'Lightning Signed Message:' +
        msg_bytes
    )
    
    #print("formatted_msg :", formatted_msg.hex())

    # Double SHA256
    message_hash = hashlib.sha256(
        hashlib.sha256(formatted_msg).digest()
    ).digest()
    
    #print("message_hash :", message_hash.hex())

    # 公開鍵を復元
    recovered_pubkey, recovery_code = recover_public_key_from_signature(signature, message_hash)
    #print(f"復元された公開鍵: {recovered_pubkey.hex()}")
    #print(f"復元された公開鍵のリカバリフラグ: {recovery_code}")
    #print(f"提供された公開鍵: {pubkey_bytes.hex()}")

    if recovered_pubkey == pubkey_bytes:
        #print("公開鍵が一致します")
        return True
    else:
        #print("公開鍵が一致しません")
        return False

def node_auth(username, password):
    latest_channel_file = get_latest_listchannel_file(data_dir)

    with open(latest_channel_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        channels = data.get("channels", [])
        remote_pubkeys = [channel["remote_pubkey"] for channel in channels]
        remote_pubkeys.append("039cdd937f8d83fb2f78c8d7ddc92ae28c9dbb5c4827181cfc80df60dee1b7bf19")

    if username not in remote_pubkeys:
        return False

    date_str = datetime.now().strftime('%Y%m%d')
    message = f"{date_str}_{username}"
    #print("message: ", message)
    #print("password: ", password)

    # データをzbase32デコード
    signature_byte_array = zbase32_decode(password)

    # 署名を検証
    is_valid = verify_message(message, signature_byte_array, bytes.fromhex(username))

    if is_valid:
        return True
    else:
        return False


# メイン関数
def main():
    global RE_DATA_GLOBAL, CHAN_DATA_GLOBAL, RE_DATA_JSON, CHAN_DATA_JSON_P

    #node informationのデータ
    node_name = f"<h3>Node name: " + "Kazumyon</h3>"
    node_pubkey = f"<h3>Node pubkey: " + "039cdd937f8d83fb2f78c8d7ddc92ae28c9dbb5c4827181cfc80df60dee1b7bf19</h3>"
    node_info = f"""<h3>Node information: </h3>
      <h3>--- Min channel size >= 2,000,000sat </h3>
      <h3>--- Peerswap is available (LBTC only) </h3><br>
      <h3>--- Twitter : @KazumyonL </h3>
      <h3>--- Telegram : @Kazumyon </h3>
    """

    banner = f"""<div style="background-color: #f0f0f0; padding: 10px; text-align: center;">
                    <h2>Welcome to the LND Routing Dashboard</h2>
                    <br>
                 </div>"""

    # Google Analyticsのトラッキングコードを追加
    google_analytics = """
    <script async src="https://www.googletagmanager.com/gtag/js?id=G-51Z2XC2Z6H"></script>
    <script>
    window.dataLayer = window.dataLayer || [];
    function gtag(){dataLayer.push(arguments);}
    gtag('js', new Date());

    gtag('config', 'id=G-51Z2XC2Z6H');
    </script>
    """

    # CSSの定義部分を更新
    css = """
    .scrollable { overflow-x: auto; white-space: nowrap; }
    .#my_chatbot {
        height: 1800px !important;
        max-height: 1800px !important;
        overflow-y: auto !important;
    }
    """

    with open('./data/usage.txt', 'r', encoding='utf-8') as f:
        usage = f.read()


     # Gradioを利用してデータを表示
    with gr.Blocks(css=css, head=google_analytics) as iface:
        #gr.Markdown(google_analytics)
        gr.Markdown(banner)  # バナーを追加
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
                gr.Column([min_transaction_amount, calc_amountxsat, update_button])
                gr.Column([data_num_node_output])
                gr.Column([image_output])

            with gr.TabItem("AI analysis"):
                gr.Markdown("# AI analysis")
                translate_to_english = gr.Checkbox(label="Translate questions to English")
                
                """
                chat_interface = gr.ChatInterface(
                    fn=lambda message, history: chat(message, history, RE_DATA_JSON),
                    type="messages"
                )
                """

                # Chatbotコンポーネントを定義
                chatbot = gr.Chatbot(
                    label="AI Chat",
                    elem_id="my_chatbot"
                )

                # ユーザー入力用Textbox
                input_box = gr.Textbox(
                    label="Your message",
                    placeholder="Enter your text here..."
                )

                # 送信ボタン
                send_button = gr.Button("Send")

                # チャットのロジック
                def chatbot_fn(user_message, history):
                    # OpenAI形式のメッセージリストを作成
                    messages = []
                    for h in history:
                        if h[0]:  # ユーザーの発話
                            messages.append({"role": "user", "content": h[0]})
                        if h[1]:  # アシスタントの発話
                            messages.append({"role": "assistant", "content": h[1]})
    
                    # 新しい質問を追加
                    messages.append({"role": "user", "content": user_message})
                    
                    print("user_message: ", user_message)
                    print("messages: ", messages)
                    #print("RE_DATA_JSON: ", RE_DATA_JSON)
                    # chat関数を呼び出し、引数として渡す
                    reply = chat(user_message, messages, RE_DATA_JSON)
    
                    print("reply: ", reply)
                    # 履歴に追加して返す
                    new_history = history + [(user_message, reply)]
                    print("new_history: ", new_history)

                    return new_history


                # ボタンが押されたときの処理
                send_button.click(
                    fn=chatbot_fn,
                    inputs=[input_box, chatbot],
                    outputs=chatbot
                )

                gr.Markdown("## Sample questions")

                # Sample questions in Japanese and English
                questions_jp = [
                    "各チャンネルの出力nodeの平均手数料が高く総流出量が多いチャンネルはありますか?",
                    "各チャンネルの総流出量や総流入量が偏って多くLocal balanceの割合が偏ってしまうチャンネルはありますか?",
                    "各チャンネルのLocal balanceの割合が30%-70%に近く総流出量と総流入量が多いnodeはありますか?",
                    "手数料をたくさん稼いでいるチャンネル上位５つをあげてください?",
                    "各チャンネルの特性に基づいて、手数料の稼ぎやすさや資金の流れの偏りを分析してください。"
                ]

                questions_en = [
                    "Are there any channels where the average fee of the output node is high and the total outflow is large?",
                    "Are there any channels where the total outflow and total inflow are biased and the Local balance ratio is biased?",
                    "Are there any nodes where the Local balance ratio is close to 30%-70% and the total outflow and total inflow are large?",
                    "Please list the top 5 channels that earn the most fees.",
                    "Please analyze the ease of earning fees and the bias of fund flow based on the characteristics of each channel."
                ]

                def display_questions(translate):
                    if translate:
                        questions = questions_en
                    else:
                        questions = questions_jp
                    return "\n".join([f"<span style='font-size: 20px;'>- {question}</span><br>" for question in questions])

                questions_markdown = gr.Markdown(display_questions(False))
                translate_to_english.change(fn=display_questions, inputs=translate_to_english, outputs=questions_markdown)
                
                questions_markdown

            with gr.TabItem("Usage"):
                gr.Markdown("# Usage")  # 見出しを追加
                gr.Markdown(usage)
    
    iface.launch(auth=node_auth)
    #GRADIO_SHAREを設定必要かも

    #sys.exit()

if __name__ == '__main__':
    main()