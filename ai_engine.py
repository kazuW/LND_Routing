from openai import OpenAI
from pydantic import BaseModel, Field

import os
import json

class Channel_description(BaseModel):
    alias : str = Field(..., title="alias", description="node name")
    ave_fee : float = Field(..., title="ave_fee", description="average fee")
    ave_min : float = Field(..., title="ave_min", description="minimum fee")
    ave_max : float = Field(..., title="ave_max", description="maximum fee")
    amt_sat : float = Field(..., title="amt_sat", description="amount of funds(sat)")
    count : int = Field(..., title="count", description="Number of routing occurrences")

class ChannelDetail_description(BaseModel):
    channel_peer : str = Field(..., title="channel_peer", description="channel peer(node name)")
    input_peer : list[Channel_description] = Field(..., title="input_peer", description="input peer(node name)")
    output_peer : list[Channel_description] = Field(..., title="output_peer", description="output peer(node name)")
    input_amt_sat : float = Field(..., title="input_amt_sat", description="The amount of funds(sat) flowing from the input node to the own node")
    output_amt_sat : float = Field(..., title="output_amt_sat", description="The amount of funds(sat) flowing from the own node to the output node")
    capacity : int = Field(..., title="capacity", description="channel capacity")
    local_balance_ratio : float = Field(..., title="local_balance_ratio", description="the own node funds / total funds")

class RestructuredData(BaseModel):
    ChannelDetail : list[ChannelDetail_description] = Field(..., title="Channel Detail", description="channel detail")
    analysis_result : str = Field(..., title="analysis_result", description="analysis result")
    start_time : str = Field(..., title="start_time", description="start time")
    end_time : str = Field(..., title="end_time", description="end time")
    channel_num : int = Field(..., title="channel_num", description="routing channel number")


def chat(message, history, re_data_json) -> str:
    # OpenAI APIを利用して、fowarding履歴の解析を行う
    client = OpenAI()

    #print("############# re_data_json #############")
    #print(re_data_json)

    message_init = [
        {
            "role": "assistant",
            "content": "このAIは、「Routing flow list」で入力したLNDのfowarding履歴を解析し、チャンネル状況を分析します。"
        },
        {
            "role": "user",
            "content": f""" \
            Routingデータのjson形式
            '{Channel_description},{ChannelDetail_description},{RestructuredData}' 
            
            Routingデータ
            '{re_data_json}'

            各チャンネルの出力nodeの平均手数料が高く総流出量が多いチャンネルはより多くの手数料を稼ぎます。
            各チャンネルの総流出量や総流入量が偏って多くLocal balanceの割合が偏ってしまうチャンネルは片方に資金が貯まりやすく双方向に動きにくいnodeです。
            各チャンネルのLocal balanceの割合が30%-70%に近く総流出量と総流入量が多いチャンネルは双方向に流れることにより多くの資金を動かしており、より多くの手数料を稼げるため良いnodeといえます。
            
            これらのデータを元に今後の質問の回答を行ってください。
            まず読み込んだデータの期間とルーティング対象のnode数を教えてください。
            """
        }
    ]

    message = message_init + history + [{"role": "user", "content": message}]

    #print("############# message #############")
    #print(message)

    try:
        response = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=message,
            temperature=0, 
            #response_format=ChannelList,
        )

        #print("response= ", response)

        if response and response.choices:
            total_tokens = response.usage.prompt_tokens
            print("total_tokens= ", total_tokens)
            return response.choices[0].message.content
        else:
            return "No response from the AI.", 0
    except Exception as e:
        print(f"Error: {e}")
        return f"Error: {e}"