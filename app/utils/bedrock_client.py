import os
import boto3
import json
import base64
from typing import List, Dict, Any

# Bedrock クライアントの初期化
bedrock_client = boto3.client(
    'bedrock-runtime',
    region_name=os.getenv('AWS_REGION', 'ap-northeast-1'),
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
)

# Nova Lite モデル ID
NOVA_LITE_MODEL_ID = "amazon.nova-lite-v1:0"

def invoke_nova_lite_text(prompt: str, max_tokens: int = 1000) -> str:
    """
    Nova Liteでテキスト生成を行う
    """
    try:
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "inferenceConfig": {
                "max_new_tokens": max_tokens,
                "temperature": 0.7
            }
        }
        
        response = bedrock_client.invoke_model(
            modelId=NOVA_LITE_MODEL_ID,
            body=json.dumps(body),
            contentType='application/json'
        )
        
        response_body = json.loads(response['body'].read())
        return response_body['output']['message']['content'][0]['text']
        
    except Exception as e:
        print(f"Error invoking Nova Lite: {e}")
        raise e

def invoke_nova_lite_vision(prompt: str, image_base64: str, max_tokens: int = 1000) -> str:
    """
    Nova Liteで画像解析を行う
    """
    try:
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "text": prompt
                        },
                        {
                            "image": {
                                "format": "jpeg",
                                "source": {
                                    "bytes": image_base64
                                }
                            }
                        }
                    ]
                }
            ],
            "inferenceConfig": {
                "max_new_tokens": max_tokens,
                "temperature": 0.7
            }
        }
        
        response = bedrock_client.invoke_model(
            modelId=NOVA_LITE_MODEL_ID,
            body=json.dumps(body),
            contentType='application/json'
        )
        
        response_body = json.loads(response['body'].read())
        return response_body['output']['message']['content'][0]['text']
        
    except Exception as e:
        print(f"Error invoking Nova Lite Vision: {e}")
        raise e
