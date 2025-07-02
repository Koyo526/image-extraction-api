#!/usr/bin/env python3
"""
Bedrock Nova Lite接続テスト用スクリプト
"""
import sys
import os
sys.path.append('app')

from utils.bedrock_client import invoke_nova_lite_text

def test_bedrock_connection():
    """Bedrock Nova Liteの接続テスト"""
    try:
        print("🔄 Bedrock Nova Liteの接続テストを開始...")
        
        # シンプルなテキスト生成テスト
        prompt = "こんにちは！簡単な自己紹介をしてください。"
        result = invoke_nova_lite_text(prompt, max_tokens=100)
        
        print("✅ 接続成功!")
        print(f"📝 レスポンス: {result}")
        return True
        
    except Exception as e:
        print(f"❌ 接続エラー: {e}")
        return False

if __name__ == "__main__":
    success = test_bedrock_connection()
    if success:
        print("\n🎉 Bedrock Nova Liteの設定が完了しました！")
    else:
        print("\n⚠️  設定を確認してください。")
        print("- AWS認証情報が正しく設定されているか")
        print("- 東京リージョン(ap-northeast-1)でNova Liteが利用可能か")
        print("- IAMポリシーでBedrockの権限があるか")
