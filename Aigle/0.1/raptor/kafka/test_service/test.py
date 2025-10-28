#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
檔案上傳和處理請求分發測試工具
整合 HTTP API 和 Kafka 消息處理
"""

import asyncio
import aiohttp
import json
import uuid
import os
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from aiokafka import AIOKafkaProducer

# 配置
BASE_URL = "http://192.168.157.165:8086"
KAFKA_BOOTSTRAP_SERVERS = "192.168.157.165:19002,192.168.157.165:19003,192.168.157.165:19004"

# Kafka Topics
KAFKA_TOPICS = {
    "document": "document-processing-requests",
    "image": "image-processing-requests", 
    "video": "video-processing-requests",
    "audio": "audio-processing-requests"
}

def get_file_type(filename: str) -> str:
    """根據檔案名稱判斷檔案類型"""
    if not filename:
        return "unknown"
    
    ext = filename.lower().split('.')[-1] if '.' in filename else ""
    
    # 文件類型
    document_exts = ['pdf', 'doc', 'docx', 'txt', 'html', 'htm', 'csv', 'xlsx', 'xls', 'ppt', 'pptx']
    # 圖片類型  
    image_exts = ['jpg', 'jpeg', 'png']
    # 影片類型
    video_exts = ['mp4', 'avi', 'mov', 'wmv', 'flv', 'webm', 'mkv', '3gp']
    # 音檔類型
    audio_exts = ['mp3', 'wav', 'flac', 'aac', 'ogg', 'wma', 'm4a']
    
    if ext in document_exts:
        return "document"
    elif ext in image_exts:
        return "image"
    elif ext in video_exts:
        return "video"
    elif ext in audio_exts:
        return "audio"
    else:
        return "unknown"

def create_processing_request_message(
    original_message: Dict[str, Any],
    asset_path: str,
    version_id: str,
    file_type: str,
    access_token: str,
    filename: str,
    status: str
) -> Dict[str, Any]:
    """創建處理請求消息"""
    
    # 根據檔案類型決定目標服務
    service_map = {
        "document": "document_orchestrator_service",
        "image": "image_orchestrator_service", 
        "video": "video_orchestrator_service",
        "audio": "audio_orchestrator_service"
    }
    
    # 根據檔案類型決定動作
    action_map = {
        "document": "document_processing",
        "image": "image_processing",
        "video": "video_processing", 
        "audio": "audio_processing"
    }
    
    target_service = service_map.get(file_type, "unknown_service")
    action = action_map.get(file_type, "unknown_processing")
    
    message = {
        "message_id": str(uuid.uuid4()),
        "correlation_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_service": "file_upload_service",
        "target_service": target_service,
        "message_type": "REQUEST",
        "priority": "MEDIUM",
        "payload": {
            "request_id": str(uuid.uuid4()),
            "user_id": original_message.get("user_id", "unknown"),
            "access_token": access_token,
            "action": action,
            "parameters": {
                "asset_path": asset_path,
                "version_id": version_id,
                "primary_filename": filename,
                "file_type": file_type,
                "status": status
            },
            "file_path": f"{asset_path}/{version_id}/{filename}",
            "metadata": {
                "upload_timestamp": datetime.now(timezone.utc).isoformat(),
                "original_metadata": original_message.get("metadata", {})
            }
        },
        "retry_count": 0,
        "ttl": 3600
    }
    
    return message

def encode_message_value(message: Dict[str, Any]) -> bytes:
    """編碼消息為 bytes"""
    return json.dumps(message, ensure_ascii=False).encode('utf-8')

class FileUploadProcessingTester:
    def __init__(self):
        self.base_url = BASE_URL
        self.kafka_bootstrap_servers = KAFKA_BOOTSTRAP_SERVERS
        self.producer = None
        
    async def start_kafka_producer(self):
        """啟動 Kafka Producer"""
        self.producer = AIOKafkaProducer(
            bootstrap_servers=self.kafka_bootstrap_servers,
            value_serializer=lambda x: json.dumps(x, ensure_ascii=False).encode('utf-8')
        )
        await self.producer.start()
        print("✅ Kafka Producer 已啟動")
        
    async def stop_kafka_producer(self):
        """停止 Kafka Producer"""
        if self.producer:
            await self.producer.stop()
            print("✅ Kafka Producer 已停止")
        
    async def get_access_token(self, username: str, password: str) -> Optional[str]:
        """獲取訪問令牌"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/token",
                    data={
                        "username": username,
                        "password": password
                    }
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        access_token = result.get("access_token")
                        print(f"✅ 成功獲取 access_token")
                        return access_token
                    else:
                        print(f"❌ 登入失敗: {resp.status}")
                        error_text = await resp.text()
                        print(f"   錯誤詳情: {error_text}")
                        return None
        except Exception as e:
            print(f"❌ 獲取 token 時發生錯誤: {e}")
            return None
    
    async def upload_file(self, file_path: str, access_token: str) -> Optional[Dict[str, Any]]:
        """上傳檔案"""
        try:
            if not os.path.exists(file_path):
                print(f"❌ 檔案不存在: {file_path}")
                return None
                
            filename = os.path.basename(file_path)
            
            async with aiohttp.ClientSession() as session:
                form = aiohttp.FormData()
                form.add_field("primary_file", open(file_path, "rb"), filename=filename)
                form.add_field("archive_ttl", "30")
                form.add_field("destroy_ttl", "30")
                
                async with session.post(
                    f"{self.base_url}/fileupload",
                    headers={"Authorization": f"Bearer {access_token}"},
                    data=form
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        print(f"✅ 檔案上傳成功:")
                        print(f"   Asset Path: {result.get('asset_path')}")
                        print(f"   Version ID: {result.get('version_id')}")
                        print(f"   Primary Filename: {result.get('primary_filename')}")
                        print(f"   Upload Date: {result.get('upload_date')}")
                        print(f"   Status: {result.get('status')}")
                        return result
                    else:
                        print(f"❌ 檔案上傳失敗: {resp.status}")
                        error_text = await resp.text()
                        print(f"   錯誤詳情: {error_text}")
                        return None
        except Exception as e:
            print(f"❌ 上傳檔案時發生錯誤: {e}")
            return None
    
    async def send_processing_request(self, upload_result: Dict[str, Any], access_token: str, user_id: str = "test_user", processing_mode: str = None):
        """發送處理請求到對應的 Kafka topic"""
        try:
            asset_path = upload_result.get("asset_path")
            version_id = upload_result.get("version_id") 
            filename = upload_result.get("primary_filename")
            status = upload_result.get("status")
            
            if not all([asset_path, version_id, filename, status]):
                print("❌ 上傳結果缺少必要資訊")
                return False
            
            # 判斷檔案類型
            file_type = get_file_type(filename)
            if file_type == "unknown":
                print(f"❌ 不支援的檔案類型: {filename}")
                return False
            
            print(f"📄 檔案類型: {file_type}")
            
            # 創建原始消息（模擬）
            original_message = {
                "user_id": user_id,
                "metadata": {
                    "upload_source": "test_script",
                    "client_ip": "127.0.0.1"
                }
            }
            
            # 如果有處理模式，加入 metadata
            if processing_mode:
                original_message["metadata"]["processing_mode"] = processing_mode
                print(f"⚙️  處理模式: {processing_mode}")
            
            # 創建處理請求消息
            processing_message = create_processing_request_message(
                original_message=original_message,
                asset_path=asset_path,
                version_id=version_id,
                file_type=file_type,
                access_token=access_token,
                filename=filename,
                status=status
            )
            
            # 選擇對應的 topic
            target_topic = KAFKA_TOPICS.get(file_type)
            if not target_topic:
                print(f"❌ 找不到對應的 topic: {file_type}")
                return False
            
            print(f"📨 發送到 Kafka topic: {target_topic}")
            
            # 發送到 Kafka
            await self.producer.send(target_topic, processing_message)
            print(f"✅ 處理請求已發送到 {target_topic}")
            print(f"   Message ID: {processing_message['message_id']}")
            print(f"   Correlation ID: {processing_message['correlation_id']}")
            print(f"   Target Service: {processing_message['target_service']}")
            print(f"   Action: {processing_message['payload']['action']}")
            
            # 顯示消息內容（可選）
            print(f"\n📋 消息內容:")
            print(json.dumps(processing_message, indent=2, ensure_ascii=False))
            
            return True
                
        except Exception as e:
            print(f"❌ 發送處理請求時發生錯誤: {e}")
            return False
    
    def get_processing_mode(self, file_path: str) -> Optional[str]:
        """根據檔案類型獲取處理模式"""
        filename = os.path.basename(file_path)
        file_extension = filename.lower().split('.')[-1] if '.' in filename else ""
        
        if file_extension == "pdf":
            print(f"\n📄 檢測到 PDF 檔案，請選擇處理模式:")
            print("  1. default - 預設模式 (文字提取)")
            print("  2. ocr - OCR 模式 (圖像識別)")
            
            while True:
                choice = input("請選擇處理模式 (1/2, 預設為 1): ").strip()
                if choice == "" or choice == "1":
                    return "default"
                elif choice == "2":
                    return "ocr"
                else:
                    print("❌ 請輸入 1 或 2")
        
        return None
    
    async def run_test(self, username: str, password: str, file_path: str):
        """執行完整測試流程"""
        print("🚀 開始檔案上傳和處理測試")
        print("=" * 50)
        
        try:
            # 0. 啟動 Kafka Producer
            print("0️⃣ 啟動 Kafka Producer...")
            await self.start_kafka_producer()
            
            # 1. 獲取 access token
            print("\n1️⃣ 獲取 access token...")
            access_token = await self.get_access_token(username, password)
            if not access_token:
                return
            
            # 2. 檢查處理模式（如果是 PDF）
            processing_mode = self.get_processing_mode(file_path)
            
            # 3. 上傳檔案
            print("\n2️⃣ 上傳檔案...")
            upload_result = await self.upload_file(file_path, access_token)
            if not upload_result:
                return
            
            # 4. 發送處理請求
            print("\n3️⃣ 發送處理請求...")
            success = await self.send_processing_request(
                upload_result, 
                access_token, 
                username, 
                processing_mode
            )
            
            if success:
                print("\n✅ 測試完成! 處理請求已發送到相應的服務")
                print("📝 注意: 這只是發送處理請求，實際處理結果需要監控對應的結果 topic")
            else:
                print("\n❌ 處理請求發送失敗")
                
        except Exception as e:
            print(f"❌ 測試過程發生錯誤: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await self.stop_kafka_producer()

def get_user_inputs():
    """獲取用戶輸入"""
    print("請輸入測試參數:")
    print("-" * 30)
    
    # 獲取認證資訊
    username = input("用戶名 (預設: user1): ").strip()
    if not username:
        username = "user1"
    
    password = input("密碼 (預設: dht888888): ").strip()
    if not password:
        password = "dht888888"
    
    # 獲取檔案路徑
    file_path = input("檔案路徑: ").strip()
    if not file_path:
        print("❌ 檔案路徑不能為空")
        return None, None, None
    
    if not os.path.exists(file_path):
        print(f"❌ 檔案不存在: {file_path}")
        return None, None, None
    
    return username, password, file_path

async def main():
    """主函數"""
    print("📁🔄 檔案上傳和處理請求分發測試工具")
    print("=" * 50)
    print("功能說明:")
    print("1. 使用 HTTP API 進行用戶認證")
    print("2. 使用 HTTP API 上傳檔案")
    print("3. 根據檔案類型發送 Kafka 處理請求")
    print("4. 支援 PDF 檔案的處理模式選擇")
    print("")
    
    # 獲取用戶輸入
    username, password, file_path = get_user_inputs()
    if not all([username, password, file_path]):
        return
    
    # 顯示測試資訊
    print(f"\n📋 測試資訊:")
    print(f"   用戶名: {username}")
    print(f"   檔案路徑: {file_path}")
    print(f"   檔案名: {os.path.basename(file_path)}")
    print(f"   檔案大小: {os.path.getsize(file_path)} bytes")
    print(f"   預期檔案類型: {get_file_type(os.path.basename(file_path))}")
    
    # 確認執行
    confirm = input("\n是否繼續執行測試? (y/N): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("❌ 測試已取消")
        return
    
    # 執行測試
    tester = FileUploadProcessingTester()
    await tester.run_test(username, password, file_path)

if __name__ == "__main__":
    # 執行測試
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n❌ 測試被用戶中斷")
    except Exception as e:
        print(f"❌ 程式執行錯誤: {e}")
        import traceback
        traceback.print_exc()
