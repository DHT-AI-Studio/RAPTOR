#!/usr/bin/env python3
"""
顯示 step 的變化
"""

import redis
import json
from datetime import datetime

class SimpleStepListener:
    def __init__(self):
        # 連接 Redis
        self.redis_client = redis.Redis(
            host='192.168.157.165',
            port=6381,
            decode_responses=True
        )
        
        # 建立 Pub/Sub
        self.pubsub = self.redis_client.pubsub()
        
        # 記錄上一次的 step，避免重複顯示
        self.last_steps = {}
        
    def start(self):
        """開始監聽"""
        
        # 訂閱所有 orchestrator keys
        self.pubsub.psubscribe('__keyspace@0__:*orchestrator:*')
        
        print("=" * 70)
        print(" Step listener")
        print("=" * 70)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        print()
        
        # 持續監聽
        for message in self.pubsub.listen():
            if message['type'] == 'pmessage':
                self.handle_message(message)
    
    def handle_message(self, message):
        """處理 Redis 事件"""
        
        # 取得 key
        channel = message['channel']
        key = channel.replace('__keyspace@0__:', '')
        
        try:
            # 讀取 state
            json_str = self.redis_client.get(key)
            
            # 如果 key 不存在或為空
            if not json_str:
                return
            
            # 解析 JSON
            state = json.loads(json_str)
            
            # 只處理有 step 的更新
            if state and 'step' in state:
                current_step = state['step']
                
                # 檢查 step 
                if self.last_steps.get(key) != current_step:
                    self.last_steps[key] = current_step
                    self.display_step_change(key, current_step)
                    
        except json.JSONDecodeError:
            # JSON 解析失敗
            pass
        except Exception as e:
            pass
    
    def display_step_change(self, key, step):
        """顯示 step 變化"""
        
        # 解析 key
        parts = key.split(':')
        service_type = parts[0]  # all orchestrator
        correlation_id = parts[1] if len(parts) > 1 else 'unknown'
        
        # simplied service name
        service = service_type.replace('_orchestrator', '').upper()
        
        # simplied correlation_id (顯示前 8 碼)
        short_id = correlation_id[:8]
        
        # get time
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        print(f"[{timestamp}] {service:5} | {short_id} | {step}")


def main():
    try:
        listener = SimpleStepListener()
        listener.start()
    except KeyboardInterrupt:
        print("\n")
        print("=" * 70)
        print("listener stop")
        print("=" * 70)
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == '__main__':
    main()
