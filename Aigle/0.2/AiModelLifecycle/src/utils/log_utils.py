import logging
import sys

# --- 僅在需要時設定一次 Logger ---
# 取得一個名為 'my_app_logger' 的 logger
# 使用 __name__ 可以讓 logger 的名稱是模組的路徑，有助於追蹤訊息來源
logger = logging.getLogger(__name__)

# 檢查 logger 是否已經有 handlers，避免重複加入
if not logger.handlers:
    logger.setLevel(logging.DEBUG)  # 設定 logger 的最低處理層級

    # --- 建立一個 handler 來將訊息輸出到 console ---
    # 使用 StreamHandler 將訊息發送到 sys.stderr
    handler = logging.StreamHandler(sys.stderr)

    # --- 定義訊息的格式 ---
    # 使用 ANSI escape codes 來設定顏色
    class ColorFormatter(logging.Formatter):
        # 定義不同層級的顏色
        COLORS = {
            'WARNING': '\033[93m',  # 黃色
            'ERROR': '\033[91m',    # 紅色
            'RESET': '\033[0m'      # 重設顏色
        }

        def format(self, record):
            # 取得原始的格式化訊息
            message = super().format(record)
            # 根據層級套用顏色
            return self.COLORS.get(record.levelname, '') + message + self.COLORS['RESET']

    # 建立一個格式化器
    formatter = ColorFormatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # 將 handler 加入到 logger
    logger.addHandler(handler)

# --- 封裝成簡單易用的函式 ---
def log_warning(message):
    """
    印出黃色的警告訊息。

    Args:
        message (str): 要顯示的警告內容。
    """
    logger.warning(message)

def log_error(message, exc_info=False):
    """
    印出紅色的錯誤訊息。

    Args:
        message (str): 要顯示的錯誤內容。
        exc_info (bool): 是否要包含異常的堆疊追蹤資訊。
                         在 except 區塊中設為 True 非常有用。
    """
    logger.error(message, exc_info=exc_info)

# --- 使用範例 ---
if __name__ == '__main__':
    print("這是一般的 print 輸出。")

    log_warning("這是一個警告訊息，例如：某個函式即將被棄用。")
    log_error("這是一個錯誤訊息，例如：無法連接到資料庫。")

    # 在處理異常時的範例
    try:
        result = 1 / 0
    except ZeroDivisionError:
        log_error("計算發生錯誤，除數為零。", exc_info=True)