_engine = None

def get_engine():
    global _engine
    if _engine is None:
        print("🔧 [VisionEngine] Initializing model...")
        from scripts.model_engine import QwenVisionEngine
        _engine = QwenVisionEngine()
        print("✅ [VisionEngine] Model loaded successfully.")
    return _engine
