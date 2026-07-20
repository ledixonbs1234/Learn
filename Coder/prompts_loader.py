# oder/prompts_loader.py
import os
from pathlib import Path
from functools import lru_cache

# Định vị thư mục chứa prompts tĩnh
PROMPTS_DIR = Path(__file__).parent.resolve() / "prompts"

@lru_cache(maxsize=32)
def load_prompt(filename: str) -> str:
    """
    Nạp nội dung file prompt tĩnh từ thư mục prompts/.
    Sử dụng lru_cache để tránh truy cập đĩa cứng (I/O) liên tục trong đồ thị tuần hoàn.
    """
    file_path = PROMPTS_DIR / filename
    
    # Cơ chế dự phòng (Fallback) nếu chạy từ các môi trường đóng gói khác nhau
    if not file_path.exists():
        alternative_path = Path(__file__).parent.parent.resolve() / "prompts" / filename
        if alternative_path.exists():
            file_path = alternative_path
        else:
            raise FileNotFoundError(f"Không tìm thấy file prompt tĩnh tại: {file_path}")
            
    return file_path.read_text(encoding="utf-8")