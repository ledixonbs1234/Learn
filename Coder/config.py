# config.py
import os
import fnmatch
from pathlib import Path

# ==========================================
# CẤU HÌNH TRACING LANGSMITH
# ==========================================
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "LangGraph-Production-Coder-V3"

from langchain_openai import ChatOpenAI

# Khởi tạo Model
model = ChatOpenAI(
    base_url="http://localhost:20128/v1",
    api_key="khongco",  # type: ignore
    model="kiro",
    temperature=0.1
)

class GitIgnoreMatcher:
    """Phân tích cú pháp .gitignore để xác định xem một đường dẫn có bị bỏ qua hay không."""
    def __init__(self, workspace_path: Path):
        self.workspace = workspace_path
        self.patterns = []
        
        gitignore_file = workspace_path / ".gitignore"
        if gitignore_file.exists():
            try:
                for line in gitignore_file.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#"):
                        self.patterns.append(line)
            except Exception:
                pass
                
        self.patterns.extend([
            ".git", "__pycache__", "*.pyc", ".DS_Store", "node_modules", 
            ".dart_tool", "build", "ios/Pods", "android/.gradle"
        ])

    def is_ignored(self, path: Path) -> bool:
        try:
            rel_path = path.relative_to(self.workspace)
        except ValueError:
            return False
            
        rel_str = rel_path.as_posix()
        
        for pattern in self.patterns:
            pat = pattern.strip()
            if pat.endswith('/'):
                pat = pat[:-1]
            
            if (fnmatch.fnmatch(rel_str, pat) or 
                fnmatch.fnmatch(rel_str, pat + "/*") or 
                fnmatch.fnmatch(rel_str, "*/" + pat) or 
                fnmatch.fnmatch(rel_str, "*/" + pat + "/*")):
                return True
                
            if fnmatch.fnmatch(rel_path.name, pat):
                return True
                
            for part in rel_path.parts:
                if fnmatch.fnmatch(part, pat):
                    return True
                    
        return False


def sanitize_and_resolve_path(workspace: str, raw_target_path: str, create_parent: bool = False) -> Path:
    cleaned = raw_target_path.replace('"', '').replace("'", "").replace("\\", "/").strip()
    # GIẢI PHÁP: Sử dụng expanduser() để dịch dấu ~ thành thư mục Home [2]
    workspace_path = Path(workspace).expanduser().resolve()
    
    target_path = Path(cleaned)
    if cleaned.startswith("~"):
        target_path = target_path.expanduser()
    
    if target_path.is_absolute():
        try:
            target_path = target_path.relative_to(workspace_path)
        except ValueError:
            parts = target_path.parts
            if parts[0].endswith(":") or parts[0] == "/":
                target_path = Path(*parts[1:])
                
    final_path = (workspace_path / target_path).resolve()
    
    try:
        final_path.relative_to(workspace_path)
    except ValueError:
        raise ValueError("Cảnh báo bảo mật: Đường dẫn nằm ngoài vùng an toàn.")
        
    if create_parent:
        final_path.parent.mkdir(parents=True, exist_ok=True)
    return final_path