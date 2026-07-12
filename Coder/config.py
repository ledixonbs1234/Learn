# config.py
import base64
import os
import fnmatch
from pathlib import Path
import re
from typing import Dict, List, Set
import uuid

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
fast_model = ChatOpenAI(
    base_url="http://localhost:20128/v1",
    api_key="khongco",  # type: ignore
    model="kiro",
    temperature=0.1,
    reasoning_effort=None
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
    workspace_path = Path(workspace).expanduser().resolve()
    
    # 🌟 GIẢI PHÁP NGOẠI LỆ AN TOÀN: Cho phép truy xuất thư mục OpenWiki toàn cục
    global_openwiki_path = Path("~/.openwiki").expanduser().resolve()
    
    target_path = Path(cleaned)
    if cleaned.startswith("~"):
        target_path = target_path.expanduser()
        
    try:
        resolved_target = target_path.resolve()
        # Nếu đường dẫn trỏ thẳng vào ~/.openwiki hoặc các thư mục con của nó
        if resolved_target == global_openwiki_path or global_openwiki_path in resolved_target.parents:
            if create_parent:
                resolved_target.parent.mkdir(parents=True, exist_ok=True)
            return resolved_target
    except Exception:
        pass
    
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



# =====================================================================
# CẤU HÌNH MỐC ĐÁNH DẤU HỆ THỐNG CẤP ĐỘ PRODUCTION
# =====================================================================
VCS_MARKERS: Set[str] = {".git", ".hg", ".svn"}
WORKSPACE_MARKERS: Set[str] = {"pnpm-workspace.yaml", "lerna.json", "nx.json", "turbo.json", "bun.lockb"}
LANGUAGE_MARKERS: Set[str] = {
    "pyproject.toml", "setup.py", "poetry.lock", "requirements.txt",
    "package.json", "tsconfig.json", "yarn.lock", "package-lock.json",
    "Cargo.toml", "Cargo.lock", "go.mod", "go.sum", "pubspec.yaml",
    "pom.xml", "build.gradle", "settings.gradle", "CMakeLists.txt", "Makefile", "project.godot"
}
FALLBACK_MARKERS: Set[str] = {"CONTEXT.md", "README.md", "main.py", "index.js", ".env", "docker-compose.yml"}
SANDBOX_BLACK_LIST: Set[str] = {"node_modules", ".venv", "venv", "env", "build", "dist", ".gradle", ".git", ".idea", ".vscode"}
SYSTEM_DIR_NAMES: Set[str] = {"usr", "lib", "lib64", "bin", "sbin", "etc", "var", "opt", "sys", "proc", "dev", "System", "Windows", "Program Files", "Program Files (x86)", "Users"}

def find_project_root_heuristic(start_path: Path) -> Path:
    """Thuật toán định vị gốc dự án (Project Root) cấp độ Production."""
    try:
        current_dir = start_path.expanduser().resolve()
        if current_dir.is_file():
            current_dir = current_dir.parent

        parts = current_dir.parts
        escape_idx = -1
        for idx, part in enumerate(parts):
            if part in SANDBOX_BLACK_LIST:
                escape_idx = idx
                break
                
        if escape_idx != -1:
            escaped_path = Path(*parts[:escape_idx])
            if escaped_path.anchor != str(escaped_path):
                current_dir = escaped_path

        candidates: Dict[str, List[Path]] = {
            "project_roots": [],
            "workspace_roots": [],
            "fallback_roots": []
        }

        user_home = Path.home().resolve()
        max_depth = 25
        depth = 0

        while depth < max_depth:
            if len(current_dir.parts) > 1:
                first_level = current_dir.parts[1] if current_dir.parts[0] in ("/", "C:", "D:", "E:") else current_dir.parts[0]
                if first_level in SYSTEM_DIR_NAMES:
                    break

            try:
                existing_items = {item.name for item in current_dir.iterdir()}
            except (PermissionError, OSError):
                break

            has_project_marker = any(marker in existing_items for marker in LANGUAGE_MARKERS)
            has_workspace_marker = (
                any(marker in existing_items for marker in WORKSPACE_MARKERS) or
                any(marker in existing_items for marker in VCS_MARKERS)
            )
            has_fallback_marker = any(marker in existing_items for marker in FALLBACK_MARKERS)

            if has_project_marker:
                candidates["project_roots"].append(current_dir)
            if has_workspace_marker:
                candidates["workspace_roots"].append(current_dir)
            if has_fallback_marker:
                candidates["fallback_roots"].append(current_dir)

            if current_dir.parent == current_dir or current_dir == user_home:
                break

            current_dir = current_dir.parent
            depth += 1

        if candidates["project_roots"]:
            return candidates["project_roots"][0]
        if candidates["workspace_roots"]:
            return candidates["workspace_roots"][0]
        if candidates["fallback_roots"]:
            return candidates["fallback_roots"][0]

        fallback_path = start_path.expanduser().resolve()
        return fallback_path.parent if fallback_path.is_file() else fallback_path

    except Exception as e:
        print(f"[Cảnh báo hệ thống] Lỗi trong quá trình tìm project root: {str(e)}")
        try:
            return start_path.parent if start_path.is_file() else start_path
        except Exception:
            return Path(".").resolve()
        
        
def sanitize_tool_result_content(tool_name: str, result: any, workspace_path: str = ".") -> str:
    """
    Phát hiện và xử lý các kết quả công cụ có chứa chuỗi Base64 ảnh hoặc dữ liệu quá dài.
    Ghi nhận ảnh xuống ổ đĩa và thay thế bằng đường dẫn để tránh làm nghẽn bộ nhớ Token (CONTENT_LENGTH_EXCEEDS_THRESHOLD).
    """
    result_str = str(result)
    
    # 1. Phát hiện chuỗi Base64 của ảnh (hoặc từ các tool screenshot)
    is_screenshot_tool = "screenshot" in tool_name.lower() or "take_screenshot" in tool_name.lower()
    has_image_header = "data:image" in result_str or "iVBORw0KG" in result_str
    
    if (is_screenshot_tool or has_image_header) and len(result_str) > 5000:
        try:
            base64_data = result_str
            header_match = re.search(r"data:image/[^;]+;base64,", result_str)
            if header_match:
                base64_data = result_str[header_match.end():]
            
            base64_data = re.sub(r"\s+", "", base64_data)
            
            sc_dir = Path(workspace_path).expanduser().resolve() / "screenshots"
            sc_dir.mkdir(parents=True, exist_ok=True)
            
            file_name = f"screenshot_{tool_name}_{uuid.uuid4().hex[:8]}.png"
            file_path = sc_dir / file_name
            
            img_data = base64.b64decode(base64_data)
            file_path.write_bytes(img_data)
            
            try:
                rel_path = file_path.relative_to(Path(workspace_path).expanduser().resolve())
            except ValueError:
                rel_path = file_path
                
            return (
                f"[Hệ thống tối ưu hóa]: Đã lưu ảnh chụp màn hình vật lý thành công tại `{rel_path}`.\n"
                f"Nội dung văn bản thô Base64 đã được lược bỏ để ngăn lỗi quá tải dung lượng token "
                f"(CONTENT_LENGTH_EXCEEDS_THRESHOLD). Bạn có thể xem ảnh trực tiếp từ đường dẫn trên."
            )
        except Exception as e:
            return f"[Cảnh báo]: Công cụ chụp ảnh trả về dữ liệu quá dài nhưng gặp lỗi khi lưu file ({str(e)}). Dữ liệu đã bị cắt ngắn để bảo vệ ngữ cảnh: {result_str[:500]}... [Đã cắt bớt]"
            
    # 2. Cắt ngắn nếu nội dung text thô của bất cứ tool nào vượt quá giới hạn an toàn (ví dụ: > 100KB)
    if len(result_str) > 100000:
        return (
            f"{result_str[:100000]}\n\n"
            f"... [Hệ thống tự động cắt bớt {len(result_str) - 100000} ký tự dư thừa "
            f"để tránh lỗi CONTENT_LENGTH_EXCEEDS_THRESHOLD của API Kiro] ..."
        )
        
    return result_str