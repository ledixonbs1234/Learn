# oder/nodes/utils.py
import base64
import os
import re
import platform
import shutil
import subprocess
import mimetypes
from pathlib import Path
from typing import Any, List, Tuple, Optional
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage

def get_text_content_safely(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    elif isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif "text" in item and "type" not in item:
                    text_parts.append(item["text"])
            elif isinstance(item, str):
                text_parts.append(item)
        return " ".join(text_parts).strip()
    return ""

def sanitize_llm_response_content(response: AIMessage) -> AIMessage:
    if not response or not isinstance(response, AIMessage):
        return response
    content_str = response.content
    if isinstance(content_str, str) and content_str.strip():
        cleaned_content = re.sub(r"<thinking>.*?</thinking>", "", content_str, flags=re.DOTALL)
        cleaned_content = re.sub(r"<thought>.*?</thought>", "", cleaned_content, flags=re.DOTALL)
        response.content = cleaned_content.strip()
    return response

def compact_historical_file_messages(messages: List[BaseMessage]) -> List[BaseMessage]:
    compacted_messages = []
    last_ai_with_tools_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.type == "ai" and getattr(msg, "tool_calls", None):
            last_ai_with_tools_idx = i
            break
            
    for idx, msg in enumerate(messages):
        if msg.type == "tool" and msg.name in ["read_files", "read_file_lines", "write_file", "apply_search_replace_patch", "write_and_run_script"]:
            if idx < last_ai_with_tools_idx:
                original_content = str(msg.content)
                file_info = "tệp tin cũ"
                matches = re.findall(r"=== TỆP TIN:\s*[`']?([^`'\n]+)[`']?\s*===", original_content)
                if not matches:
                    matches = re.findall(r"tệp(?: tương đối)?:?\s*['`]?([^'`\n]+)['`]?", original_content)
                if matches:
                    file_info = f"tệp `{matches[0]}`"
                
                compacted_msg = ToolMessage(
                    content=f"[Đã tự động thu gọn dữ liệu cũ của {file_info} để tối ưu hóa bộ nhớ token. Nội dung mới nhất đã được cập nhật ở các bước sau nếu có chỉnh sửa]",
                    name=msg.name,
                    tool_call_id=msg.tool_call_id,
                    id=msg.id
                )
                compacted_messages.append(compacted_msg)
            else:
                compacted_messages.append(msg)
        else:
            compacted_messages.append(msg)
    return compacted_messages

def compact_reading_tool_messages(messages: List[BaseMessage]) -> List[BaseMessage]:
    compacted_messages = []
    for msg in messages:
        if msg.type == "tool" and msg.name in ["read_files"]:
            content_str = str(msg.content)
            found_files = []
            matches_read = re.findall(r"=== TỆP TIN:\s*[`']?([^`'\n]+)[`']?\s*===", content_str)
            if matches_read:
                found_files.extend(matches_read)
            file_info = f" của tệp {', '.join([f'`{f}`' for f in found_files])}" if found_files else ""
            
            compacted_msg = ToolMessage(
                content=f"[Đã nạp thành công dữ liệu vật lý{file_info} vào File Registry. Hãy sử dụng cấu trúc mã nguồn cập nhật mới nhất trong System Prompt để làm việc]",
                name=msg.name,
                tool_call_id=msg.tool_call_id,
                id=msg.id
            )
            compacted_messages.append(compacted_msg)
        else:
            compacted_messages.append(msg)
    return compacted_messages

def clear_compiler_cache(workspace_path: Path, ext: str):
    try:
        if ext == ".py":
            for pycache in workspace_path.rglob("__pycache__"):
                if pycache.is_dir():
                    shutil.rmtree(pycache, ignore_errors=True)
            for pyc in workspace_path.rglob("*.pyc"):
                if pyc.is_file():
                    pyc.unlink(missing_ok=True)
        elif ext in [".ts", ".tsx", ".js", ".jsx"]:
            ts_cache = workspace_path / "node_modules" / ".cache"
            if ts_cache.exists():
                shutil.rmtree(ts_cache, ignore_errors=True)
    except Exception as e:
        print(f"[Cảnh báo] Không thể dọn dẹp cache biên dịch: {str(e)}")

def find_nearest_config(start_path: Path, config_name: str, max_depth: int = 5) -> Optional[Path]:
    current = start_path.resolve()
    if current.is_file():
        current = current.parent
    for _ in range(max_depth):
        target = current / config_name
        if target.exists() and target.is_file():
            return current
        if current.parent == current:
            break
        current = current.parent
    return None

def execute_validation_cmd(cmd: List[str], cwd: Path, timeout: int = 30) -> Tuple[int, str]:
    executable = cmd[0]
    is_windows = platform.system() == "Windows"
    resolved_executable = shutil.which(executable)
    if not resolved_executable and is_windows:
        for ext in [".cmd", ".bat", ".exe"]:
            if shutil.which(executable + ext):
                cmd[0] = executable + ext
                resolved_executable = shutil.which(cmd[0])
                break
    if not resolved_executable:
        return (-99, f"Cảnh báo: Trình biên dịch/phân tích '{executable}' chưa được cài đặt trên hệ thống.")
    try:
        env_copy = os.environ.copy()
        env_copy["PYTHONIOENCODING"] = "utf-8"
        env_copy["PYTHONUTF8"] = "1"
        res = subprocess.run(
            cmd, cwd=str(cwd), capture_output=True, text=True,
            encoding="utf-8", errors="replace", env=env_copy, timeout=timeout
        )
        combined_output = (res.stdout or "") + "\n" + (res.stderr or "")
        return (res.returncode, combined_output.strip())
    except subprocess.TimeoutExpired:
        return (-2, f"Lỗi: Lệnh kiểm thử '{' '.join(cmd)}' bị treo và vượt quá thời gian chờ.")
    except Exception as e:
        return (-3, f"Lỗi hệ thống khi chạy lệnh kiểm thử: {str(e)}")

def clean_compiler_logs(raw_logs: str) -> str:
    lines = raw_logs.splitlines()
    filtered_lines = []
    noise_keywords = [
        "artifact instance of", "skipping update", "found plugin", "skipping generating",
        "generating", "starting test", "stopping scan", "listening to compiler", "compiling",
        "started flutter_tester process", "connected to test device", "waiting for test harness",
        "test harness is no longer needed", "ensuring test device is terminated", "terminating flutter_tester",
        "shutting down devtools", "deleting temporary directory", "runtime for phase", "exiting with code"
    ]
    error_keywords = ["error", "fail", "exception", "cause", "unhandled", "invalid", "undefined", "failed assertion"]
    for line in lines:
        clean_line = line.strip()
        if not clean_line or any(noise in clean_line.lower() for noise in noise_keywords):
            continue
        has_error_kw = any(kw in clean_line.lower() for kw in error_keywords)
        has_line_indicator = ":" in clean_line and (".dart" in clean_line or ".py" in clean_line or ".ts" in clean_line)
        is_test_failure_summary = "✗" in clean_line or "[E]" in clean_line
        if has_error_kw or has_line_indicator or is_test_failure_summary:
            filtered_lines.append(line)
    if not filtered_lines:
        if len(lines) > 20:
            return "\n".join(lines[:10] + ["... [Đã cắt bớt các log hệ thống không quan trọng] ..."] + lines[-10:])
        return raw_logs
    return "\n".join(filtered_lines)

def extract_path_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    quoted_paths = re.findall(r'["\']([^"\']+)["\']', text)
    for qp in quoted_paths:
        qp_clean = qp.strip()
        if qp_clean:
            try:
                resolved = Path(qp_clean).expanduser().resolve()
                if resolved.exists():
                    return str(resolved)
            except Exception:
                pass
    code_paths = re.findall(r'`([^`]+)`', text)
    for cp in code_paths:
        cp_clean = cp.strip()
        if cp_clean:
            try:
                resolved = Path(cp_clean).expanduser().resolve()
                if resolved.exists():
                    return str(resolved)
            except Exception:
                pass
    words = text.split()
    for word in words:
        cleaned = word.strip('`\'".,;()[]{}*')
        if not cleaned:
            continue
        is_path_like = (
            cleaned.startswith('/') or cleaned.startswith('~/') or 
            cleaned.startswith('./') or cleaned.startswith('.\\') or
            (len(cleaned) > 1 and cleaned[1] == ':' and (cleaned[2] == '/' or cleaned[2] == '\\'))
        )
        if is_path_like:
            try:
                resolved = Path(cleaned).expanduser().resolve()
                if resolved.exists():
                    return str(resolved)
            except Exception:
                pass
    return None

def find_extension_dir_heuristic(workspace_path: Path) -> Optional[str]:
    try:
        for path in workspace_path.rglob("manifest.json"):
            parts_lower = [p.lower() for p in path.parts]
            if not any(black in parts_lower for black in ["node_modules", ".venv", "venv", "env", "build", "dist", ".git"]):
                return str(path.parent.resolve())
    except Exception:
        pass
    return None

def resolve_special_system_paths(text: str) -> Optional[str]:
    text_lower = text.lower()
    home = Path.home()
    system_paths_map = {
        "desktop": home / "Desktop",
        "desktop có gì": home / "Desktop",
        "documents": home / "Documents",
        "tài liệu": home / "Documents",
        "downloads": home / "Downloads",
        "tải về": home / "Downloads",
    }
    for keyword, path in system_paths_map.items():
        if keyword in text_lower and path.exists():
            return str(path.resolve())
    return None

def verify_workspace_safety(workspace_path: str, allow_explicit: bool = False) -> bool:
    if allow_explicit:
        return True
    try:
        resolved_workspace = Path(workspace_path).expanduser().resolve()
        current_agent_dir = Path(__file__).parent.parent.parent.resolve()
        if resolved_workspace == current_agent_dir or resolved_workspace in current_agent_dir.parents:
            return False
        control_files = ["browser_subgraph.py", "mcp_helper.py", "routers.py"]
        if any((resolved_workspace / f).exists() for f in control_files):
            return False
        return True
    except Exception:
        return False

def encode_image_to_data_uri(image_path: Path) -> str:
    if not image_path.exists() or not image_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file ảnh tại: {image_path}")
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "image/png"
    img_bytes = image_path.read_bytes()
    encoded_string = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded_string}"