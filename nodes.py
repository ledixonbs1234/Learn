import base64
import json
import mimetypes
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path
import tempfile
from typing import Dict, Any, Optional,  Tuple, List
import uuid
from venv import logger
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langgraph.types import interrupt
from config import find_project_root_heuristic, model, sanitize_and_resolve_path, fast_model, sanitize_tool_result_content
from prompts_loader import load_prompt
from skills_engine import AgentSkillsEngine
from state import AgentState, PlanUpdate, TaskTriage, Task
from tools import (
    ActivateSkillTool, AskQuestionsTool, ChromeDebuggerTool, CompleteTaskTool, GitManager, QueryOpenWikiTool, ReadFileLinesTool, RunSkillScriptTool, SearchKeywordTool, UniversalSymbolSearchTool, WebAutonomousExecutorTool, WebInteractAndTestTool, WorkspaceTools, 
    ReadFilesTool, WriteAndRunScriptTool, WriteFileTool, ApplyPatchTool, 
    ListDirectoryTool, RunTerminalTool, get_markdown_language
)

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
    """
    Duyệt qua lịch sử hội thoại, giữ nguyên nội dung chi tiết của các ToolMessage 
    ở lượt chạy GẦN NHẤT (ở cuối danh sách tin nhắn), còn toàn bộ các ToolMessage 
    đọc/sửa file ở các lượt chạy trước đó sẽ bị làm gọn (compact) nội dung thành 
    mã tóm tắt gọn nhẹ để giải phóng token và tránh gây nhiễu loạn ngữ cảnh.
    """
    compacted_messages = []
    
    # Bước 1: Xác định vị trí của tin nhắn AI cuối cùng có chứa cuộc gọi công cụ (tool_calls)
    last_ai_with_tools_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.type == "ai" and getattr(msg, "tool_calls", None):
            last_ai_with_tools_idx = i
            break
            
    # Bước 2: Tiến hành thu hoạch và rút gọn các tin nhắn cũ
    for idx, msg in enumerate(messages):
        # Chỉ can thiệp vào các ToolMessage liên quan đến việc đọc/ghi file
        if msg.type == "tool" and msg.name in ["read_files", "read_file_lines", "write_file", "apply_search_replace_patch", "write_and_run_script"]:
            # Nếu ToolMessage này KHÔNG thuộc về lượt phản hồi cuối cùng (xuất hiện trước tin nhắn AI cuối cùng gọi công cụ)
            if idx < last_ai_with_tools_idx:
                original_content = str(msg.content)
                file_info = "tệp tin cũ"
                
                # Trích xuất thông tin tên tệp từ tiêu đề nếu có
                matches = re.findall(r"=== TỆP TIN:\s*[`']?([^`'\n]+)[`']?\s*===", original_content)
                if not matches:
                    matches = re.findall(r"tệp(?: tương đối)?:?\s*['`]?([^'`\n]+)['`]?", original_content)
                if matches:
                    file_info = f"tệp `{matches[0]}`"
                
                # Tạo tin nhắn stub gọn nhẹ thay thế cho nội dung khổng lồ trước đó
                compacted_msg = ToolMessage(
                    content=f"[Đã tự động thu gọn dữ liệu cũ của {file_info} để tối ưu hóa bộ nhớ token. Nội dung mới nhất đã được cập nhật ở các bước sau nếu có chỉnh sửa]",
                    name=msg.name,
                    tool_call_id=msg.tool_call_id,
                    id=msg.id
                )
                compacted_messages.append(compacted_msg)
            else:
                # Giữ nguyên vẹn đối với lượt chạy hoạt động gần nhất ở cuối đồ thị
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
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env_copy, 
            timeout=timeout
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
        if not clean_line:
            continue
            
        if any(noise in clean_line.lower() for noise in noise_keywords):
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
            cleaned.startswith('/') or 
            cleaned.startswith('~/') or 
            cleaned.startswith('./') or 
            cleaned.startswith('.\\') or
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
        if keyword in text_lower:
            if path.exists():
                return str(path.resolve())
    return None

def verify_workspace_safety(workspace_path: str, allow_explicit: bool = False) -> bool:
    if allow_explicit:
        return True
    try:
        resolved_workspace = Path(workspace_path).expanduser().resolve()
        current_agent_dir = Path(__file__).parent.parent.resolve()
        
        if resolved_workspace == current_agent_dir or resolved_workspace in current_agent_dir.parents:
            return False
            
        control_files = ["browser_subgraph.py", "mcp_helper.py", "routers.py"]
        if any((resolved_workspace / f).exists() for f in control_files):
            return False
            
        return True
    except Exception:
        return False

def triage_node_stateful(state: AgentState, catalog_summary: str, mcp_summary: str) -> TaskTriage:
    messages = state.get("messages", [])
    current_workspace = state.get("workspace_path", "")
    
    active_session_context = (
        "=== NGỮ CẢNH PHIÊN HOẠT ĐỘNG HIỆN TẠI (ACTIVE SESSION) ===\n"
        f"- Thư mục làm việc hiện hành (Workspace): `{current_workspace or 'Chưa thiết lập'}`\n"
    )

    context_messages = []
    if len(messages) > 1:
        recent_history = messages[-5:-1]
        context_messages.append(SystemMessage(content="--- LỊCH SỬ HỘI THOẠI GẦN NHẤT ĐỂ THAM CHIẾU NGỮ CẢNH ---"))
        context_messages.extend(recent_history)

    user_msg = messages[-1]
    user_query_text = get_text_content_safely(user_msg.content)

    # 🌟 CẢI TIẾN: Nạp system prompt từ file tĩnh chuyên biệt
    system_prompt_template = load_prompt("triage_stateful.txt")
    system_prompt = system_prompt_template.replace(
        "{catalog_summary}", catalog_summary
    ).replace(
        "{mcp_summary}", mcp_summary
    )

    structured_llm = model.with_structured_output(TaskTriage, method="function_calling")
    
    triage_output = structured_llm.invoke([
        SystemMessage(content=system_prompt),
        SystemMessage(content=active_session_context),
        *context_messages,
        HumanMessage(content=f"Yêu cầu hiện tại của người dùng: {user_query_text}")
    ])
    
    return triage_output


def encode_image_to_data_uri(image_path: Path) -> str:
    """Đọc ảnh từ đĩa cứng và chuyển đổi thành cấu trúc Base64 Data URI."""
    if not image_path.exists() or not image_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file ảnh tại: {image_path}")
    
    # Tự động nhận diện MIME type (ví dụ: image/png, image/jpeg)
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "image/png" # Fallback mặc định
        
    img_bytes = image_path.read_bytes()
    encoded_string = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded_string}"

def detect_and_triage_node(state: AgentState) -> Dict[str, Any]:
    messages = state["messages"]
    user_msg = messages[-1]
    user_query_text = get_text_content_safely(user_msg.content)
    
    # 1. Định vị Workspace an toàn
    random_hex = uuid.uuid4().hex[:8]
    temp_workspace_path = Path(tempfile.gettempdir()) / f"agent_{os.getpid()}_{random_hex}"
    temp_workspace_path.mkdir(parents=True, exist_ok=True)
    safe_temp_dir = str(temp_workspace_path.resolve())
    
    existing_workspace = state.get("workspace_path", "")
    if not existing_workspace or existing_workspace == ".":
        existing_workspace = safe_temp_dir

    detected_path_str = extract_path_from_text(user_query_text)
    if not detected_path_str:
        path_keywords = ["thư mục", "folder", "dự án", "project", "mở", "quét", "ls", "dir", "làm việc tại", "tại", "cd"]
        if any(kw in user_query_text.lower() for kw in path_keywords) or not existing_workspace:
            detected_path_str = resolve_special_system_paths(user_query_text)

    provisional_workspace = existing_workspace
    pivoted_msg = ""

    if detected_path_str:
        try:
            resolved_path = Path(detected_path_str).expanduser().resolve()
            if resolved_path.exists():
                provisional_workspace = str(find_project_root_heuristic(resolved_path))
                if provisional_workspace != existing_workspace and existing_workspace:
                    pivoted_msg = f"🔄 **[Chuyển đổi Workspace]**: Phát hiện yêu cầu chuyển đổi thư mục làm việc sang: `{provisional_workspace}`\n"
        except Exception:
            pass
    else:
        if existing_workspace:
            pivoted_msg = f"🔄 **[Kế thừa Workspace]**: Sử dụng lại thư mục làm việc hiện hành: `{existing_workspace}`\n"

    # =====================================================================
    # XỬ LÝ NHÚNG HÌNH ẢNH VÀO TIN NHẮN NGƯỜI DÙNG (MULTIMODAL UPDATE)
    # =====================================================================
    image_paths = state.get("image_paths", []) or []
    updated_messages = []
    
    # Chỉ xử lý khi có danh sách ảnh đầu vào và tin nhắn cuối là HumanMessage
    if image_paths and (user_msg.type == "human" or isinstance(user_msg, HumanMessage)):
        # Tạo cấu trúc nội dung đa phương thức mới
        multimodal_content = [{"type": "text", "text": user_query_text}]
        
        for path_str in image_paths:
            try:
                # Phân giải đường dẫn ảnh tương đối dựa trên workspace hoạt động
                safe_img_path = sanitize_and_resolve_path(provisional_workspace, path_str)
                if safe_img_path.exists() and safe_img_path.is_file():
                    data_uri = encode_image_to_data_uri(safe_img_path)
                    
                    # Thêm phân đoạn ảnh vào cấu trúc Multimodal
                    multimodal_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": data_uri,
                            "detail": "low"  # 'low' giúp tiết kiệm token đáng kể nếu không cần phân tích điểm ảnh siêu nhỏ
                        }
                    })
            except Exception as e:
                print(f"[Cảnh báo hệ thống] Lỗi xử lý hình ảnh '{path_str}': {str(e)}")
                
        # Khởi tạo tin nhắn HumanMessage mới có ID trùng khớp với tin nhắn cũ để thực hiện ghi đè
        updated_user_msg = HumanMessage(
            content=multimodal_content,
            id=user_msg.id  # ĐÂY LÀ ĐIỂM QUAN TRỌNG NHẤT
        )
        updated_messages.append(updated_user_msg)

    # 2. Thực hiện quét kỹ năng và MCP cấu hình
    temp_state = state.copy()
    temp_state["workspace_path"] = provisional_workspace
    
    skills_engine = AgentSkillsEngine(provisional_workspace)
    catalog = skills_engine.scan_catalog()
    
    catalog_summary = ""
    if catalog:
        catalog_summary = "\n=== 📚 DANH MỤC KỸ NĂNG HỆ THỐNG HIỆN CÓ ===\n"
        for item in catalog:
            catalog_summary += f"- Kỹ năng: `{item['name']}`\n  Điều kiện kích hoạt: {item['description']}\n"

    mcp_summary = ""
    try:
        from mcp_helper import MCPRegistryManager
        mcp_manager = MCPRegistryManager(provisional_workspace)
        raw_servers = mcp_manager.load_config()
        if raw_servers:
            mcp_summary = "\n=== 🔌 MÁY CHỦ MCP NGOẠI VI KHẢ DỤNG ===\n"
            for name, cfg in raw_servers.items():
                mcp_summary += f"- Máy chủ: `{name}` (Hành vi: {cfg.get('transport', 'stdio')})\n"
    except Exception:
        pass

    if provisional_workspace and provisional_workspace != existing_workspace:
        temp_ext = find_extension_dir_heuristic(Path(provisional_workspace))
        temp_state["extension_path"] = temp_ext or ""

    try:
        triage_output = triage_node_stateful(temp_state, catalog_summary, mcp_summary)
        task_type = triage_output.task_type
        is_simple = triage_output.is_simple
        detailed_analysis = triage_output.detailed_analysis
        recommended_skills = triage_output.recommended_skills or []
        recommended_mcp_servers = triage_output.recommended_mcp_servers or []
    except Exception as e:
        task_type = "clarify" if not provisional_workspace else "analysis"
        is_simple = True
        detailed_analysis = f"Lỗi hệ thống phân loại: {str(e)}"
        recommended_skills = []
        recommended_mcp_servers = []

    final_workspace = provisional_workspace
    if task_type == "clarify" or not final_workspace:
        interrupt_payload = {
            "type": "path_clarification",
            "prompt": "Hệ thống phát hiện bạn muốn làm việc với ứng dụng nhưng chưa cấu hình thư mục làm việc cụ thể.",
            "fields": [
                {
                    "name": "target_workspace_path",
                    "label": "Đường dẫn thư mục dự án của bạn",
                    "placeholder": "Ví dụ: C:/Users/Name/Projects/my-app",
                    "type": "text",
                    "required": True
                }
            ]
        }
        user_response = interrupt(interrupt_payload)
        
        if isinstance(user_response, dict) and "target_workspace_path" in user_response:
            detected_path_str = str(user_response["target_workspace_path"]).strip()
        elif isinstance(user_response, str):
            detected_path_str = user_response.strip()

        try:
            final_workspace = str(Path(detected_path_str).expanduser().resolve())
        except Exception:
            final_workspace = "."

    is_user_explicit = (detected_path_str is not None)
    
    if not is_simple and not verify_workspace_safety(final_workspace, allow_explicit=is_user_explicit):
        return {
            "workspace_path": final_workspace,
            "plan": [], 
            "task_type": "analysis", 
            "is_simple": True,
            "detailed_analysis": "Ngắt hoạt động do vi phạm rào chắn bảo mật Workspace dành cho tác vụ phức tạp.",
            "messages": [AIMessage(content="🚨 **[CẢNH BÁO BẢO MẬT]**: Tác vụ phát triển phức tạp yêu cầu một Workspace an toàn bên ngoài thư mục Agent. Vui lòng chỉ định một thư mục làm việc hợp lệ.")]
        }

    active_skills = {} 
    analysis_triad = ["grill-with-docs", "write-a-prd", "domain-modeling"]
    requires_triad = any(skill in recommended_skills for skill in analysis_triad)

    plan = []
    if is_simple:
        plan = [
            Task(id="T1", description=f"Thực hiện trực tiếp tác vụ tại `{final_workspace}`: {user_query_text}", dependencies=[], status="pending")
        ]
    else:
        if requires_triad:
            survey_desc = (
                f"Bắt đầu pha khảo sát. Bạn hãy gọi công cụ `activate_agent_skill` để kích hoạt tuần tự và "
                f"thực thi các kỹ năng `grill-with-docs` (chất vấn làm rõ yêu cầu về: {user_query_text}), "
                f"`domain-modeling` (đồng bộ bảng thuật ngữ vào `CONTEXT.md`), "
                f"và `write-a-prd` (thiết lập tệp đặc tả `PRD.md`)."
            )
        else:
            survey_desc = f"Khảo sát cấu trúc file và mã nguồn tại `{final_workspace}` liên quan đến yêu cầu: {user_query_text} và thu thập dữ liệu để lập kế hoạch chi tiết."

        plan = [
            Task(id="T_SURVEY", description=survey_desc, dependencies=[], status="pending")
        ]
        task_type = "analysis"

    skills_log_msg = ""
    if recommended_skills:
        skills_log_msg = f"- 📋 **Kỹ năng đề xuất (Đã nạp sẵn cho pha khảo sát):** {', '.join([f'`{s}`' for s in recommended_skills])}\n"

    mcp_log_msg = ""
    if recommended_mcp_servers:
        mcp_log_msg = f"- 🔌 **Máy chủ MCP được AI kích hoạt:** {', '.join([f'`{s}`' for s in recommended_mcp_servers])}\n"

    triage_info_msg = (
        f"📊 **[Hệ thống Phân phối thông minh]**:\n"
        f"{pivoted_msg}"
        f"- **Workspace hoạt động:** `{final_workspace}`\n"
        f"- **Chế độ kiểm soát:** {'Đơn giản (Fast-Track)' if is_simple else 'Phức tạp (Multi-Step Discovery)'}\n"
        f"- **Pha hoạt động khởi động:** `{task_type.upper()}`\n"
        f"{skills_log_msg}"
        f"{mcp_log_msg}\n"
        f"🎯 **[Phân tích mục tiêu kỹ thuật]**:\n{detailed_analysis}"
    )

    # 3. Trả về kết quả cập nhật trạng thái đồ thị
    return {
        "workspace_path": final_workspace,
        "plan": plan,
        "task_type": task_type,
        "is_simple": is_simple,
        "detailed_analysis": detailed_analysis,
        # Trả về updated_messages (chứa HumanMessage đè ID cũ) kèm theo tin nhắn AI mới phân tích
        "messages": updated_messages + [AIMessage(content=triage_info_msg)],
        "error_logs": "",
        "attempts": 0,
        "modified_files": [],
        "last_executed_task_ids": [],
        "replanning_count": 0,
        "step_findings": ["__RESET__"],
        "active_skills": active_skills, 
        "recommended_skills": recommended_skills,
        "active_mcp_servers": recommended_mcp_servers
    }

def get_eligible_tasks(plan: List[Any]) -> List[Any]:
    completed_ids = set()
    for t in plan:
        t_id = t.get("id") if isinstance(t, dict) else getattr(t, "id", None)
        t_status = t.get("status") if isinstance(t, dict) else getattr(t, "status", None)
        if t_status == "completed":
            completed_ids.add(t_id)
            
    eligible = []
    for t in plan:
        t_status = t.get("status") if isinstance(t, dict) else getattr(t, "status", None)
        t_deps = t.get("dependencies", []) if isinstance(t, dict) else (getattr(t, "dependencies", None) or [])
        
        if t_status == "pending":
            if all(dep in completed_ids for dep in t_deps):
                eligible.append(t)
    return eligible

def doubt_reviewer_node(state: AgentState) -> Dict[str, Any]:
    modified_files = state.get("modified_files", [])
    file_registry = state.get("file_registry", {})
    doubt_attempts = state.get("doubt_attempts", 0)
    
    if not modified_files or doubt_attempts >= 3:
        return {"doubt_findings": ""}
        
    latest_file = modified_files[-1]
    artifact_code = file_registry.get(latest_file, "")
    
    if not artifact_code:
        try:
            safe_path = sanitize_and_resolve_path(state["workspace_path"], latest_file)
            if safe_path.exists():
                artifact_code = safe_path.read_text(encoding="utf-8")
        except Exception:
            pass

    if not artifact_code:
        return {"doubt_findings": ""}

    adversarial_prompt_template = load_prompt("doubt_reviewer.txt")
    adversarial_prompt = adversarial_prompt_template.replace(
        "{latest_file}", latest_file
    ) + "\n```\n" + artifact_code + "\n```"

    response = fast_model.invoke([
        SystemMessage(content="Bạn đang thực thi quy trình thẩm định đối kháng thuộc kỹ năng `doubt-driven-development`."),
        HumanMessage(content=adversarial_prompt)
    ])
    
    return {
        "doubt_findings": response.content,
        "doubt_attempts": doubt_attempts + 1
    }

def doubt_gate_node(state: AgentState) -> Dict[str, Any]:
    doubt_findings = state.get("doubt_findings", "")
    modified_files = state.get("modified_files", [])
    
    if not doubt_findings or not modified_files:
        return {}

    latest_file = modified_files[-1]

    interrupt_payload = {
        "title": "🔍 THẨM ĐỊNH ĐỐI KHÁNG (DOUBT-DRIVEN DEVELOPMENT)",
        "file_under_review": latest_file,
        "single_model_findings": doubt_findings,
        "prompt": (
            "Hệ thống phát hiện một số rủi ro logic tiềm ẩn trong đoạn mã nguồn bạn vừa viết.\n"
            "- Nhấn Approve (hoặc gửi phản hồi rỗng/'yes'/'ok') để BỎ QUA và tiến hành commit.\n"
            "- Nhập 'gemini' hoặc 'codex' để kích hoạt THẨM ĐỊNH CHÉO ngoại vi thông qua mô hình độc lập (Cross-Model Escalation).\n"
            "- Nhập ý kiến phản hồi khác hoặc yêu cầu sửa lỗi để chuyển thông tin này quay lại cho Executor khắc phục lỗi."
        )
    }

    user_input = interrupt(interrupt_payload)
    user_action = str(user_input).strip().lower() if user_input else ""

    if user_action in ["", "yes", "approve", "ok", "skip"]:
        return {
            "error_logs": "",
            "doubt_findings": "",
            "messages": [AIMessage(content="✅ **[Doubt Bypassed]** Người dùng đã phê duyệt mã nguồn. Tiến hành hoàn tất tác vụ.")]
        }

    if user_action in ["gemini", "codex"]:
        cli_tool = user_action
        cli_executable = "gemini" if cli_tool == "gemini" else "codex"
        
        if not shutil.which(cli_executable):
            feedback_msg = HumanMessage(
                content=f"⚠️ Lỗi: Không tìm thấy thực thi CLI `{cli_executable}` trong biến môi trường PATH của bạn."
            )
            return {
                "error_logs": f"Không tìm thấy công cụ ngoại vi `{cli_executable}`",
                "messages": [feedback_msg]
            }

        file_registry = state.get("file_registry", {})
        artifact_code = file_registry.get(latest_file, "")
        
        cross_prompt = (
            f"Thẩm định đối kháng chéo (Adversarial Cross-Model Review) cho file {latest_file}.\n"
            "Hãy tìm ra các lỗ hổng, lỗi logic hoặc điểm chưa tối ưu mà mô hình trước đã bỏ qua.\n\n"
            "MÃ NGUỒN:\n"
            f"{artifact_code}"
        )

        temp_prompt_file_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8", delete=False) as temp_prompt_file:
                temp_prompt_file.write(cross_prompt)
                temp_prompt_file_path = temp_prompt_file.name

            with open(temp_prompt_file_path, "r", encoding="utf-8") as stdin_file:
                env_copy = os.environ.copy()
                env_copy["PYTHONIOENCODING"] = "utf-8"
                env_copy["PYTHONUTF8"] = "1"
                
                cmd = [cli_executable]
                if cli_executable == "gemini":
                    cmd.extend(["--approval-mode", "plan", "-p", ""])
                elif cli_executable == "codex":
                    cmd.extend(["exec", "--sandbox", "read-only", "-"])
                
                res = subprocess.run(
                    cmd,
                    stdin=stdin_file,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=env_copy,
                    timeout=45
                )

            combined_output = (res.stdout or "") + "\n" + (res.stderr or "")
            escalated_findings = combined_output.strip()
            
            return {
                "doubt_findings": f"🛡️ **[KẾT QUẢ THẨM ĐỊNH CHÉO TỪ {cli_executable.upper()}]**:\n\n{escalated_findings}",
                "messages": [AIMessage(content=f"🔍 Kích hoạt thành công rà soát chéo từ {cli_executable.upper()}. Đang chờ ý kiến phê duyệt cuối cùng.")]
            }
            
        except Exception as err:
            return {
                "error_logs": f"Lỗi hệ thống khi khởi chạy Cross-Model: {str(err)}",
                "messages": [AIMessage(content=f"❌ Thao tác gọi mô hình chéo thất bại: {str(err)}")]
            }
        finally:
            if temp_prompt_file_path and Path(temp_prompt_file_path).exists():
                try:
                    Path(temp_prompt_file_path).unlink(missing_ok=True)
                except Exception:
                    pass

    feedback_message = HumanMessage(
        content=(
            "⚠️ Yêu cầu sửa đổi mã nguồn dựa trên kết quả thẩm định đối kháng:\n"
            f"Ý kiến người dùng: '{user_input}'\n"
            f"Các lỗi cần khắc phục:\n{doubt_findings}"
        )
    )
    
    return {
        "error_logs": f"Cần khắc phục lỗi logic thẩm định: {user_input}",
        "doubt_findings": "",
        "messages": [feedback_message]
    }

def context_loader_node(state: AgentState) -> Dict[str, Any]:
    ws = state["workspace_path"]
    
    git_dir = Path(ws) / ".git"
    git_branch = "no_git"
    git_msg = "ℹ️ Không phát hiện Git repository. Kích hoạt chế độ Sửa đổi trực tiếp (Bypass Git)."
    
    if git_dir.exists():
        try:
            git_manager = GitManager(ws)
            git_branch = git_manager.init_and_prepare_branch()
            git_msg = f"Đã cấu hình nhánh Git hoạt động: `{git_branch}`"
        except Exception as e:
            git_branch = "no_git"
            git_msg = f"⚠️ Có lỗi xảy ra khi nạp Git ({str(e)}). Tự động chuyển sang chế độ Sửa đổi trực tiếp."
            
    workspace_context = ""
    context_path = Path(ws) / "CONTEXT.md"
    context_msg = "📋 Không tìm thấy tệp cấu hình `CONTEXT.md`."
    
    if context_path.exists():
        try:
            workspace_context = context_path.read_text(encoding="utf-8")
            context_msg = "📋 Đã tải xong ngữ cảnh thông tin dự án từ tệp `CONTEXT.md`."
        except Exception as e:
            workspace_context = f"Lỗi khi đọc file CONTEXT.md: {str(e)}"
            context_msg = f"⚠️ Gặp sự cố khi đọc tệp `CONTEXT.md`: {str(e)}"
            
    ext_dir = find_extension_dir_heuristic(Path(ws))
    ext_msg = ""
    if ext_dir:
        ext_msg = f"\n📦 **[Tự động nhận diện Extension]**: Đã định vị thư mục Chrome Extension tại: `{ext_dir}`"
    else:
        ext_msg = "\n📦 **[Tự động nhận diện Extension]**: Không tìm thấy manifest.json trực tiếp trong thư mục workspace."
        
    return {
        "workspace_context": workspace_context,
        "git_branch": git_branch,
        "extension_path": ext_dir or "",
        "browser_console_logs": "",
        "messages": [AIMessage(content=f"{git_msg}\n{context_msg}{ext_msg}")]
    }

def triage_node(state: AgentState) -> Dict[str, Any]:
    messages = state["messages"]
    user_msg = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) or getattr(msg, "type", None) == "human":
            user_msg = msg
            break
            
    if not user_msg:
        user_msg = messages[0]
        
    user_query_text = get_text_content_safely(user_msg.content)
    
    structured_llm = model.with_structured_output(TaskTriage, method="function_calling")
    
    system_prompt = load_prompt("triage_simple.txt")
    
    try:
        triage_output = structured_llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query_text}
        ])
        is_simple = getattr(triage_output, "is_simple", False)
        task_type = getattr(triage_output, "task_type", "development")
        detailed_analysis = getattr(triage_output, "detailed_analysis", "")
    except Exception:
        is_simple = False
        task_type = "development"
        detailed_analysis = "Không thể phân tích tự động mục tiêu yêu cầu của người dùng."

    plan = []
    if is_simple:
        plan = [
            Task(
                id="T1",
                description=f"Thực hiện trực tiếp yêu cầu: {user_query_text}",
                dependencies=[],
                status="pending"
            )
        ]
    else:
        plan = [
            Task(
                id="T_SURVEY",
                description="Khảo sát cấu trúc thư mục, tệp cấu hình manifest.json và mã nguồn chính của dự án bằng các công cụ khảo sát để nắm vững kiến trúc trước khi lập kế hoạch.",
                dependencies=[],
                status="pending"
            )
        ]
        task_type = "analysis" 
        
    return {
        "plan": plan,
        "task_type": task_type,
        "is_simple": is_simple,
        "detailed_analysis": detailed_analysis,
        "messages": [
            AIMessage(
                content=f"📊 **[Phân loại tác vụ]**: Hệ thống xác định yêu cầu thuộc diện "
                        f"{'ĐƠN GIẢN (Fast-Track)' if is_simple else 'PHỨC TẠP (Multi-Step Discovery)'} | "
                        f"Pha hoạt động khởi động: `{task_type.upper()}`.\n\n"
                        f"🎯 **[Phân tích mục tiêu]**:\n{detailed_analysis}"
            )
        ]
    }

def context_compressor_node(state: AgentState) -> Dict[str, Any]:
    messages = state.get("messages", [])
    ws = state.get("workspace_path", ".")
    active_skills = state.get("active_skills", {}) or {}
    workspace_root = Path(ws).expanduser().resolve()
    
    latest_task_summary = ""
    task_id = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc["name"] == "complete_agent_task":
                    latest_task_summary = tc["args"].get("summary", "")
                    task_id = tc["args"].get("task_id", "")
                    break
            if latest_task_summary:
                break

    new_summaries = []
    if latest_task_summary:
        new_summaries.append(f"Task {task_id}: {latest_task_summary}")

    triage_msg_id = None
    for msg in messages:
        if msg.type == "ai" or isinstance(msg, AIMessage):
            content_str = str(msg.content)
            if "Phân phối thông minh" in content_str or "Phân loại tác vụ" in content_str:
                triage_msg_id = msg.id
                break
                
    final_summary_id = None
    if messages:
        last_msg = messages[-1]
        if (last_msg.type == "ai" or isinstance(last_msg, AIMessage)) and not getattr(last_msg, "tool_calls", None):
            final_summary_id = last_msg.id

    deletion_list = []
    has_kept_root_user_msg = False
    
    for msg in messages:
        if not msg.id:
            continue
            
        is_human = (msg.type == "human" or isinstance(msg, HumanMessage))
        
        if is_human and not has_kept_root_user_msg:
            has_kept_root_user_msg = True
            continue
            
        is_triage = (msg.id == triage_msg_id)
        is_final_summary = (msg.id == final_summary_id)
        
        if not (is_human or is_triage or is_final_summary):
            deletion_list.append(RemoveMessage(id=msg.id))

    compiled_context_parts = []
    
    prd_path = workspace_root / "PRD.md"
    if prd_path.exists():
        try:
            compiled_context_parts.append(f"### [YÊU CẦU SẢN PHẨM (PRD.md)]\n{prd_path.read_text(encoding='utf-8')}")
        except Exception:
            pass

    context_file_path = workspace_root / "CONTEXT.md"
    if context_file_path.exists():
        try:
            compiled_context_parts.append(f"### [KIẾN TRÚC HỆ THỐNG & THUẬT NGỮ (CONTEXT.md)]\n{context_file_path.read_text(encoding='utf-8')}")
        except Exception:
            pass

    historical_summaries = state.get("completed_task_summaries", []) + new_summaries
    if historical_summaries:
        history_block = "### [LỊCH SỬ THỰC THI PHIÊN CHẠY (BỘ NHỚ TẠM THỜI IN-MEMORY)]\n"
        history_block += "\n".join([f"- {s}" for s in historical_summaries])
        compiled_context_parts.append(history_block)

    super_context = "\n\n---\n\n".join(compiled_context_parts)

    normalized_modified_set = set()
    for f in state.get("modified_files", []) or []:
        try:
            f_path = Path(f).resolve()
            if f_path.exists():
                try:
                    rel = str(f_path.relative_to(workspace_root))
                    normalized_modified_set.add(rel)
                except ValueError:
                    normalized_modified_set.add(str(f_path))
        except Exception:
            pass

    current_registry = state.get("file_registry", {}) or {}
    registry_eviction_updates = {}
    for cached_file in current_registry.keys():
        if cached_file not in normalized_modified_set:
            registry_eviction_updates[cached_file] = None

    cleaned_active_skills = dict(active_skills)
    discovery_triad = ["grill-with-docs", "write-a-prd", "domain-modeling"]
    for skill_name in discovery_triad:
        cleaned_active_skills.pop(skill_name, None)
    
    clean_checkpoint_msg = AIMessage(
        content=(
            f"🔄 **[Hệ thống nén ngữ cảnh]**:\n"
            f"- Đã lưu tóm tắt nhiệm vụ `{task_id or 'N/A'}` vào bộ đệm trạng thái tạm thời.\n"
            f"- Đã thu gọn và giải phóng thành công lịch sử tin nhắn rác khỏi bộ nhớ RAM.\n"
            f"- Giải phóng {len(registry_eviction_updates)} tệp tin không thay đổi khỏi File Registry."
        )
    )
    
    return {
        "workspace_context": super_context,
        "messages": deletion_list + [clean_checkpoint_msg],
        "active_skills": cleaned_active_skills,
        "completed_task_summaries": new_summaries,
        "file_registry": registry_eviction_updates
    }
    
def fluxmem_distillation_node(state: AgentState) -> Dict[str, Any]:
    messages = state.get("messages", [])
    ws = state["workspace_path"]
    plan = state.get("plan", [])
    
    latest_task_id = ""
    latest_task_desc = "Tác vụ thực thi hệ thống"
    
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc["name"] == "complete_agent_task":
                    latest_task_id = tc["args"].get("task_id", "")
                    break
            if latest_task_id:
                break
                
    if not latest_task_id:
        return {} 
        
    for t in plan:
        t_id = t.id if isinstance(t, Task) else t.get("id")
        if t_id == latest_task_id:
            latest_task_desc = t.description if isinstance(t, Task) else t.get("description")
            break

    current_task_messages = []
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and "📊 [Hệ thống Phân phối thông minh]" in str(msg.content):
            break
        current_task_messages.insert(0, msg)
        
    steps_prompt_parts = []
    for m in current_task_messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                steps_prompt_parts.append(
                    f"- **AI đã gọi công cụ:** `{tc['name']}` với các đối số: `{json.dumps(tc['args'], ensure_ascii=False)}`"
                )
        elif m.type == "tool":
            content_str = str(m.content)
            if len(content_str) > 800:
                content_str = content_str[:800] + "... [Đã cắt bớt dữ liệu thô dài] ..."
            steps_prompt_parts.append(f"  -> *Kết quả phản hồi của công cụ:* {content_str}")

    steps_prompt = "\n".join(steps_prompt_parts) if steps_prompt_parts else "Không ghi nhận lượt gọi công cụ trực tiếp."

    system_prompt = (
        "Bạn là Chuyên gia chưng cất kỹ năng (FluxMem Distillation Agent).\n"
        "Dưới đây là vết lịch sử chạy thực tế thành công của Agent:\n\n"
        f"[MỤC TIÊU]: \"{latest_task_desc}\"\n"
        "[HÀNH ĐỘNG CHI TIẾT]:\n"
        f"{steps_prompt}\n\n"
        "Hãy viết một quy trình (Procedural Skill) gồm 3-5 bước tổng quát hóa, mô tả chính xác cách thiết thiết kế, biên tập, các công cụ tối ưu cần gọi và cách phòng ngừa lỗi biên dịch cho mục tiêu này.\n"
        "Chỉ trả về nội dung quy trình dạng Markdown, không viết thêm lời mở đầu hay giải thích"
    )

    try:
        distilled_response = fast_model.invoke([
            SystemMessage(content=system_prompt)
        ])
        
        procedural_skill_md = distilled_response.content
        wiki_dir = Path("~/.openwiki/wiki").expanduser().resolve()
        wiki_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            slug_prompt = (
                "Dựa vào mô tả nhiệm vụ sau, hãy tạo ra một định danh (slug) ngắn gọn từ 2-4 từ, "
                "viết thường, KHÔNG dấu, phân cách bằng duy nhất dấu gạch dưới, mô tả khái quát kỹ năng kỹ thuật "
                "cốt lõi của tác vụ này.\n"
                "⚠️ YÊU CẦU NGHIÊM NGẶT:\n"
                "- TUYỆT ĐỐI KHÔNG bao gồm bất kỳ đường dẫn thư mục, ổ đĩa C:/, tên file cục bộ, hoặc tên người dùng nào.\n"
                "- Nếu nhiệm vụ là cào dữ liệu, slug có thể là 'web_data_scraping'.\n"
                "- Nếu nhiệm vụ là cấu hình database, slug có thể là 'database_configuration'.\n\n"
                f"Mô tả nhiệm vụ ban đầu: {latest_task_desc}\n\n"
                "Chỉ trả về chuỗi định danh duy nhất (ví dụ: 'setup_playwright_scraper', 'write_prd_specification'):"
            )
            slug_response = fast_model.invoke([SystemMessage(content=slug_prompt)])
            
            # 🌟 CẢI TIẾN 1: Khử hoàn toàn các thẻ <thinking> và <thought> rò rỉ từ mô hình lý luận (Reasoning Model)
            slug_response = sanitize_llm_response_content(slug_response)
            raw_slug_content = slug_response.content.strip().lower()
            
            # 🌟 CẢI TIẾN 2: Trích xuất các từ hợp lệ dạng slug (chữ cái và gạch dưới, độ dài từ 3 đến 40 ký tự)
            # Thay vì dính chuỗi thô, ta bóc tách riêng lẻ các từ khóa tiềm năng
            slug_candidates = re.findall(r'\b[a-z0-9_]{3,40}\b', raw_slug_content)
            
            # Danh sách từ dừng (stopwords) hệ thống để tránh trích xuất nhầm các câu giải thích của AI
            system_stopwords = {
                "thinking", "the", "user", "is", "asking", "me", "to", "create", "a", "slug", 
                "identifier", "based", "on", "task", "description", "markdown", "python", 
                "task_id", "procedural", "skill", "would", "be", "something", "like"
            }
            safe_candidates = [c for c in slug_candidates if c not in system_stopwords]
            
            if safe_candidates:
                # Ưu tiên chọn từ khóa cuối cùng (thường là kết luận lựa chọn slug của mô hình)
                safe_task_name = safe_candidates[-1]
            else:
                safe_task_name = f"procedural_task_{latest_task_id.lower()}"
                
            # 🌟 CẢI TIẾN 3: Ép giới hạn độ dài ký tự tối đa (Strict Bound) để tuyệt đối không vi phạm giới hạn MAX_PATH của OS
            safe_task_name = safe_task_name[:45]
            
        except Exception as slug_err:
            print(f"[Cảnh báo] Lỗi sinh slug bằng AI: {str(slug_err)}. Chuyển sang fallback phòng ngự.")
            safe_task_name = f"procedural_task_{latest_task_id.lower()}"
            
        file_name = f"skill_{safe_task_name}.md"
        dest_file = wiki_dir / file_name
        
        dest_file.write_text(procedural_skill_md, encoding="utf-8")
        
        log_msg = (
            f"⚡ **[Global FluxMem Distillation]**: Đã chưng cất tri thức thành công! "
            f"Lưu trữ vật lý tại OpenWiki toàn cục: `~/.openwiki/wiki/{file_name}`."
        )
        return {
            "messages": [AIMessage(content=log_msg)]
        }
    except Exception as e:
        print(f"[Cảnh báo] Lỗi trong quá trình chưng cất quy trình toàn cục: {str(e)}")
        return {}
def crawl_project_sources(workspace_path: Path) -> str:
    """Quét đệ quy toàn bộ mã nguồn hợp lệ trong dự án, loại bỏ các file bị ignore."""
    from config import GitIgnoreMatcher
    matcher = GitIgnoreMatcher(workspace_path)
    sources = []
    
    allowed_extensions = {
        '.py', '.js', '.jsx', '.ts', '.tsx', '.dart', '.go', '.rs', '.java', 
        '.cpp', '.h', '.hpp', '.cs', '.kt', '.swift', '.json', '.yaml', '.yml', 
        '.md', '.html', '.css', '.toml', '.xml', '.gradle', '.bat', '.sh'
    }
    
    try:
        for root, dirs, files in os.walk(str(workspace_path)):
            dirs[:] = [d for d in dirs if not matcher.is_ignored(Path(root) / d)]
            
            for file in files:
                file_path = Path(root) / file
                if matcher.is_ignored(file_path):
                    continue
                if file_path.suffix.lower() not in allowed_extensions:
                    continue
                try:
                    content = file_path.read_text(encoding='utf-8', errors='replace')
                    rel_path = file_path.relative_to(workspace_path).as_posix()
                    sources.append(f"=== TỆP TIN: `{rel_path}` ===\n{content}\n")
                except Exception:
                    pass
    except Exception as e:
        print(f"[Cảnh báo Debugger] Lỗi duyệt thư mục: {str(e)}")
        
    return "\n\n".join(sources)


def run_isolated_debugger_agent(workspace_path: str, user_query: str) -> str:
    """Khởi chạy một Agent gỡ lỗi độc lập ở chế độ One-shot để đưa ra giải pháp sửa đổi tối ưu."""
    from langchain_core.messages import SystemMessage, HumanMessage
    from config import model
    import re
    
    print("🤖 [Debugger Agent] Đang phân tích toàn bộ mã nguồn của dự án...")
    sources_data = crawl_project_sources(Path(workspace_path))
    
    if not sources_data:
        return "Không phát hiện mã nguồn hợp lệ hoặc thư mục dự án trống."
        
    prompt_template = load_prompt("isolated_debugger.txt")
    prompt = prompt_template.replace(
        "{sources_data}", sources_data
    ).replace(
        "{user_query}", user_query
    )
    
    try:
        response = model.invoke([
            SystemMessage(content="Bạn đang thực hiện nhiệm vụ gỡ lỗi cô lập, một chu kỳ suy nghĩ duy nhất (One-shot debug)."),
            HumanMessage(content=prompt)
        ])
        
        cleaned_content = re.sub(r"<thinking>.*?</thinking>", "", response.content, flags=re.DOTALL)
        cleaned_content = re.sub(r"<thought>.*?</thought>", "", cleaned_content, flags=re.DOTALL)
        return cleaned_content.strip()
    except Exception as e:
        return f"Lỗi trong quá trình chạy Debugger Agent độc lập: {str(e)}"


def isolated_debugger_node(state: AgentState) -> Dict[str, Any]:
    """
    Nút trung gian kiểm tra và chạy phân tích lỗi của dự án.
    Nếu là yêu cầu liên quan đến lỗi, chạy gỡ lỗi cô lập và đưa thẳng báo cáo vào luồng tin nhắn hội thoại.
    """
    ws = state.get("workspace_path", ".")
    messages = state.get("messages", [])
    debugger_proposal = state.get("debugger_proposal", "")

    # Đảm bảo không chạy lại nếu đề xuất đã tồn tại
    if debugger_proposal:
        return {}

    user_query = ""
    for msg in reversed(messages):
        if msg.type == "human" or isinstance(msg, HumanMessage):
            user_query = get_text_content_safely(msg.content)
            break

    # Phát hiện xem yêu cầu hiện tại có cần can thiệp gỡ lỗi hệ thống hay không
    bug_keywords = ["lỗi", "error", "bug", "crash", "exception", "failed", "sửa lỗi", "fix", "hỏng"]
    detailed_analysis = state.get("detailed_analysis", "")
    error_logs = state.get("error_logs", "")

    is_bug_fixing_task = (
        any(kw in user_query.lower() for kw in bug_keywords) or 
        any(kw in detailed_analysis.lower() for kw in bug_keywords) or
        bool(error_logs)
    )

    if not is_bug_fixing_task:
        return {"debugger_proposal": ""}

    proposal_result = run_isolated_debugger_agent(ws, user_query or detailed_analysis or error_logs)
    
    # Chuyển đổi phương án thành một AIMessage hội thoại tự nhiên, 
    # nó sẽ tự động được xếp cuối lịch sử hội thoại trước khi Executor chạy.
    proposal_msg = AIMessage(
        content=(
            "=== [BẢN PHÂN TÍCH & PHƯƠNG ÁN SỬA LỖI TỐI ƯU CỦA TÔI] ===\n"
            "Sau khi phân tích toàn bộ mã nguồn của dự án một cách độc lập, tôi đã xác định được nguyên nhân và đúc kết được phương án giải quyết tối ưu dưới đây:\n\n"
            f"{proposal_result}\n\n"
            "Bây giờ, tôi sẽ bắt đầu gọi các công cụ chỉnh sửa tệp tin thích hợp để áp dụng giải pháp này."
        )
    )

    return {
        "debugger_proposal": proposal_result,
        "messages": [proposal_msg]
    }
        
def executor_node(state: AgentState) -> Dict[str, Any]:
    ws = state.get("workspace_path", ".")
    plan = state.get("plan", [])
    error_logs = state.get("error_logs", "")
    file_registry = state.get("file_registry", {})
    messages = list(state["messages"])
    task_type = state.get("task_type", "development")
    extension_path = state.get("extension_path", "")
    active_skills = state.get("active_skills", {}) or {}
    debugger_proposal = state.get("debugger_proposal", "")

    state_updates = {}
    git_branch = state.get("git_branch", "")
    workspace_context = state.get("workspace_context", "")
    
    # Đồng bộ hóa cấu hình Git/Context nếu thiếu
    if not git_branch:
        git_dir = Path(ws) / ".git"
        if git_dir.exists():
            try:
                git_manager = GitManager(ws)
                git_branch = git_manager.init_and_prepare_branch()
            except Exception:
                git_branch = "no_git"
        else:
            git_branch = "no_git"
        state_updates["git_branch"] = git_branch

    if not workspace_context:
        context_path = Path(ws) / "CONTEXT.md"
        if context_path.exists():
            try:
                workspace_context = context_path.read_text(encoding="utf-8")
            except Exception:
                workspace_context = "Không thể đọc CONTEXT.md"
        else:
            workspace_context = "📋 Chưa có tệp cấu hình CONTEXT.md."
        state_updates["workspace_context"] = workspace_context

    if not extension_path:
        ext_dir = find_extension_dir_heuristic(Path(ws))
        if ext_dir:
            extension_path = ext_dir
            state_updates["extension_path"] = extension_path

    # Quản lý danh sách nhiệm vụ hợp lệ
    parsed_plan = []
    for t in plan:
        if isinstance(t, dict):
            parsed_plan.append(Task(**t))
        else:
            parsed_plan.append(t)

    eligible_tasks = get_eligible_tasks(parsed_plan)
    if not eligible_tasks:
        pending_tasks = [t for t in parsed_plan if t.status == "pending"]
        if pending_tasks:
            eligible_tasks = [pending_tasks[0]]
            
    tasks_str = ""
    if eligible_tasks:
        tasks_str = "\n".join([f"- [{t.id}] {t.description}" for t in eligible_tasks])
    else:
        tasks_str = "- [Khảo sát tổng thể]: Tìm hiểu cấu trúc và giải quyết yêu cầu người dùng."

    # Quản lý thư viện kỹ năng
    skills_engine = AgentSkillsEngine(ws)
    catalog = skills_engine.scan_catalog()
    
    catalog_prompt = ""
    if catalog:
        catalog_prompt = "\n=== 📚 THƯ VIỆN KỸ NĂNG KHẢ DỤNG (TIER 1: CATALOG) ===\n"
        for item in catalog:
            catalog_prompt += f"- **{item['name']}**: {item['description']}\n"

    # Đồng bộ hóa kích hoạt kỹ năng
    current_turn_tool_messages = []
    for msg in reversed(messages):
        if getattr(msg, "type", None) == "tool":
            current_turn_tool_messages.append(msg)
        else:
            break

    last_ai_message = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            last_ai_message = msg
            break

    if last_ai_message and last_ai_message.tool_calls:
        tool_results_map = {tm.tool_call_id: tm for tm in current_turn_tool_messages}
        for tc in last_ai_message.tool_calls:
            if tc["name"] == "activate_agent_skill":
                tc_id = tc["id"]
                associated_tool_msg = tool_results_map.get(tc_id)
                if associated_tool_msg and "Lỗi" not in str(associated_tool_msg.content):
                    requested_skill = tc["args"].get("skill_name")
                    if requested_skill:
                        active_skills[requested_skill] = str(associated_tool_msg.content)

    active_skills_prompt = ""
    if active_skills:
        active_skills_prompt = "\n=== ⚡ CÁC KỸ NĂNG ĐANG HOẠT ĐỘNG (TIER 2: ACTIVE STATUS) ===\n"
        for s_name in active_skills.keys():
            desc = ""
            if catalog:
                for item in catalog:
                    if item["name"] == s_name:
                        desc = item["description"]
                        break
            active_skills_prompt += f"- **{s_name}** (Trạng thái: Đã kích hoạt): {desc or 'Kích hoạt thành công. Hãy gọi activate_agent_skill để lấy tài liệu chi tiết.'}\n"

    # Định nghĩa các công cụ chạy
    activate_skill_tool = ActivateSkillTool(workspace_path=ws)
    run_skill_script_tool = RunSkillScriptTool(workspace_path=ws)
    search_keyword_tool = SearchKeywordTool(workspace_path=ws)
    complete_task_tool = CompleteTaskTool(workspace_path=ws)
    query_openwiki = QueryOpenWikiTool(workspace_path=ws)
    web_autonomous_executor = WebAutonomousExecutorTool(workspace_path=ws)
    chrome_debugger_tool = ChromeDebuggerTool(workspace_path=ws)

    mcp_tools = []
    active_mcp_servers = state.get("active_mcp_servers", [])
    try:
        from mcp_helper import MCPRegistryManager
        mcp_manager = MCPRegistryManager(ws)
        mcp_tools = mcp_manager.get_tools_sync(active_servers=active_mcp_servers)
    except Exception as e:
        print(f"[Dynamic MCP Warning] Không thể nạp công cụ MCP: {str(e)}")

    parsed_plan = []
    for t in plan:
        if isinstance(t, dict):
            parsed_plan.append(Task(**t))
        else:
            parsed_plan.append(t)

    has_pending_tasks = any(t.status == "pending" for t in parsed_plan)
    if not has_pending_tasks:
        return {
            "plan": parsed_plan,
            "messages": [AIMessage(content="🎉 **[Hệ thống tự động duyệt hoàn thành]**: Toàn bộ nhiệm vụ trong lộ trình đã hoàn thành thành công.")]
        }

    if task_type == "analysis":
        read_files = ReadFilesTool(workspace_path=ws)
        list_directory = ListDirectoryTool(workspace_path=ws)
        search_symbols = UniversalSymbolSearchTool(workspace_path=ws)
        read_file_lines = ReadFileLinesTool(workspace_path=ws)
        ask_questions_tool = AskQuestionsTool(workspace_path=ws)
        write_file = WriteFileTool(workspace_path=ws)
        apply_patch = ApplyPatchTool(workspace_path=ws)
        
        tools = [
            activate_skill_tool, run_skill_script_tool, web_autonomous_executor, chrome_debugger_tool,
            read_files, list_directory, search_symbols, read_file_lines, ask_questions_tool, 
            search_keyword_tool, write_file, apply_patch, complete_task_tool, query_openwiki
        ] + mcp_tools
        
        system_prompt_template = load_prompt("executor_analysis.txt")
        system_prompt = system_prompt_template.replace(
            "{tasks_str}", tasks_str
        ).replace(
            "{ws}", ws
        )
    else:
        read_files = ReadFilesTool(workspace_path=ws)
        write_file = WriteFileTool(workspace_path=ws)
        apply_patch = ApplyPatchTool(workspace_path=ws)
        search_symbols = UniversalSymbolSearchTool(workspace_path=ws)
        list_directory = ListDirectoryTool(workspace_path=ws)
        run_terminal_command = RunTerminalTool(workspace_path=ws)
        read_file_lines = ReadFileLinesTool(workspace_path=ws)
        write_and_run_script = WriteAndRunScriptTool(workspace_path=ws)
        ask_questions_tool = AskQuestionsTool(workspace_path=ws)
       
        tools = [
            activate_skill_tool, run_skill_script_tool, read_files, write_file, web_autonomous_executor,
            apply_patch, list_directory, run_terminal_command, search_symbols, chrome_debugger_tool,
            read_file_lines, ask_questions_tool, write_and_run_script, search_keyword_tool, complete_task_tool, query_openwiki
        ] + mcp_tools
        
        system_prompt_template = load_prompt("executor_development.txt")
        system_prompt = system_prompt_template.replace(
            "{tasks_str}", tasks_str
        ).replace(
            "{ws}", ws
        )

    updated_file_registry = dict(file_registry or {})
    cache_updated = False

    for f_path_key in list(updated_file_registry.keys()):
        try:
            safe_path = sanitize_and_resolve_path(ws, f_path_key, create_parent=False)
            if safe_path.exists() and safe_path.is_file():
                disk_content = safe_path.read_text(encoding="utf-8")
                if updated_file_registry[f_path_key] != disk_content:
                    updated_file_registry[f_path_key] = disk_content
                    cache_updated = True
            else:
                updated_file_registry.pop(f_path_key, None)
                cache_updated = True
        except Exception:
            pass

    if cache_updated:
        state_updates["file_registry"] = updated_file_registry
        file_registry = updated_file_registry

    registry_context_str = ""
    if file_registry:
        registry_context_str = "\n=== 📦 CÁC FILE ĐÃ NẠP VÀO BỘ NHỚ (ĐÃ ĐỒNG BỘ VỚI ĐĨA) ===\n"
        for file_path, content in file_registry.items():
            lang = get_markdown_language(file_path)
            lines = content.splitlines()
            formatted_lines = [f"{idx+1:04d} | {line}" for idx, line in enumerate(lines)]
            registry_context_str += (
                f"\n--- TỆP TIN: `{file_path}` ---\n"
                f"```{lang}\n" + "\n".join(formatted_lines) + "\n```\n"
            )

    system_instructions = system_prompt + catalog_prompt + active_skills_prompt
    if git_branch and git_branch != "no_git":
        system_instructions += f"\n- Nhánh Git đang hoạt động: `{git_branch}`"

    model_with_tools = model.bind_tools(tools)

    # 🌟 CẢI TIẾN QUAN TRỌNG: Gọi hàm thu gọn tin nhắn file cũ để làm sạch lịch sử trượt
    optimized_history = compact_historical_file_messages(messages)

    # Khởi tạo bối cảnh hệ thống sạch sẽ, không nhiễm độc System Prompt
    input_messages = [SystemMessage(content=system_instructions)]

    if workspace_context:
        context_body = "=== NGỮ CẢNH DỰ ÁN HIỆN HÀNH (WORKSPACE STATE) ===\n"
        context_body += f"\n--- THÔNG TIN NỀN TẢNG (CONTEXT.md) ---\n{workspace_context}\n"
        input_messages.append(SystemMessage(content=context_body))

    if task_type == "development" and error_logs:
        input_messages.append(HumanMessage(content=f"LƯU Ý SỬA LỖI TỪ VÒNG KIỂM THỬ:\n{error_logs}\nHãy sửa triệt để."))
        
    response = model_with_tools.invoke(input_messages + optimized_history)
    response = sanitize_llm_response_content(response)
    
    if not response.tool_calls:
        if task_type == "analysis":
            findings = []
            if response.content:
                findings = [f"### Báo cáo khảo sát chủ động:\n{response.content}"]
                
            updated_plan = []
            eligible_ids = {t.id for t in eligible_tasks}
            for t in parsed_plan:
                t_copy = t.model_copy()
                if t_copy.id in eligible_ids:
                    t_copy.status = "completed"
                updated_plan.append(t_copy)

            state_updates.update({
                "messages": [response],
                "plan": updated_plan,
                "last_executed_task_ids": list(eligible_ids),
                "active_skills": active_skills,
                "debugger_proposal": debugger_proposal
            })
            if findings:
                state_updates["step_findings"] = findings
            return state_updates
        else:
            has_executed_action = any(
                getattr(msg, "type", None) == "tool" and msg.name in ["write_file", "apply_search_replace_patch", "run_terminal_command", "run_skill_script", "complete_agent_task"]
                for msg in reversed(messages)
            )
            content_lower = response.content.lower() if response.content else ""
            explicitly_finished = any(kw in content_lower for kw in ["hoàn thành", "hoàn tất", "done", "finished", "đã hoàn thành"])

            if has_executed_action or explicitly_finished:
                updated_plan = []
                eligible_ids = {t.id for t in eligible_tasks}
                for t in parsed_plan:
                    t_copy = t.model_copy()
                    if t_copy.id in eligible_ids:
                        t_copy.status = "completed"
                    updated_plan.append(t_copy)

                state_updates.update({
                    "messages": [response],
                    "plan": updated_plan,
                    "last_executed_task_ids": list(eligible_ids),
                    "active_skills": active_skills,
                    "debugger_proposal": debugger_proposal
                })
                return state_updates
            else:
                warning_feedback = HumanMessage(
                    content="⚠️ Cảnh báo: Bạn chưa gọi công cụ 'complete_agent_task' để hoàn thành nhiệm vụ này. Hãy gọi công cụ đó để kết thúc công việc."
                )
                state_updates.update({
                    "messages": [response, warning_feedback],
                    "active_skills": active_skills,
                    "debugger_proposal": debugger_proposal
                })
                return state_updates
                
    state_updates.update({
        "messages": [response],
        "active_skills": active_skills,
        "debugger_proposal": debugger_proposal
    })
    return state_updates

def replanner_node(state: AgentState) -> Dict[str, Any]:
    replanning_count = state.get("replanning_count", 0)
    ws = state["workspace_path"]
    plan = state["plan"]
    messages = state["messages"]
    task_type = state.get("task_type", "development")
    workspace_context = state.get("workspace_context", "")
    error_logs = state.get("error_logs", "")
    active_skills = dict(state.get("active_skills", {}))
    
    active_skills_prompt = ""
    if active_skills:
        active_skills_prompt = "\n=== ⚡ CÁC KỸ NĂNG ĐANG HOẠT ĐỘNG (INSTRUCTIONS) ===\n"
        for s_name, s_instructions in active_skills.items():
            active_skills_prompt += f"\n--- CHỒNG CHỈ DẪN KỸ NĂNG `{s_name}` ---\n{s_instructions}\n"

    is_survey_transition = (
        len(plan) == 1 and 
        (plan[0].id if isinstance(plan[0], Task) else plan[0].get("id")) == "T_SURVEY" and
        (plan[0].status if isinstance(plan[0], Task) else plan[0].get("status")) == "completed"
    )
    
    if replanning_count >= 5 or (not error_logs and not is_survey_transition):
        action_msg = "bypass_limit" if replanning_count >= 5 else "bypass_no_error"
        proposal_message = AIMessage(
            content=json.dumps({"action": action_msg}, ensure_ascii=False),
            name="replanner_proposal"
        )
        return {
            "messages": [proposal_message]
        }
    
    plan_str = "\n".join([
        f"- [{t.id if isinstance(t, Task) else t.get('id')}] {t.description if isinstance(t, Task) else t.get('description')} "
        f"(Trạng thái: {t.status if isinstance(t, Task) else t.get('status')}, "
        f"Phụ thuộc: {t.dependencies if isinstance(t, Task) else t.get('dependencies')})"
        for t in plan
    ])
    
    if is_survey_transition:
        # 🌟 CẢI TIẾN: Nạp từ file replanner_survey.txt
        system_prompt_template = load_prompt("replanner_survey.txt")
        system_prompt = system_prompt_template.replace("{ws}", ws)
    else:
        # 🌟 CẢI TIẾN: Nạp từ file replanner_bug.txt
        system_prompt_template = load_prompt("replanner_bug.txt")
        system_prompt = system_prompt_template.replace("{ws}", ws)
    system_prompt += active_skills_prompt 
    if workspace_context:
        system_prompt += f"\n\n--- NGỮ CẢNH HỆ THỐNG (CONTEXT.md) ---\n{workspace_context}"
        
    user_prompt = f"Kế hoạch hiện tại:\n{plan_str}\n\n"
    if error_logs:
        user_prompt += f"🚨 LỖI BIÊN DỊCH CẦN SỬA ĐỔI KẾ HOẠCH:\n{error_logs}\n\n"
    user_prompt += "Hãy đưa ra phân tích và đề xuất cập nhật kế hoạch phù hợp thông qua cuộc gọi hàm."
    
    structured_llm = model.with_structured_output(PlanUpdate, method="function_calling")
    
    try:
        decision = structured_llm.invoke([
            {"role": "system", "content": system_prompt},
            *messages,
            {"role": "user", "content": user_prompt}
        ])
        
        should_modify = getattr(decision, "should_modify_plan", False)
        explanation = getattr(decision, "explanation", "")
        updated_tasks = getattr(decision, "updated_tasks", [])
        updated_task_type = getattr(decision, "task_type", task_type)
        
        if not updated_tasks or not should_modify:
            proposal_message = AIMessage(
                content="📋 [Hệ thống tự động duyệt qua: Kế hoạch hiện tại đã tối ưu, không cần cập nhật thêm]",
                name="replanner_proposal",
                additional_kwargs={"proposal_payload": {"action": "bypass_no_error", "tasks": []}}
            )
            return {"messages": [proposal_message],"active_skills": active_skills }

        refined_tasks = []
        for task_data in updated_tasks:
            task_obj = task_data if isinstance(task_data, Task) else Task(**task_data)
            refined_tasks.append(task_obj)
            
        proposal_data = {
            "action": "propose",
            "explanation": explanation,
            "task_type": updated_task_type,
            "tasks": [t.model_dump() for t in refined_tasks]
        }
        
    except Exception as e:
        proposal_data = {
            "action": "bypass_no_error",
            "explanation": f"Kích hoạt cơ chế tự phục hồi do lỗi hệ thống: {str(e)}",
            "task_type": task_type,
            "tasks": []
        }
        
    proposal_message = AIMessage(
        content=f"🔄 **[Hệ thống đề xuất lộ trình hành động mới]**\n\n{proposal_data.get('explanation', 'Đang cập nhật lộ trình...')}",
        name="replanner_proposal",
        additional_kwargs={"proposal_payload": proposal_data}
    )
    
    return {
        "replanning_count": replanning_count + 1,
        "messages": [proposal_message],
        "active_skills": active_skills 
    }

def replanner_interrupt_node(state: AgentState) -> Dict[str, Any]:
    messages = state["messages"]
    plan = state["plan"]
    
    proposal_payload = {}
    for msg in reversed(messages):
        if getattr(msg, "name", None) == "replanner_proposal":
            proposal_payload = msg.additional_kwargs.get("proposal_payload", {})
            break
            
    if not proposal_payload:
        logger.warning("[Replanner Gate] Không tìm thấy dữ liệu đề xuất từ replanner_node.")
        return {
            "error_logs": "",
            "attempts": 0,
            "modified_files": []
        }
        
    action = proposal_payload.get("action", "bypass_no_error")
    proposed_tasks = proposal_payload.get("tasks", [])
    
    if action in ["bypass_limit", "bypass_no_error"] or not proposed_tasks:
        fallback_tasks = []
        for t in plan:
            fallback_tasks.append(t if isinstance(t, Task) else Task(**t))
            
        logger.info("[Replanner Gate] Tự động duyệt qua kế hoạch rỗng hoặc lệnh bypass.")
        return {
            "plan": fallback_tasks,
            "error_logs": "",
            "attempts": 0,
            "modified_files": [],
            "messages": [AIMessage(content="⏭️ **[Tự động điều phối]**: Kế hoạch hiện tại đã tối ưu, hệ thống tự động hoàn tất pha duyệt.")]
        }
        
    interrupt_payload = {
        "type": "replanner_interrupt",
        "title": "📋 ĐÁNH GIÁ & PHÊ DUYỆT KẾ HOẠCH HÀNH ĐỘNG",
        "explanation": proposal_payload.get("explanation", "Hệ thống phát hiện cần thay đổi lộ trình để tiếp tục thực hiện."),
        "proposed_tasks": proposed_tasks,
        "prompt": (
            "Hệ thống đề xuất điều chỉnh lộ trình như trên.\n"
            "- Nhấn Approve (hoặc gửi phản hồi 'yes', chuỗi rỗng) để ĐỒNG Ý lộ trình mới.\n"
            "- Gửi phản hồi 'skip' hoặc 'no' để BỎ QUA và giữ nguyên lộ trình cũ.\n"
            "- Bạn cũng có thể sửa đổi danh sách Task trực tiếp trên giao diện để cấu hình kế hoạch tùy chỉnh."
        )
    }
    
    user_input = interrupt(interrupt_payload)
    
    user_input_clean = ""
    if isinstance(user_input, str):
        user_input_clean = user_input.strip().lower()
    
    if user_input is None or user_input_clean in ["", "yes", "approve", "ok"]:
        refined_tasks = [Task(**t) for t in proposed_tasks]
        return {
            "plan": refined_tasks,
            "task_type": proposal_payload.get("task_type", "development"),
            "error_logs": "",
            "attempts": 0,
            "modified_files": [],
            "messages": [AIMessage(content="✅ **[Kế hoạch được duyệt]** Áp dụng lộ trình phát triển và kiểm thử mới thành công.")]
        }
        
    if user_input_clean in ["skip", "no", "cancel", "decline"]:
        fallback_tasks = []
        for t in plan:
            task_obj = t if isinstance(t, Task) else Task(**t)
            fallback_tasks.append(task_obj)
            
        # Đảm bảo có ít nhất một task pending để hệ thống không bị dừng đột ngột
        has_pending = any(t.status == "pending" for t in fallback_tasks)
        if not has_pending and fallback_tasks:
            fallback_tasks[-1].status = "pending"
            
        logger.info("[Replanner Gate] Người dùng từ chối kế hoạch mới. Sử dụng kế hoạch cũ.")
        return {
            "plan": fallback_tasks,
            "error_logs": "",
            "attempts": 0,
            "modified_files": [],
            "messages": [AIMessage(content="⏭️ **[Người dùng bỏ qua kế hoạch mới]** Tiếp tục lộ trình thực thi cũ.")]
        }
        
    custom_tasks_raw = []
    
    if isinstance(user_input, list):
        custom_tasks_raw = user_input
    elif isinstance(user_input, dict) and "tasks" in user_input:
        custom_tasks_raw = user_input["tasks"]
    elif isinstance(user_input, str):
        try:
            parsed = json.loads(user_input)
            if isinstance(parsed, list):
                custom_tasks_raw = parsed
            elif isinstance(parsed, dict) and "tasks" in parsed:
                custom_tasks_raw = parsed["tasks"]
        except json.JSONDecodeError:
            pass

    if custom_tasks_raw:
        try:
            custom_tasks = [Task(**t) for t in custom_tasks_raw]
            logger.info(f"[Replanner Gate] Áp dụng thành công kế hoạch tùy chỉnh gồm {len(custom_tasks)} tasks.")
            return {
                "plan": custom_tasks,
                "task_type": proposal_payload.get("task_type", "development"),
                "error_logs": "",
                "attempts": 0,
                "modified_files": [],
                "messages": [AIMessage(content=f"✏️ **[Kế hoạch tùy chỉnh]** Đã áp dụng lộ trình gồm {len(custom_tasks)} bước do bạn thiết lập.")]
            }
        except Exception as parse_err:
            logger.error(f"[Replanner Gate] Lỗi định dạng dữ liệu Task tùy chỉnh: {str(parse_err)}")
            
    logger.info(f"[Replanner Gate] Ghi nhận phản hồi văn bản tự do: {user_input}")
    feedback_message = HumanMessage(
        content=(
            "⚠️ Ý kiến điều chỉnh lộ trình từ người dùng:\n"
            f"'{user_input}'\n"
            "Hãy phân tích và cập nhật lại kế hoạch hành động tương ứng dựa trên ý kiến này."
        )
    )
    
    fallback_tasks = [t if isinstance(t, Task) else Task(**t) for t in plan]
    return {
        "plan": fallback_tasks,
        "error_logs": f"Người dùng yêu cầu thay đổi lộ trình: {user_input}",
        "attempts": 0,
        "messages": [feedback_message]
    }

def tool_node(state: AgentState) -> Dict[str, Any]:
    ws = state["workspace_path"]
    read_files = ReadFilesTool(workspace_path=ws)
    write_file = WriteFileTool(workspace_path=ws)
    apply_patch = ApplyPatchTool(workspace_path=ws)
    list_directory = ListDirectoryTool(workspace_path=ws)
    run_terminal_command = RunTerminalTool(workspace_path=ws)
    search_symbols = UniversalSymbolSearchTool(workspace_path=ws)
    read_file_lines = ReadFileLinesTool(workspace_path=ws)
    web_interact_tool = WebInteractAndTestTool(workspace_path=ws)
    ask_questions_tool = AskQuestionsTool(workspace_path=ws)
    write_and_run_script = WriteAndRunScriptTool(workspace_path=ws)
    search_keyword_tool = SearchKeywordTool(workspace_path=ws)
    complete_task_tool = CompleteTaskTool(workspace_path=ws) 
    query_openwiki = QueryOpenWikiTool(workspace_path=ws)
    web_autonomous_executor = WebAutonomousExecutorTool(workspace_path=ws) 
    chrome_debugger_tool = ChromeDebuggerTool(workspace_path=ws)
    
    tools_map = {
        "read_files": read_files,
        "write_file": write_file,
        "apply_search_replace_patch": apply_patch,
        "list_directory": list_directory,
        "run_terminal_command": run_terminal_command,
        "search_symbols_universal": search_symbols,
        "read_file_lines": read_file_lines,
        "web_interact_and_test": web_interact_tool,
        "ask_questions_if_underspecified": ask_questions_tool,
        "activate_agent_skill": ActivateSkillTool(workspace_path=ws),
        "run_skill_script": RunSkillScriptTool(workspace_path=ws),
        "write_and_run_script": write_and_run_script,
        "search_keyword": search_keyword_tool,
        "query_global_openwiki": query_openwiki,
        "complete_agent_task": complete_task_tool,
        "web_autonomous_executor": web_autonomous_executor,
        "chrome_devtools_debugger": chrome_debugger_tool,
    }

    try:
        from mcp_helper import MCPRegistryManager
        mcp_manager = MCPRegistryManager(ws)
        mcp_tools = mcp_manager.get_tools_sync()
        for mt in mcp_tools:
            tools_map[mt.name] = mt
    except Exception as e:
        print(f"[Dynamic MCP Warning] Lỗi đồng bộ hóa công cụ tại Tool Node: {str(e)}")
    
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {}
        
    tool_messages = []
    modified_files = list(state.get("modified_files", []))
    file_registry = dict(state.get("file_registry", {}))
    impacted_files = set()
    completed_task_ids = []
    
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"] or {}
        tool_id = tool_call["id"]
        
        if tool_name == "complete_agent_task":
            t_id = tool_args.get("task_id")
            if t_id:
                completed_task_ids.append(t_id)
        
        raw_path = tool_args.get("file_path") or tool_args.get("file_paths")
        if tool_name in ["write_file", "apply_search_replace_patch", "read_files", "read_file_lines"]:
            if raw_path:
                if isinstance(raw_path, list):
                    impacted_files.update(raw_path)
                else:
                    impacted_files.add(str(raw_path))
        
        tool_instance = tools_map.get(tool_name)
        if not tool_instance:
            result = f"Lỗi: Không tìm thấy công cụ '{tool_name}'."
        else:
            try:
                sync_tools = {
                    "read_files", "write_file", "apply_search_replace_patch", "list_directory", 
                    "run_terminal_command", "search_symbols_universal", "read_file_lines", 
                    "web_interact_and_test", "ask_questions_if_underspecified", "activate_agent_skill", 
                    "run_skill_script", "write_and_run_script", "search_keyword", "query_global_openwiki", 
                    "complete_agent_task", "web_autonomous_executor", "chrome_devtools_debugger"
                }
                
                if tool_name not in sync_tools and hasattr(tool_instance, "ainvoke"):
                    from mcp_helper import run_sync
                    result = run_sync(tool_instance.ainvoke(tool_args))
                else:
                    result = tool_instance.invoke(tool_args)
                    
                # 🌟 CẢI TIẾN QUAN TRỌNG: Làm giàu phản hồi của write_file và apply_patch bằng nội dung file thực tế kèm dòng
                if tool_name in ["write_file", "apply_search_replace_patch"] and "Lỗi" not in str(result):
                    if raw_path and not isinstance(raw_path, list):
                        try:
                            safe_path = sanitize_and_resolve_path(ws, raw_path, create_parent=True)
                            if str(safe_path) not in modified_files:
                                modified_files.append(str(safe_path))
                                
                            # Đọc ngược dữ liệu vừa ghi từ đĩa để hiển thị chi tiết trong ToolMessage
                            if safe_path.exists() and safe_path.is_file():
                                updated_content = safe_path.read_text(encoding="utf-8")
                                lang = get_markdown_language(str(raw_path))
                                lines = updated_content.splitlines()
                                formatted_lines = [f"{idx+1:04d} | {line}" for idx, line in enumerate(lines)]
                                
                                # Đè nội dung thô vào biến kết quả
                                result = (
                                    f"✅ [Ghi nhận chỉnh sửa thành công] {result}\n\n"
                                    f"=== TỆP TIN: `{raw_path}` (Nội dung mới nhất sau khi chỉnh sửa) ===\n"
                                    f"```{lang}\n" + "\n".join(formatted_lines) + "\n```"
                                )
                        except Exception as write_err:
                            result = f"{result} (Cảnh báo: không thể lấy mã nguồn sau chỉnh sửa: {str(write_err)})"
            except Exception as e:
                result = f"Lỗi thực thi công cụ '{tool_name}': {str(e)}"
                
        sanitized_result = sanitize_tool_result_content(tool_name, result, ws)

        # 🌟 CẢI TIẾN QUAN TRỌNG: Loại bỏ việc nén stubs rác tại read_files!
        # Cho phép kết quả thô của read_files và read_file_lines đi thẳng vào ToolMessage
        tool_messages.append(ToolMessage(content=str(sanitized_result), name=tool_name, tool_call_id=tool_id))
        
    # Đồng bộ hóa bộ đệm file registry trên đĩa
    BINARY_EXTENSIONS = {".xlsx", ".xls", ".png", ".jpg", ".jpeg", ".zip", ".pdf", ".exe"}
    for file_path in impacted_files:
        try:
            safe_path = sanitize_and_resolve_path(ws, file_path, create_parent=False)
            if safe_path.exists() and safe_path.is_file():
                if safe_path.suffix.lower() in BINARY_EXTENSIONS:
                    continue
                current_content = safe_path.read_text(encoding="utf-8")
                workspace_root = Path(ws).expanduser().resolve()
                rel_path = str(safe_path.relative_to(workspace_root))
                file_registry[rel_path] = current_content
        except Exception:
            pass
            
    updated_plan = list(state.get("plan", []))
    if completed_task_ids:
        parsed_plan = []
        for t in updated_plan:
            if isinstance(t, dict):
                parsed_plan.append(Task(**t))
            else:
                parsed_plan.append(t)
                
        for t in parsed_plan:
            if t.id in completed_task_ids:
                t.status = "completed"
        updated_plan = parsed_plan

    return {
        "messages": tool_messages,
        "modified_files": modified_files,
        "file_registry": file_registry,
        "plan": updated_plan
    }

def human_interaction_gate_node(state: AgentState) -> Dict[str, Any]:
    messages = state["messages"]
    
    target_tool_msg = None
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage) and msg.name == "ask_questions_if_underspecified":
            target_tool_msg = msg
            break
            
    if not target_tool_msg:
        return {}
        
    try:
        data = json.loads(target_tool_msg.content)
        if not isinstance(data, dict) or data.get("status") != "requires_human_response":
            return {}
        payload = data.get("payload", {})
    except Exception:
        return {}

    user_input = interrupt(payload)
    
    feedback_content = f"### [Phản hồi của người dùng cho các câu hỏi]:\n{json.dumps(user_input, ensure_ascii=False)}"
    feedback_message = HumanMessage(
        content=feedback_content,
        name="human_interaction_feedback"
    )
    
    return {
        "messages": [feedback_message]
    }



def synthesis_node(state: AgentState) -> Dict[str, Any]:
    """
    [CẬP NHẬT KIỂM SOÁT MIỀN TÀI LIỆU - DOMAIN GATE PROTECTION]
    Nút tổng hợp tài liệu CONTEXT.md có cổng bảo vệ để tránh sinh tài liệu rác gây ảo giác.
    """
    ws = state["workspace_path"]
    findings = state.get("step_findings", [])
    git_branch = state.get("git_branch", "no_git")
    
    if not findings:
        return {"messages": [AIMessage(content="Không thu thập được thông tin khảo sát để tổng hợp.")]}
        
    compiled_data = "\n\n---\n\n".join(findings)
    
    # 🌟 CỔNG BẢO VỆ CHỐNG ẢO GIÁC: Kiểm tra xem có phải tác vụ phi lập trình (Notion/Ops) hoặc sửa lỗi cục bộ không
    is_non_coding = any(
        kw in compiled_data.lower() or kw in str(state.get("messages", [])).lower()
        for kw in ["notion", "figma", "jira", "trello", "google sheet", "excel", "devops", "ops", "tối ưu"]
    )
    is_localized_edit = any(
        kw in compiled_data.lower() or "localized code edit" in compiled_data.lower()
        for kw in ["sửa lỗi cục bộ", "chỉnh sửa file", "localized"]
    )
    
    if is_non_coding:
        return {
            "messages": [AIMessage(content="⏭️ **[Bypass Synthesis]**: Phát hiện tác vụ Phi lập trình / Vận hành (Ops). Hệ thống tự động bỏ qua bước thiết lập `CONTEXT.md` để bảo vệ bối cảnh đồ thị sạch.")]
        }
        
    if is_localized_edit and not (Path(ws) / "CONTEXT.md").exists():
        return {
            "messages": [AIMessage(content="⏭️ **[Bypass Synthesis]**: Phát hiện tác vụ chỉnh sửa tệp tin cục bộ. Bỏ qua việc tạo tệp `CONTEXT.md` mới để tối ưu bộ nhớ.")]
        }

    synthesis_prompt = load_prompt("synthesis.txt")
    
    try:
        response = fast_model.invoke([
            SystemMessage(content=synthesis_prompt),
            HumanMessage(content=f"Thông tin thu thập:\n\n{compiled_data}")
        ])
        
        md_content = response.content
        if isinstance(md_content, str) and md_content.strip():
            cleaned_md = md_content.strip()
            if cleaned_md.startswith("```markdown"):
                cleaned_md = cleaned_md[11:]
            elif cleaned_md.startswith("```"):
                cleaned_md = cleaned_md[3:]
            if cleaned_md.endswith("```"):
                cleaned_md = cleaned_md[:-3]
            cleaned_md = cleaned_md.strip()
            
            if git_branch != "no_git":
                tools_mgr = WorkspaceTools(ws)
                tools_mgr.write_file("CONTEXT.md", cleaned_md)
                
                git_manager = GitManager(ws)
                git_manager._run_cmd(["git", "add", "CONTEXT.md"], ignore_error=True)
                
                message_content = (
                    "**Tổng hợp tài liệu hoàn tất:** Đã biên dịch tri thức khảo sát, "
                    "lưu vật lý thành tệp `CONTEXT.md` và đưa vào Git staging thành công."
                )
            else:
                message_content = (
                    "**Tổng hợp tài liệu hoàn tất (Chế độ In-Memory):** Tri thức khảo sát "
                    "đã được tổng hợp và nạp trực tiếp vào ngữ cảnh trạng thái đồ thị (`workspace_context`). "
                    "Tệp tin `CONTEXT.md` vật lý **không** được tạo trên ổ đĩa do hệ thống phát hiện không sử dụng Git."
                )
            
            return {
                "workspace_context": cleaned_md,
                "messages": [AIMessage(content=message_content)]
            }
    except Exception as e:
        return {"messages": [AIMessage(content=f"Cảnh báo: Có lỗi xảy ra khi tổng hợp tệp CONTEXT.md: {str(e)}")]}
    return {}

def commit_node(state: AgentState) -> Dict[str, Any]:
    ws = state["workspace_path"]
    task_type = state.get("task_type", "development")
    git_branch = state.get("git_branch", "no_git")
    
    if git_branch == "no_git":
        return {
            "messages": [AIMessage(content="Đã hoàn thành toàn bộ yêu cầu của bạn. Chế độ Không-Git được kích hoạt, bỏ qua commit.")]
        }
        
    git_manager = GitManager(ws)
    status = git_manager._run_cmd(["git", "status", "--porcelain"], ignore_error=True)
    
    plan = state["plan"]
    plan_steps_str = "\n".join([
        f"- [{getattr(t, 'id', None) or t.get('id')}] {getattr(t, 'description', None) or t.get('description')} ({getattr(t, 'status', None) or t.get('status')})"
        for t in plan
    ])
    
    if status and not status.startswith("ERROR"):
        commit_msg = f"feat(ai): automatic execution ({task_type}) \n\nSteps:\n{plan_steps_str}"
        git_manager.commit_changes(commit_msg)
        msg = f"Đã hoàn thành yêu cầu và commit các thay đổi lên nhánh `{git_branch}`."
    else:
        msg = "Đã hoàn thành quy trình công việc. Không có thay đổi tệp tin vật lý nào cần commit lên Git."
        
    return {"messages": [AIMessage(content=msg)]}