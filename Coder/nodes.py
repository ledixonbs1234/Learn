import asyncio
import json
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
from mcp_helper import run_agent_with_devtools_mcp
from skills_engine import AgentSkillsEngine
from state import AgentState, PlanUpdate, RuntimeVerificationResult, TaskTriage, Task
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

    system_prompt = (
        "Bạn là một điều phối viên Agent thông minh cấp cao (Triage Supervisor).\n"
        "Nhiệm vụ của bạn là phân tích yêu cầu mới của người dùng để phân loại chính xác hướng xử lý.\n\n"
        "Bạn đã được cung cấp Lịch sử hội thoại gần nhất và Thông tin phiên hoạt động hiện tại.\n"
        "Hãy tận dụng thông tin này để giải quyết các đại từ mơ hồ.\n\n"
        "⚠️ QUY TẮC ĐÁNH GIÁ SỰ TRÔI LỆCH PHIÊN VÀ PHÂN LOẠI (BẮT BUỘC):\n"
        "1. KIỂM TRA SỰ TIẾP NỐI (Follow-up Check)...\n"
        "   - Nếu yêu cầu mới là một câu hỏi hỏi thêm, yêu cầu giải thích, hoặc yêu cầu chỉnh sửa dựa trên dự án "
        "     đang mở trong phiên hoạt động hiện tại -> Đây là một câu hỏi TIẾP NỐI (Follow-up).\n"
        "   - Đối với câu hỏi tiếp nối, bạn KHÔNG ĐƯỢC chọn task_type = 'clarify' (yêu cầu hỏi lại path). Hãy thiết lập "
        "     task_type dựa trên bản chất yêu cầu ('analysis' nếu chỉ hỏi đáp giải thích, 'development' nếu yêu cầu sửa code).\n"
        "2. KIỂM TRA YÊU CẦU ĐỘC LẬP MỚI (Context Shift):\n"
        "   - Nếu người dùng đột ngột yêu cầu làm một việc hoàn toàn mới không liên quan đến thư mục hiện hành "
        "     (ví dụ: đang quét desktop lại yêu cầu 'sửa lỗi app ở thư mục D:/project-abc'), hoặc yêu cầu tạo mới app "
        "     nhưng không nói ở đâu -> Đặt task_type = 'clarify' để hệ thống hỏi lại đường dẫn mới.\n\n"
        "⚠️ ÁNH XẠ KỸ NĂNG CHỦ ĐỘNG (TÙY CHỌN - KHÔNG BẮT BUỘC):\n"
        "Dưới đây là danh sách các Kỹ năng kỹ thuật (Skills) khả dụng có sẵn trong hệ thống.\n"
        "Nhiệm vụ của bạn là đối chiếu yêu cầu hiện tại của người dùng với mô tả và điều kiện kích hoạt (triggers) của từng Kỹ năng dưới đây.\n"
        "Nếu yêu cầu của người dùng thực sự khớp với mục đích của một kỹ năng nào đó, bạn hãy điền tên kỹ năng đó vào trường 'recommended_skills'. Nếu không có kỹ năng nào thực sự khớp hoặc không cần thiết, bạn hoàn toàn có thể để trống trường này (mảng rỗng []).\n"
        f"{catalog_summary}\n\n"
        "⚠️ LỰA CHỌN MÁY CHỦ MCP PHÙ HỢP (BẮT BUỘC - TIẾT KIỆM TOKEN):\n"
        "Dưới đây là danh sách các Máy chủ MCP ngoài hiện hành đang được tích hợp vào dự án của bạn.\n"
        "Hãy đối chiếu mục tiêu công việc của người dùng. Nếu tác vụ đòi hỏi các công cụ từ máy chủ nào, bạn hãy đề xuất máy chủ đó "
        "bằng cách ghi chính xác tên định danh máy chủ vào mảng 'recommended_mcp_servers' (ví dụ: ['devtools'] hoặc ['notion']). "
        "Tuyệt đối KHÔNG đề xuất các máy chủ không liên quan để tránh làm tràn ngập bối cảnh bằng các công cụ thừa.\n"
        f"{mcp_summary}"
    )

    structured_llm = model.with_structured_output(TaskTriage, method="function_calling")
    
    triage_output = structured_llm.invoke([
        SystemMessage(content=system_prompt),
        SystemMessage(content=active_session_context),
        *context_messages,
        HumanMessage(content=f"Yêu cầu hiện tại của người dùng: {user_query_text}")
    ])
    
    return triage_output

def detect_and_triage_node(state: AgentState) -> Dict[str, Any]:
    messages = state["messages"]
    user_msg = messages[-1]
    user_query_text = get_text_content_safely(user_msg.content)
    
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

    return {
        "workspace_path": final_workspace,
        "plan": plan,
        "task_type": task_type,
        "is_simple": is_simple,
        "detailed_analysis": detailed_analysis,
        "messages": [AIMessage(content=triage_info_msg)],
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

    adversarial_prompt = (
        "Bạn là một kiểm toán viên mã nguồn đối kháng chuyên nghiệp (Adversarial Reviewer).\n"
        "Nhiệm vụ của bạn là rà soát đoạn mã nguồn dưới đây và tìm ra ít nhất 3 điểm yếu kỹ thuật, "
        "các giả định sai lầm, các trường hợp biên chưa được xử lý, hoặc nguy cơ bảo mật tiềm ẩn.\n\n"
        "⚠️ YÊU CẦU NGHIÊM NGẶT:\n"
        "- Chỉ tập trung chỉ ra lỗi logic thực tế, lỗ hổng cấu trúc hoặc rủi ro runtime.\n"
        "- TUYỆT ĐỐI KHÔNG khen ngợi, không viết tóm tắt vô nghĩa.\n"
        "- Định dạng câu trả lời bằng tiếng Việt, rõ ràng theo từng đầu dòng kèm chỉ dẫn file:dòng cụ thể.\n\n"
        f"ARTIFACT MÃ NGUỒN CẦN THẨM ĐỊNH ({latest_file}):\n"
        "```\n"
        f"{artifact_code}\n"
        "```"
    )

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
    
    system_prompt = (
        "Bạn là một điều phối viên Agent thông minh cấp cao (Triage Supervisor).\n"
        "Nhiệm vụ của bạn là phân tích yêu cầu của người dùng để xác định xem yêu cầu đó nên được xử lý qua luồng Fast-Track (Đơn giản) hay luồng Lập kế hoạch (Phức tạp).\n\n"
        "Đồng thời, hãy viết một bản phân tích chi tiết vào thuộc tính 'detailed_analysis' để định hướng cho các Agent ở các bước sau. Bản phân tích này cần làm rõ:\n"
        "1. Mục tiêu cốt lõi cuối cùng người dùng muốn đạt được.\n"
        "2. Các tệp tin, thư mục hoặc phân hệ mã nguồn cụ thể có thể sẽ bị tác động hoặc cần đọc/sửa đổi.\n"
        "3. Các ràng buộc về logic kỹ thuật, ngôn ngữ trình bày, hoặc các trường hợp biên cần lưu ý.\n"
        "4. Gợi ý sơ bộ về phương pháp thực hiện tối ưu.\n\n"
        "Hãy dựa trên các tiêu chí phân loại nghiêm ngặt sau để phân loại:\n\n"
        "1. KIỂM TRA TÁC VỤ ĐƠN GIẢN (is_simple = True):\n"
        "   - Tương tác Web trực tiếp: Các yêu cầu truy cập website, nhấp nút, điền form, chụp ảnh màn hình hoặc chạy thử JS trên một trang cụ thể (ví dụ: 'vào trang web abc.com và nhấn...').\n"
        "   - Khảo sát hệ thống đơn giản: Đọc nội dung 1-2 tệp tin cụ thể, liệt kê thư mục, hoặc tìm kiếm symbol.\n"
        "   - Thực thi Terminal trực tiếp: Chạy một lệnh terminal đơn lẻ.\n"
        "   - Chỉnh sửa nhỏ: Sửa đổi nhanh chỉ một vài dòng mã hoặc ghi đè một tệp tin ngắn dưới 100 dòng.\n\n"
        "2. KIỂM TRA TÁC VỤ PHỨC TẠP (is_simple = False):\n"
        "   - Phát triển tính năng mới (Feature Development): Yêu cầu viết mới hoặc can thiệp chỉnh sửa logic phức tạp trên nhiều tệp tin nguồn khác nhau.\n"
        "   - Sửa lỗi hệ thống diện rộng (Complex Bug Fixing): Đòi hỏi phải phân tích kiến trúc, tìm kiếm ký hiệu xuyên suốt mã nguồn trước khi sửa đổi.\n"
        "   - Phân tích & Viết tài liệu tổng thể dự án: Khảo sát sâu toàn bộ workspace lớn.\n\n"
        "⚠️ QUY TẮC CHỌN TASK_TYPE (RẤT QUAN TRỌNG):\n"
        "   - Chọn 'analysis' CHỈ KHI yêu cầu thuần túy là đọc hiểu, giải thích cấu trúc mã nguồn, dịch thuật hoặc khảo sát tĩnh dự án (KHÔNG sửa đổi code, KHÔNG chạy lệnh terminal, và KHÔNG tương tác/chạy thử nghiệm trình duyệt web).\n"
        "   - BẮT BUỘC CHỌN 'development' cho mọi trường hợp còn lại, bao gồm: Có viết/sửa code, chạy lệnh terminal, HOẶC cần khởi chạy trình duyệt web thật (Dynamic Web Testing) để nạp extension, tương tác web, kiểm tra hành vi runtime của ứng dụng."
    )
    
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

def chrome_extension_debugger_node(state: AgentState) -> Dict[str, Any]:
    ext_path = state.get("extension_path")
    
    if not ext_path:
        return {"messages": [AIMessage(content="Bỏ qua gỡ lỗi: Không tìm thấy Extension Path.")]}

    user_query = (
        f"Hãy kết nối CDP vào Chrome, nạp Extension từ thư mục '{ext_path}', "
        f"kiểm tra xem có bất kỳ thông báo lỗi console hoặc lỗi network request nào "
        f"liên quan đến Extension hoạt động không."
    )
    
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():
        import nest_asyncio
        nest_asyncio.apply()
        
    debug_raw_output = loop.run_until_complete(
        run_agent_with_devtools_mcp(
            model=model,
            prompt_message=user_query,
            chat_history=list(state.get("messages", []))
        )
    )
    
    structured_evaluator = model.with_structured_output(RuntimeVerificationResult, method="function_calling")
    
    system_prompt = (
        "Bạn là một chuyên gia QA. Hãy đọc báo cáo gỡ lỗi trình duyệt và xác định xem "
        "ứng dụng/extension có gặp lỗi runtime nghiêm trọng nào không (như crash, undefined variables, "
        "failed to load resource, hoặc lỗi CORS)."
    )
    
    try:
        eval_result = structured_evaluator.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Báo cáo gỡ lỗi thô:\n\n{debug_raw_output}")
        ])
        has_error = eval_result.has_critical_error
        error_summary = eval_result.error_summary
    except Exception:
        has_error = False
        error_summary = ""
        
    ret_state = {
        "browser_console_logs": debug_raw_output,
        "messages": [AIMessage(content=f"📋 **[Kết quả kiểm tra Runtime CDP]**:\n\n{debug_raw_output}")]
    }
    
    if has_error:
        ret_state["error_logs"] = f"❌ [Lỗi Runtime Trình Duyệt]: {error_summary}"
        
    return ret_state

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
        
    prompt = (
        "Bạn là một Chuyên Gia Gỡ Lỗi Độc Lập Hệ Thống (Isolated Debugger Agent).\n"
        "Nhiệm vụ duy nhất của bạn là phân tích toàn bộ mã nguồn của dự án được cung cấp dưới đây, "
        "đối chiếu với thông tin báo lỗi hoặc yêu cầu sửa lỗi từ người dùng, xác định chính xác nguyên nhân "
        "và đưa ra giải pháp sửa lỗi chi tiết nhất có thể.\n\n"
        "⚠️ QUY TẮC PHÂN TÍCH:\n"
        "1. Xác định rõ ràng các tệp tin, lớp, hàm hoặc dòng mã gây ra lỗi.\n"
        "2. Giải thích rõ ràng nguyên nhân tại sao lỗi xảy ra (logic lỗi, không khớp kiểu dữ liệu, runtime exception, v.v.).\n"
        "3. Đưa ra phương án sửa chữa tối ưu dưới dạng mã nguồn chỉnh sửa cụ thể (đặc biệt khuyến khích viết theo khối SEARCH/REPLACE để Executor dễ dàng thực thi).\n"
        "4. Chỉ xuất ra nội dung phân tích và mã nguồn sửa đổi trực tiếp, KHÔNG viết lời chào hỏi hay gọi bất kỳ công cụ nào.\n\n"
        "=== MÃ NGUỒN TOÀN BỘ DỰ ÁN ===\n"
        f"{sources_data}\n\n"
        "=== YÊU CẦU SỬA LỖI HỆ THỐNG ===\n"
        f"{user_query}\n\n"
        "Hãy viết báo cáo gỡ lỗi và phương án sửa chữa chi tiết bằng tiếng Việt:"
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
        
        system_prompt = (
            "=== ĐỊNH VỊ VAI TRÒ & PHÂN LOẠI TÁC VỤ KHẢO SÁT (ACTIVE DISCOVERY) ===\n"
            "Bạn là Agent Khảo Sát Thích Ứng (Adaptive Discovery Agent).\n"
            "Nhiệm vụ hiện tại:\n{tasks_str}\n"
            f"Thư mục làm việc: {ws}\n\n"
            
            "⚠️ BƯỚC 1: XÁC ĐỊNH PHẠM VI KHẢO SÁT VÀ BIÊN GIỚI TÀI LIỆU (BẮT BUỘC):\n"
            "Hãy phân tích yêu cầu hiện tại để tự động cấu hình phạm vi hoạt động của bạn:\n"
            "- [NHÓM A] CHỈNH SỬA FILE CỤC BỘ / SỬA LỖI (Localized Code Edit):\n"
            "  * Chỉ được phép quét, đọc đúng file cần sửa và các file import liên quan trực tiếp. Tuyệt đối KHÔNG đọc toàn bộ dự án.\n"
            "  * TUYỆT ĐỐI KHÔNG viết PRD.md hay tạo ADRs mới. Chỉ cập nhật thông tin cực kỳ ngắn gọn vào CONTEXT.md nếu file này đã tồn tại sẵn.\n"
            "- [NHÓM B] XÂY DỰNG TÍNH NĂNG MỚI / THAY ĐỔI LỚN (Macro Feature):\n"
            "  * Được phép khảo sát diện rộng và viết/cập nhật `PRD.md`, `CONTEXT.md`, và các quyết định kiến trúc trong `docs/adr/`.\n"
            "- [NHÓM C] TÁC VỤ PHI LẬP TRÌNH / VẬN HÀNH (Non-coding Ops - ví dụ: Tối ưu Notion, setup công cụ):\n"
            "  * Tuyệt đối KHÔNG đọc mã nguồn, KHÔNG tạo các file code rác, và TUYỆT ĐỐI KHÔNG tạo `PRD.md`, `CONTEXT.md` hay `ADRs`.\n"
            "  * Chỉ được phép khảo sát cấu trúc logic của ứng dụng mục tiêu (ví dụ: các trang, cơ sở dữ liệu Notion) và lưu thông tin dạng text hoặc sơ đồ vận hành gọn nhẹ (ví dụ: `NOTION_STRUCTURE.md`) nếu thực sự cần thiết.\n\n"

            "⚠️ BIÊN GIỚI NGHIÊM NGẶT - CHỐNG TỰ Ý LẬP KẾ HOẠCH & SỬA CODE CHẠY THẬT (BẮT BUỘC):\n"
            "1. Bạn ĐANG Ở PHA KHẢO SÁT. Nhiệm vụ của bạn chỉ là tìm hiểu hiện trạng, thu thập sự thật (facts) và viết tài liệu mô tả (nếu thuộc Nhóm B).\n"
            "2. TUYỆT ĐỐI KHÔNG tự ý viết kế hoạch triển khai (ví dụ: các bước sửa code tiếp theo), không đưa ra danh sách task cho pha sau, không viết code mẫu hoặc sửa file chạy thật của ứng dụng.\n"
            "3. Nếu bạn bắt đầu đưa ra kế hoạch thực thi hoặc sửa code, bạn đã vi phạm biên giới pha và sẽ làm sập hệ thống.\n\n"

            "⚠️ HƯỚNG DẪN TRUY XUẤT TRI THỨC TOÀN CỤC (JUST-IN-TIME RETRIEVAL):\n"
            "1. Hệ thống tích hợp bộ nhớ tri thức toàn cục (Global Brain) lưu tại thư mục hệ thống: `~/.openwiki/wiki/`.\n"
            "2. Trước khi tiến hành khảo sát, bạn BẮT BUỘC phải gọi công cụ `query_global_openwiki` để kiểm tra xem có quy trình hoặc lưu ý phòng ngừa lỗi liên quan đến nhiệm vụ này hay không.\n"
            "3. Áp dụng chính xác các lưu ý này vào tài liệu khảo sát.\n\n"

            "⚠️ QUY TẮC TỐI ƯU HÓA TOKEN BẰNG GỌI CÔNG CỤ SONG SONG (PARALLEL TOOL CALLING):\n"
            "1. Bạn được KHUYẾN KHÍCH MẠNH MẼ gọi NHIỀU CÔNG CỤ CÙNG LÚC (Parallel Tool Calling) trong một lượt phản hồi nếu các công cụ đó độc lập hoặc phục vụ chung một nhiệm vụ hiện hành.\n"
            "2. Ví dụ: Hãy gọi đồng thời nhiều cuộc gọi `read_file_lines` cho các tệp tin khác nhau, hoặc kết hợp `search_keyword` và `list_directory` trong cùng một turn để tiết kiệm token.\n\n"

            "⚠️ QUY TẮC KÍCH HOẠT KỸ NĂNG (BẮT BUỘC):\n"
            "1. Trước khi thực hiện hành động chuyên biệt có kỹ năng tương ứng trong danh sách '=== THƯ VIỆN KỸ NĂNG KHẢ DỤNG ===' mà chưa được kích hoạt, bạn BẮT BUỘC phải gọi công cụ `activate_agent_skill` để nạp hướng dẫn kỹ năng đó.\n"
            "2. Khi kỹ năng đã hoạt động, tuân thủ tuyệt đối cấu trúc và quy trình được ghi rõ trong hướng dẫn kỹ năng đó. Tuyệt đối không tự suy diễn cấu trúc.\n\n"

            "⚠️ QUY TẮC CÔ LẬP NHIỆM VỤ ĐƠN (SINGLE TASK ISOLATION - BẮT BUỘC):\n"
            "1. Bạn CHỈ ĐƯỢC PHÉP tập trung thực hiện duy nhất (1) nhiệm vụ đang được liệt kê tại phần 'Nhiệm vụ hiện tại' ở trên (ví dụ: chỉ giải quyết T_SURVEY).\n"
            "2. TUYỆT ĐỐI KHÔNG tự ý thực hiện tiếp các nhiệm vụ tiếp theo trong lộ trình (T1, T2...) mặc dù bạn có thể nhìn thấy chúng trong lịch sử trò chuyện.\n"
            "3. Ngay sau khi hoàn thành xong nhiệm vụ được chỉ định, bạn BẮT BUỘC phải gọi công cụ 'complete_agent_task' để bàn giao kết quả và chuyển quyền điều phối về cho đồ thị Orchestrator dọn dẹp bộ nhớ.\n\n"

            "⚠️ QUY TẮC THIẾT LẬP TÀI LIỆU & KHẢO SÁT (BẮT BUỘC):\n"
            "1. Thúc đẩy tiến trình phỏng vấn không khoan nhượng (Grilling): hãy tiếp tục gọi `ask_questions_if_underspecified` nếu các yêu cầu nghiệp vụ chưa được làm rõ tuyệt đối.\n"
            "2. Khi khảo sát mã nguồn, hãy ưu tiên dùng `search_keyword` để định vị trước, sau đó dùng `read_file_lines` để đọc phân đoạn thay vì đọc cả file lớn.\n\n"

            "⚠️ QUY TẮC BẮT BUỘC KHI GỌI CÔNG CỤ complete_agent_task:\n"
            "Do toàn bộ nhật ký gọi công cụ thô sẽ bị dọn dẹp khỏi RAM ngay sau khi nhiệm vụ kết thúc để tiết kiệm token, "
            "bản tóm tắt (summary) của bạn khi gọi công cụ `complete_agent_task` BẮT BUỘC phải chứa đầy đủ thông tin khảo sát súc tích và THÍCH ỨNG THEO NHÓM NHIỆM VỤ đã phân loại:\n"
            "- [Nhóm nhiệm vụ đã xác định]: Ghi rõ Nhóm A, Nhóm B, hay Nhóm C.\n"
            "- [Cấu trúc và Phát hiện chính]: Liệt kê các thư mục, tệp tin quan trọng đã phát hiện (nếu là Nhóm A, B) hoặc mô tả sơ đồ logic workspace (nếu là Nhóm C).\n"
            "- [Kết quả Phỏng vấn & Đặc tả]: Tóm tắt các quyết định nghiệp vụ then chốt đã đồng thuận với người dùng.\n"
            "- [Đường dẫn Tài liệu]: Xác nhận các tệp tài liệu thực tế được cập nhật/tạo mới (nếu là Nhóm B). Với Nhóm A và Nhóm C, ghi rõ 'Bỏ qua tạo tài liệu PRD/ADR/CONTEXT theo rào cản pha'."
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
        
        system_prompt = (
            "Bạn là kỹ sư phần mềm thực thi chuyên nghiệp (Write-Access Mode).\n"
            f"Nhiệm vụ phát triển:\n{tasks_str}\n"
            f"Thư mục làm việc: {ws}\n\n"
            
            "⚠️ QUY TẮC TỐI ƯU HÓA TOKEN BẰNG GỌI CÔNG CỤ SONG SONG (PARALLEL TOOL CALLING):\n"
            "1. Hãy tận dụng tối đa cơ chế GỌI CÔNG CỤ SONG SONG (Parallel Tool Calling) của mô hình để thực thi nhiều hành động độc lập trong cùng một lượt phản hồi.\n"
            "2. Ví dụ: Bạn có thể sửa đổi nhiều file thông qua việc gọi đồng thời nhiều công cụ `apply_search_replace_patch` cho các tệp khác nhau, hoặc kết hợp việc đọc file và tìm kiếm ký hiệu cùng lúc.\n"
            "3. Tuyệt đối không chia nhỏ các hành động độc lập thành nhiều chu kỳ suy nghĩ tuần tự nếu có thể gộp chung vào một lượt gọi song song để tránh lãng phí và tích lũy token ngữ cảnh không đáng có.\n\n"

            "⚠️ QUY TẮC KÍCH HOẠT & TUÂN THỦ KỸ NĂNG (BẮT BUỘC):\n"
            "1. Trước khi thực hiện viết code, sửa lỗi, hoặc viết test, hãy rà soát danh sách '=== THƯ VIỆN KỸ NĂNG KHẢ DỤNG (TIER 1) ==='. "
            "Nếu có kỹ năng hỗ trợ phù hợp (ví dụ: 'test-driven-development'), bạn BẮT BUỘC phải gọi công cụ `activate_agent_skill` để tải chỉ dẫn sâu.\n"
            "3. Ưu tiên chạy các script chuyên dụng của kỹ năng bằng `run_skill_script` thay vì tự gõ lệnh terminal thủ công nếu hệ thống có sẵn.\n\n"
            "⚠️ QUY TẮC CÔ LẬP NHIỆM VỤ ĐƠN (SINGLE TASK ISOLATION - BẮT BUỘC):\n"
            "1. Bạn CHỈ ĐƯỢC PHÉP thực hiện duy nhất (1) nhiệm vụ đang được liệt kê tại phần 'Nhiệm vụ phát triển' ở trên (ví dụ: chỉ giải quyết T1).\n"
            "2. TUYỆT ĐỐI KHÔNG tự ý thực hiện các nhiệm vụ tiếp theo trong lộ trình (T2, T3...) mặc dù bạn có thể nhìn thấy chúng trong lịch sử trò chuyện. Việc tự ý gom nhiều nhiệm vụ để thực hiện liên tiếp trong một lượt sẽ bỏ qua bước dọn dẹp tin nhắn rác của đồ thị, gây sập hệ thống do quá tải token.\n"
            "3. Ngay sau khi hoàn thành nhiệm vụ được giao, bạn BẮT BUỘC phải gọi công cụ 'complete_agent_task' để bàn giao kết quả và chuyển quyền điều phối về cho đồ thị dọn dẹp bộ nhớ trước khi tiếp nhận nhiệm vụ mới,và không tạo file tóm tắt thừa vì nội dung đã được lưu trong complete_agent_task\n\n"
            "⚠️ HƯỚNG DẪN TIẾT KIỆM TOKEN:\n"
            "Ưu tiên sử dụng `apply_search_replace_patch` thay vì ghi đè lại toàn bộ tệp tin lớn bằng `write_file`.\n\n"
            "⚠️ QUY TẮC BẮT BUỘC KHI THỰC HIỆN KIỂM THỬ THỦ CÔNG (MANUAL TESTING & QA VERIFICATION):\n"
            "1. TUYỆT ĐỐI KHÔNG chạy các lệnh terminal duy trì liên tục (persistent/blocking process) như `flutter run` hoặc `npm start` một cách đồng bộ vì sẽ làm nghẽn toàn bộ luồng suy nghĩ của hệ thống. "
            "Hãy chạy ở chế độ nền (ví dụ: `start /B flutter run` trên Windows, hoặc `flutter run &` trên macOS/Linux) hoặc hướng dẫn rõ ràng để người dùng tự khởi động ứng dụng trên thiết bị/emulator của họ.\n"
            "2. TUYỆT ĐỐI KHÔNG chỉ viết checklist ra file `.md` tĩnh rồi kết thúc tác vụ mà không đợi phản hồi. "
            "Bạn BẮT BUỘC phải gọi công cụ `ask_questions_if_underspecified` để gửi một biểu mẫu (Form) câu hỏi động đến người dùng. "
            "Cấu trúc biểu mẫu phải chuyển hóa các hạng mục kiểm thử (như Khởi động App, Trang Settings, Chức năng Tìm kiếm, Đọc truyện) thành các câu hỏi dạng 'select' (Lựa chọn) với hai giá trị phản hồi: 'pass' (Thành công) và 'fail' (Thất bại) để người dùng tích chọn trực tiếp trên giao diện.\n"
            "3. Phân tích kết quả phản hồi của người dùng trong lượt tin nhắn (HumanMessage) tiếp theo để xử lý thông minh:\n"
            "   - Nếu tất cả các hạng mục đều đạt trạng thái 'pass', hãy gọi công cụ `complete_agent_task` để chính thức đóng nhiệm vụ.\n"
            "   - Nếu có bất kỳ hạng mục nào bị người dùng đánh giá 'fail' hoặc báo lỗi, hãy đọc kỹ ghi chú phản hồi, tự động định vị các file mã nguồn liên quan để sửa lỗi, sau đó thiết lập và kích hoạt lại biểu mẫu kiểm thử động này để xác thực lại.\n\n"
            "⚠️ QUY TẮC BẮT BUỘC KHI GỌI CÔNG CỤ complete_agent_task (BẢO VỆ NGỮ CẢNH):\n"
            "Do toàn bộ lịch sử tin nhắn thô, logs chạy terminal, và mã nguồn cũ sẽ bị dọn dẹp sạch khỏi RAM ngay sau bước này để tiết kiệm token [1], "
            "bản tóm tắt (summary) của bạn trong công cụ complete_agent_task bắt buộc phải đóng vai trò là CẦU NỐI TRI THỨC không hao hụt (Lossless Bridge). "
            "Bạn TUYỆT ĐỐI KHÔNG được viết tóm tắt chung chung (ví dụ: 'Đã sửa tệp config.py'). "
            "Bản tóm tắt của bạn BẮT BUỘC phải ghi nhận chi tiết, chính xác các điểm kỹ thuật sau:\n"
            "1. [Tệp tin thay đổi]: Ghi cụ thể đường dẫn tương đối của các file đã tạo mới hoặc chỉnh sửa (ví dụ: `src/config.py`).\n"
            "2. [Chi tiết sửa đổi Code]: Liệt kê chính xác tên Class, Hàm, API endpoints, hoặc Biến được khai báo mới hoặc thay đổi logic "
            "(ví dụ: 'Thêm hàm fetch_user_data(user_id: int) -> dict trong Class UserManager, trả về JSON gồm {id, name, email}').\n"
            "3. [Phương án kỹ thuật & Giải thuật]: Giải thích ngắn gọn cách bạn giải quyết vấn đề (ví dụ: 'Sử dụng cơ chế Lock để ngăn race condition khi khởi tạo luồng').\n"
            "4. [Lưu ý & Chỉ dẫn kế thừa cho Task sau]: Ghi rõ các điểm cần chú ý để Task tiếp theo có thể import hoặc gọi chính xác "
            "(ví dụ: 'Task tiếp theo khi làm việc với Router cần import fetch_user_data từ src/config.py và truyền đối số dạng integer')."
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
        system_prompt = (
            "Bạn là một Kiến trúc sư kiêm Điều phối viên dự án phần mềm cấp cao.\n"
            f"Nhiệm vụ: Dựa trên dữ liệu khảo sát và thám thính dự án vừa qua tại '{ws}' (ở các tin nhắn trước), "
            "hãy thiết kế một lộ trình hành động (DAG updated_tasks) hoàn chỉnh để giải quyết trọn vẹn yêu cầu của người dùng.\n\n"
            "⚠️ QUY TẮC ĐÁNH GIÁ TRẠNG THÁI HOÀN THÀNH SỚM (OPPORTUNISTIC COMPLETION CHECK - BẮT BUỘC):\n"
            "1. Hãy phân tích kỹ lịch sử gọi công cụ và nội dung trao đổi gần nhất.\n"
            "2. Nếu bạn phát hiện ra rằng Agent trong pha Khảo sát (T_SURVEY) bằng cách nào đó ĐÃ HOÀN THÀNH TRỌN VẸN yêu cầu cốt lõi "
            "   cuối cùng của người dùng (ví dụ: đã thực thi xong toàn bộ việc tái cấu trúc các trang Notion, tạo các Hubs thành công...) "
            "   và không cần thực hiện thêm bất kỳ hành động phát triển hoặc viết mã nào nữa:\n"
            "   - Hãy đặt `should_modify_plan` là True để cập nhật lại danh sách lộ trình.\n"
            "   - Đặt `task_type` là 'analysis'.\n"
            "   - Trong danh sách `updated_tasks`, CHỈ giữ lại duy nhất nhiệm vụ 'T_SURVEY' với trạng thái là 'completed'. TUYỆT ĐỐI KHÔNG thêm bất kỳ nhiệm vụ mới nào ở trạng thái 'pending'.\n"
            "   - Giải thích thật rõ ràng trong `explanation` lý do tại sao hệ thống đã giải quyết xong bài toán và có thể kết thúc đồ thị ngay lập tức.\n"
            "3. Ngược lại, nếu mục tiêu vẫn chưa được thực hiện xong và cần viết code, cấu hình hay kiểm thử, hãy thiết kế lộ trình phát triển bình thường theo hướng dẫn bên dưới.\n\n"
            "⚠️ QUY TẮC PHÂN PHỐI KỸ NĂNG VÀO LỘ TRÌNH (BẮT BUỘC):\n"
            "1. Hãy đối chiếu yêu cầu của lộ trình với danh mục kỹ năng hiện có trong hệ thống.\n"
            "2. Nếu một nhiệm vụ đòi hỏi quy trình kỹ thuật đặc thù (ví dụ: phát triển kèm viết test, thao tác Excel, cấu hình PRD), "
            "bạn BẮT BUỘC phải ghi rõ yêu cầu kích hoạt kỹ năng tương ứng vào mô tả nhiệm vụ (Task Description).\n"
            "   (Ví dụ: '...Bắt buộc gọi công cụ activate_agent_skill để kích hoạt kỹ năng test-driven-development trước khi tiến hành viết code...')\n\n"
            "⚠️ QUY TẮC THIẾT LẬP KẾ HOẠCH PHÁT TRIỂN & KIỂM THỬ (BẮT BUỘC):\n"
            "1. Đặt `should_modify_plan` là True để áp dụng kế hoạch mới.\n"
            "2. Giữ nguyên nhiệm vụ 'T_SURVEY' với trạng thái là 'completed'.\n"
            "3. Bổ sung các nhiệm vụ mới (ví dụ: T1, T2...) mô tả chính xác các file cần xem, các file cần sửa dựa trên dữ liệu thật thu được từ pha khảo sát.\n"
            "4. TUYỆT ĐỐI BẮT BUỘC phải lập kế hoạch cho nhiệm vụ 'Kiểm thử tích hợp động trên trình duyệt thật' sử dụng công cụ `web_autonomous_executor` "
            "để ủy quyền cho bộ điều phối tự động nạp Extension, thực hiện chuỗi hành động và xác thực kết quả nghiệp vụ cuối cùng!\n"
            "5. Đặt `task_type` là 'development' (vì chúng ta sẽ sửa code và chạy trình duyệt kiểm thử)."
        )
    else:
        system_prompt = (
            "Bạn là một Kiến trúc sư kiêm Điều phối viên dự án phần mềm cấp cao.\n"
            f"Nhiệm vụ: Đánh giá tiến trình thực thi kế hoạch tại thư mục làm việc '{ws}'.\n\n"
            "Hệ thống vừa phát hiện lỗi nghiêm trọng không thể tự gỡ lỗi ở cấp độ cục bộ.\n"
            "Hãy đề xuất một kế hoạch điều chỉnh (được cập nhật trong updated_tasks) để giải quyết triệt để lỗi này.\n\n"
            "⚠️ QUY TẮC PHÂN PHỐI KỸ NĂNG VÀO LỘ TRÌNH (BẮT BUỘC):\n"
            "- Nếu phát hiện lỗi có liên quan đến việc vận hành lệch chuẩn quy trình của một kỹ năng, hãy thêm một nhiệm vụ cụ thể "
            "yêu cầu Executor kích hoạt kỹ năng đó bằng `activate_agent_skill` để rà soát và cấu trúc lại mã nguồn theo chuẩn.\n\n"
            "⚠️ QUY TẮC CẬP NHẬT KẾ HOẠCH CHO PRODUCTION (BẮT BUỘC):\n"
            "1. Đặt `should_modify_plan` là True và cập nhật danh sách nhiệm vụ trong `updated_tasks` để giải quyết vấn đề.\n"
            "2. ĐỐI VỚI CÁC NHIỆM VỤ ĐÃ HOÀN THÀNH (status: 'completed'): Bắt buộc giữ nguyên ID, mô tả và trạng thái là 'completed'.\n"
            "3. Kế hoạch cập nhật của bạn chỉ tập trung hoàn toàn vào các bước thực thi khảo sát vật lý (analysis) hoặc sửa đổi mã nguồn (development)."
        )
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

def tester_node(state: AgentState) -> Dict[str, Any]:
    modified_files = state.get("modified_files", [])
    attempts = state.get("attempts", 0)
    plan = state["plan"]
    ws = state.get("workspace_path", ".")
    last_executed_ids = state.get("last_executed_task_ids", [])
    
    parsed_plan = []
    for t in plan:
        if isinstance(t, dict):
            parsed_plan.append(Task(**t))
        else:
            parsed_plan.append(t)
            
    errors = []
    warnings = []
    workspace_root = Path(ws).expanduser().resolve()
    
    files_by_ext: Dict[str, List[Path]] = {}
    for f_path_str in modified_files:
        try:
            p = Path(f_path_str).resolve()
            if p.exists() and p.is_file():
                ext = p.suffix.lower()
                files_by_ext.setdefault(ext, []).append(p)
                clear_compiler_cache(workspace_root, ext)
        except Exception:
            pass

    for ext, files in files_by_ext.items():
        if ext == ".py":
            import sys
            for f in files:
                if f.name.startswith("test_") or f.name.endswith("_test.py"):
                    code, output = execute_validation_cmd([sys.executable, str(f)], workspace_root)
                    if code == -99:
                        warnings.append(output)
                    elif code != 0:
                        errors.append(f"❌ [Lỗi Thực Thi Unit Test Python] tại tệp `{f.name}`:\n{clean_compiler_logs(output)}")
                else:
                    code, output = execute_validation_cmd([sys.executable, "-m", "py_compile", str(f)], workspace_root)
                    if code == -99:
                        warnings.append(output)
                    elif code != 0:
                        errors.append(f"❌ [Lỗi Cú Pháp Python] tại tệp `{f.name}`:\n{clean_compiler_logs(output)}")

        elif ext == ".dart":
            for f in files:
                target_dir = find_nearest_config(f, "pubspec.yaml") or workspace_root
                code, output = execute_validation_cmd(["dart", "analyze"], target_dir)
                if code == -99:
                    warnings.append(f"{output} (Bỏ qua kiểm tra tĩnh cho `{f.name}`)")
                elif code != 0:
                    errors.append(f"❌ [Lỗi Dart Analysis] tại sub-project `{target_dir.name}`:\n{clean_compiler_logs(output)}")
                    break

        elif ext in [".ts", ".tsx", ".js", ".jsx"]:
            for f in files:
                target_dir = find_nearest_config(f, "package.json") or workspace_root
                if ext in [".ts", ".tsx"]:
                    cmd = ["npx", "tsc", "--noEmit", "--skipLibCheck"]
                    code, output = execute_validation_cmd(cmd, target_dir)
                    if code == -99:
                        warnings.append(f"{output} (Bỏ qua phân tích kiểu dữ liệu cho `{f.name}`)")
                    elif code != 0:
                        errors.append(f"❌ [Lỗi TypeScript Compile] tại `{target_dir.name}`:\n{clean_compiler_logs(output)}")
                        break

        elif ext == ".rs":
            for f in files:
                target_dir = find_nearest_config(f, "Cargo.toml") or workspace_root
                code, output = execute_validation_cmd(["cargo", "check"], target_dir)
                if code == -99:
                    warnings.append(f"{output} (Bỏ qua biên dịch Rust cho `{f.name}`)")
                elif code != 0:
                    errors.append(f"❌ [Lỗi Biên Dịch Rust] tại `{target_dir.name}`:\n{clean_compiler_logs(output)}")
                    break

        elif ext == ".go":
            for f in files:
                target_dir = find_nearest_config(f, "go.mod") or workspace_root
                code, output = execute_validation_cmd(["go", "vet", "./..."], target_dir)
                if code == -99:
                    warnings.append(f"{output} (Bỏ qua kiểm tra tĩnh Go cho `{f.name}`)")
                elif code != 0:
                    errors.append(f"❌ [Lỗi Tĩnh Go Vet] tại `{target_dir.name}`:\n{clean_compiler_logs(output)}")
                    break

    warning_msg = ""
    if warnings:
        warning_msg = "⚠️ **Cảnh báo môi trường:**\n" + "\n".join([f"- {w}" for w in warnings]) + "\n\n"

    if errors:
        combined_error = "\n\n---\n\n".join(errors)
        
        if attempts < 3:
            updated_plan = []
            for t in parsed_plan:
                t_copy = t.model_copy()
                if t_copy.id in last_executed_ids:
                    t_copy.status = "pending"
                updated_plan.append(t_copy)
                
            return {
                "error_logs": combined_error,
                "attempts": attempts + 1,
                "plan": updated_plan,
                "messages": [AIMessage(content=f"{warning_msg}⚠️ [Vòng kiểm thử thất bại] Phát hiện lỗi ở mã nguồn sửa đổi:\n\n{combined_error}\n\n⚙️ Đang gửi trả trạng thái nhiệm vụ về 'pending' để tự động sửa chữa.")]
            }
        else:
            return {
                "error_logs": combined_error,
                "attempts": attempts,
                "messages": [AIMessage(content=f"{warning_msg}❌ Đã vượt quá giới hạn {attempts} lần sửa lỗi tự động. Chuyển giao bối cảnh lỗi về cho bộ điều phối Replanner.")]
            }
                
    success_content = f"{warning_msg}✅ [Vòng kiểm thử thành công] Toàn bộ mã nguồn đã vượt qua kiểm tra tĩnh."
    return {
        "error_logs": "",
        "attempts": 0,
        "modified_files": [],
        "messages": [AIMessage(content=success_content)]
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

    synthesis_prompt = (
        "Bạn là một Kiến trúc sư Hệ thống chuyên nghiệp chuyên biên soạn tài liệu.\n"
        "Hãy tổng hợp toàn bộ thông tin khảo sát thô được ghi nhận ở các bước trước thành một tài liệu 'CONTEXT.md' duy nhất "
        "chứa đầy đủ bối cảnh dự án, sơ đồ kiến trúc và bảng thuật ngữ hệ thống.\n"
        "Yêu cầu: Viết thật cô đọng, súc tích và có cấu trúc rõ ràng. TUYỆT ĐỐI KHÔNG chèn mã nguồn dài dòng."
    )
    
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