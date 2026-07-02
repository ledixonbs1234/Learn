# oder/nodes.py
import asyncio
import json
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional,  Tuple, List, Union
from concurrent.futures import ThreadPoolExecutor
from langgraph.errors import GraphInterrupt, GraphBubbleUp
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.types import interrupt
from config import find_project_root_heuristic, model, sanitize_and_resolve_path, fast_model
from mcp_helper import run_agent_with_devtools_mcp
from state import AgentState, PlanUpdate, RuntimeVerificationResult, TaskPlan, TaskTriage, Task
from tools import (
    AskQuestionsTool, GitManager, ProposePlanTool, ReadFileLinesTool, UniversalSymbolSearchTool, WebInteractAndTestTool, WorkspaceTools, 
    ReadFilesTool, WriteFileTool, ApplyPatchTool, 
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

def compact_reading_tool_messages(messages: List[BaseMessage]) -> List[BaseMessage]:
    compacted_messages = []
    for msg in messages:
        # 🛡️ CHỈ nén 'read_files' (đọc toàn bộ file), KHÔNG nén 'read_file_lines' và 'search_symbols_universal'
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
    error_keywords = ["error", "fail", "exception", "cause", "unhandled", "invalid", "undefined"]
    
    for line in lines:
        clean_line = line.strip()
        if not clean_line:
            continue
        has_error_kw = any(kw in clean_line.lower() for kw in error_keywords)
        has_line_indicator = ":" in clean_line or ".dart" in clean_line or ".py" in clean_line or ".ts" in clean_line
        
        if has_error_kw or has_line_indicator:
            filtered_lines.append(line)
            
    if not filtered_lines:
        if len(lines) > 40:
            return "\n".join(lines[:20] + ["... [Đã lược bớt các dòng ở giữa] ..."] + lines[-20:])
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
    """
    Bộ tiền xử lý tĩnh (Deterministic Resolver) nhận diện các thư mục đặc biệt 
    của hệ điều hành để định vị chính xác yêu cầu của người dùng mà không cần LLM đoán mò.
    """
    text_lower = text.lower()
    home = Path.home()
    
    # Bản đồ ánh xạ các từ khóa chỉ định thư mục hệ thống
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


def verify_workspace_safety(workspace_path: str) -> bool:
    """
    Hệ thống phòng thủ an toàn (Security Guardrail):
    Ngăn chặn tuyệt đối việc Agent trỏ Workspace vào thư mục nguồn của chính nó.
    """
    try:
        resolved_workspace = Path(workspace_path).expanduser().resolve()
        current_agent_dir = Path(__file__).parent.parent.resolve() # Thư mục Coder/
        
        # Nếu trùng khít hoặc workspace chứa thư mục Agent -> Không an toàn
        if resolved_workspace == current_agent_dir or resolved_workspace in current_agent_dir.parents:
            return False
            
        # Kiểm tra xem có chứa các file điều khiển quan trọng của Agent không
        control_files = ["browser_subgraph.py", "mcp_helper.py", "routers.py"]
        if any((resolved_workspace / f).exists() for f in control_files):
            return False
            
        return True
    except Exception:
        return False

def detect_and_triage_node(state: AgentState) -> Dict[str, Any]:
    """
    Nút phân loại và thiết lập môi trường hoạt động cấp độ Production.
    Xử lý an toàn bảo mật, chống OOD, tự động phân tích đường dẫn hệ thống và tương tác Human-in-the-Loop.
    """
    messages = state["messages"]
    user_msg = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) or getattr(msg, "type", None) == "human":
            user_msg = msg
            break
            
    user_query_text = get_text_content_safely(user_msg.content) if user_msg else ""
    
    # ==========================================
    # BƯỚC 1: TIỀN XỬ LÝ ĐƯỜNG DẪN TĨNH (OS DETERMINISTIC PATH RESOLUTION)
    # ==========================================
    detected_path_str = resolve_special_system_paths(user_query_text)
    if not detected_path_str:
        # Nếu không có từ khóa đặc biệt, quét tìm đường dẫn thô trong câu lệnh
        detected_path_str = extract_path_from_text(user_query_text)

    # ==========================================
    # BƯỚC 2: PHÂN LOẠI TÁC VỤ QUA LLM (STRUCTURAL TRIAGE)
    # ==========================================
    structured_llm = model.with_structured_output(TaskTriage, method="function_calling")
    
    triage_prompt = (
        "Bạn là một điều phối viên Agent thông minh cấp cao (Triage Supervisor).\n"
        "Nhiệm vụ của bạn là phân tích yêu cầu của người dùng để phân loại chính xác hướng xử lý.\n\n"
        "⚠️ QUY TẮC PHÂN LOẠI KHẮT KHE (BẮT BUỘC):\n"
        "1. Nếu yêu cầu KHÔNG liên quan đến lập trình, viết code, sửa code, khảo sát hệ thống file hoặc tương tác web "
        "   (ví dụ: hỏi thời tiết, kiến thức xã hội, nấu ăn, tán gẫu...), bạn BẮT BUỘC phải đặt task_type = 'ood'.\n"
        "2. Nếu yêu cầu là sửa lỗi ('lỗi trong ứng dụng này', 'sửa lỗi app của tôi') hoặc chạy thử phần mềm "
        "   nhưng người dùng KHÔNG chỉ định rõ đường dẫn thư mục hay file nào trong câu lệnh,\n"
        "   bạn BẮT BUỘC phải đặt task_type = 'clarify' để hệ thống kích hoạt dừng luồng và hỏi lại thông tin.\n"
        "3. Nếu yêu cầu là tạo mới hoàn toàn (ví dụ: 'tạo chrome extension', 'viết ứng dụng reactjs...'),\n"
        "   hãy đặt task_type = 'development' và thiết lập is_simple = False để hệ thống lập kế hoạch tạo thư mục sandbox cách ly."
    )
    
    try:
        triage_output = structured_llm.invoke([
            SystemMessage(content=triage_prompt),
            HumanMessage(content=user_query_text)
        ])
        task_type = triage_output.task_type
        is_simple = triage_output.is_simple
        detailed_analysis = triage_output.detailed_analysis
    except Exception:
        task_type = "clarify"
        is_simple = False
        detailed_analysis = "Phân tích tự động gặp sự cố. Cần kích hoạt quy trình làm rõ."

    # ==========================================
    # BƯỚC 3: XỬ LÝ CÁC KỊCH BẢN ĐẶC BIỆT (OOD & CLARIFY)
    # ==========================================
    
    # Kịch bản 3.1: Yêu cầu ngoài phạm vi (Out-Of-Domain)
    if task_type == "ood":
        return {
            "plan": [],
            "task_type": "analysis",
            "is_simple": True,
            "messages": [
                AIMessage(content="🙏 Tôi là trợ lý chuyên biệt về khảo sát mã nguồn, lập trình phần mềm và tương tác Web tự động.\n"
                                  "Yêu cầu hiện tại của bạn nằm ngoài phạm vi hỗ trợ của tôi. Vui lòng đưa ra các yêu cầu liên quan đến lập trình.")
            ]
        }

    # Kịch bản 3.2: Mơ hồ đường dẫn cần hỏi lại (Clarify via HITL Interrupt)
    if (task_type == "clarify" or "ứng dụng này" in user_query_text.lower() or "ứng dụng của tôi" in user_query_text.lower()) and not detected_path_str:
        
        # Chuẩn bị payload ngắt có cấu trúc gửi về giao diện người dùng
        interrupt_payload = {
            "type": "path_clarification",
            "prompt": "Hệ thống phát hiện bạn muốn kiểm tra/sửa đổi ứng dụng nhưng chưa cung cấp đường dẫn thư mục cụ thể.",
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
        
        # Kích hoạt ngắt đồ thị LangGraph
        # Khi đồ thị được resume, giá trị phản hồi từ giao diện sẽ được nạp vào biến user_response
        user_response = interrupt(interrupt_payload)
        
        # Trích xuất đường dẫn được cung cấp từ phản hồi resume
        if isinstance(user_response, dict) and "target_workspace_path" in user_response:
            detected_path_str = str(user_response["target_workspace_path"]).strip()
        elif isinstance(user_response, str):
            detected_path_str = user_response.strip()

    # ==========================================
    # BƯỚC 4: THIẾT LẬP WORKSPACE AN TOÀN VÀ XỬ LÝ SANDBOX CÁCH LY
    # ==========================================
    workspace_path = None
    
    if detected_path_str:
        try:
            # Giải quyết đường dẫn tuyệt đối đã xác minh
            resolved_path = Path(detected_path_str).expanduser().resolve()
            
            if resolved_path.exists():
                # Thực hiện Heuristic tìm project root từ đường dẫn được cung cấp
                workspace_path = str(find_project_root_heuristic(resolved_path))
            else:
                # Nếu đường dẫn người dùng nhập không tồn tại vật lý
                return {
                    "plan": [],
                    "task_type": "analysis",
                    "is_simple": True,
                    "messages": [AIMessage(content=f"❌ Thất bại: Đường dẫn thư mục `{detected_path_str}` không tồn tại trên hệ thống. Vui lòng kiểm tra lại.")]
                }
        except Exception as e:
            return {
                "plan": [],
                "task_type": "analysis",
                "is_simple": True,
                "messages": [AIMessage(content=f"❌ Lỗi hệ thống khi phân tích đường dẫn: {str(e)}")]
            }
    else:
        # Nếu là tác vụ tạo mới hoàn toàn (Scaffolding) và người dùng không nhập path
        # Hệ thống tự động thiết lập thư mục Sandbox cách ly tuyệt đối nằm ngoài thư mục Agent
        sandbox_root = Path.home() / ".agent_sandboxes"
        sandbox_root.mkdir(parents=True, exist_ok=True)
        
        # Tạo ID phiên làm việc cách ly
        session_id = user_query_text[:15].strip().replace(" ", "_")
        session_id = re.sub(r'[^\w\-_\.]', '', session_id) or "default_session"
        
        sandbox_workspace = sandbox_root / session_id
        sandbox_workspace.mkdir(parents=True, exist_ok=True)
        workspace_path = str(sandbox_workspace.resolve())

    # ==========================================
    # BƯỚC 5: KIỂM TRA BẢO MẬT CUỐI CÙNG (SECURITY GUARDRAIL CHECK)
    # ==========================================
    if not verify_workspace_safety(workspace_path):
        return {
            "plan": [],
            "task_type": "analysis",
            "is_simple": True,
            "messages": [
                AIMessage(content="🚨 **[CẢNH BÁO BẢO MẬT]**:\n"
                                  "Hệ thống phát hiện thư mục làm việc được chỉ định trùng khớp hoặc nằm trong thư mục nguồn của Agent Coder.\n"
                                  "Để tránh việc Agent vô tình sửa đổi nhầm mã nguồn hệ thống, yêu cầu này đã bị chặn.\n"
                                  "Vui lòng di chuyển dự án của bạn sang một thư mục độc lập khác.")
            ]
        }

    # ==========================================
    # BƯỚC 6: KHỞI TẠO LỘ TRÌNH (DAG PLAN INITIALIZATION)
    # ==========================================
    plan = []
    if is_simple:
        plan = [
            Task(
                id="T1",
                description=f"Thực hiện trực tiếp tác vụ tại `{workspace_path}`: {user_query_text}",
                dependencies=[],
                status="pending"
            )
        ]
    else:
        plan = [
            Task(
                id="T_SURVEY",
                description=f"Khảo sát cấu trúc thư mục, các tệp tin cấu hình chính trong dự án tại `{workspace_path}` để hiểu kiến trúc trước khi triển khai.",
                dependencies=[],
                status="pending"
            )
        ]
        # Ép buộc luồng phức tạp chạy pha Khảo sát (Analysis) trước
        task_type = "analysis"

    triage_info_msg = (
        f"📊 **[Hệ thống Phân phối thông minh]**:\n"
        f"- **Môi trường hoạt động (Workspace):** `{workspace_path}`\n"
        f"- **Chế độ kiểm soát:** {'Đơn giản (Fast-Track)' if is_simple else 'Phức tạp (Multi-Step Discovery)'}\n"
        f"- **Pha hoạt động khởi động:** `{task_type.upper()}`\n\n"
        f"🎯 **[Phân tích mục tiêu kỹ thuật]**:\n{detailed_analysis}"
    )

    return {
        "workspace_path": workspace_path,
        "plan": plan,
        "task_type": task_type,
        "is_simple": is_simple,
        "detailed_analysis": detailed_analysis,
        "replanning_count": 0,
        "modified_files": [],
        "error_logs": "",
        "step_findings": [],
        "messages": [AIMessage(content=triage_info_msg)]
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
    thongtin_path = Path(ws) / "THONGTIN.md"
    context_msg = "📋 Không tìm thấy tệp cấu hình `THONGTIN.md`."
    
    if thongtin_path.exists():
        try:
            workspace_context = thongtin_path.read_text(encoding="utf-8")
            context_msg = "📋 Đã tải xong ngữ cảnh thông tin dự án từ tệp `THONGTIN.md`."
        except Exception as e:
            workspace_context = f"Lỗi khi đọc file THONGTIN.md: {str(e)}"
            context_msg = f"⚠️ Gặp sự cố khi đọc tệp `THONGTIN.md`: {str(e)}"
            
    ext_dir = find_extension_dir_heuristic(Path(ws))
    ext_msg = ""
    if ext_dir:
        ext_msg = f"\n📦 **[Tự động nhận diện Extension]**: Đã định vị thư mục Chrome Extension tại: `{ext_dir}`"
    else:
        ext_msg = f"\n📦 **[Tự động nhận diện Extension]**: Không tìm thấy manifest.json trực tiếp trong thư mục workspace."
        
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
    """
    Nút xử lý gỡ lỗi chuyên sâu sử dụng Chrome DevTools MCP.
    Đánh giá xem Extension có chạy mượt mà ở runtime hay không.
    """
    ws = state["workspace_path"]
    ext_path = state.get("extension_path")
    
    if not ext_path:
        return {"messages": [AIMessage(content="Bỏ qua gỡ lỗi: Không tìm thấy Extension Path.")]}

    user_query = (
        f"Hãy kết nối CDP vào Chrome, nạp Extension từ thư mục '{ext_path}', "
        f"kiểm tra xem có bất kỳ thông báo lỗi console hoặc lỗi network request nào "
        f"liên quan đến Extension hoạt động không."
    )
    
    # Kích hoạt MCP Client chạy bất đồng bộ một cách an toàn
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
    
    # Sử dụng fast_model để phân tích nhanh kết quả thô xem có thực sự bị lỗi hay không
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


# THAY THẾ ĐOẠN CODE TRONG oder/nodes.py BẰNG ĐOẠN DƯỚI ĐÂY

def executor_node(state: AgentState) -> Dict[str, Any]:
    ws = state["workspace_path"]
    plan = state.get("plan", [])
    error_logs = state.get("error_logs", "")
    file_registry = state.get("file_registry", {})
    messages = list(state["messages"])
    task_type = state.get("task_type", "development")
    extension_path = state.get("extension_path", "")
    
    # ==========================================
    # BƯỚC 1: TỰ ĐỘNG KHỞI TẠO NGỮ CẢNH TRƯỜNG LÀM VIỆC (BOOTSTRAP ENVIRONMENT)
    # Chạy ngầm một lần duy nhất nếu chưa có thông tin Git hoặc Context
    # ==========================================
    state_updates = {}
    git_branch = state.get("git_branch", "")
    workspace_context = state.get("workspace_context", "")
    
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
        thongtin_path = Path(ws) / "THONGTIN.md"
        if thongtin_path.exists():
            try:
                workspace_context = thongtin_path.read_text(encoding="utf-8")
            except Exception:
                workspace_context = "Không thể đọc THONGTIN.md"
        else:
            workspace_context = "📋 Chưa có tệp cấu hình THONGTIN.md."
        state_updates["workspace_context"] = workspace_context

    # Tìm kiếm manifest.json nếu chưa quét
    if not extension_path:
        ext_dir = find_extension_dir_heuristic(Path(ws))
        if ext_dir:
            extension_path = ext_dir
            state_updates["extension_path"] = extension_path

    # Định vị các nhiệm vụ cần thực thi trong Plan
    eligible_tasks = get_eligible_tasks(plan)
    if not eligible_tasks:
        pending_tasks = [t for t in plan if (t.get("status") if isinstance(t, dict) else getattr(t, "status", None)) == "pending"]
        if pending_tasks:
            eligible_tasks = [pending_tasks[0]]
            
    tasks_str = ""
    if eligible_tasks:
        tasks_str = "\n".join([
            f"- [{getattr(t, 'id', None) or t.get('id')}] {getattr(t, 'description', None) or t.get('description')}"
            for t in eligible_tasks
        ])
    else:
        tasks_str = "- [Khảo sát tổng thể]: Tìm hiểu cấu trúc và giải quyết yêu cầu người dùng."

    # Định dạng mã nguồn hiện có (Single Source of Truth)
    registry_context_str = ""
    if file_registry:
        registry_context_str = "\n=== 📦 CÁC FILE ĐÃ NẠP VÀO BỘ NHỚ ===\n"
        for file_path, content in file_registry.items():
            lang = get_markdown_language(file_path)
            lines = content.splitlines()
            formatted_lines = [f"{idx+1:04d} | {line}" for idx, line in enumerate(lines)]
            registry_context_str += (
                f"\n--- TỆP TIN: `{file_path}` ---\n"
                f"```{lang}\n" + "\n".join(formatted_lines) + "\n```\n"
            )

    # ==========================================
    # BƯỚC 2: PHÂN CHIA VÀ CẤU HÌNH CÔNG CỤ THEO GIAI ĐOẠN
    # ==========================================
    if task_type == "analysis":
        # Đăng ký đầy đủ công cụ khám phá chủ động (Active Discovery Tools)
        read_files = ReadFilesTool(workspace_path=ws)
        list_directory = ListDirectoryTool(workspace_path=ws)
        search_symbols = UniversalSymbolSearchTool(workspace_path=ws)
        read_file_lines = ReadFileLinesTool(workspace_path=ws)
        ask_questions_tool = AskQuestionsTool(workspace_path=ws)
        propose_plan_tool = ProposePlanTool(workspace_path=ws) # Công cụ duyệt kế hoạch
        
        tools = [read_files, list_directory, search_symbols, read_file_lines, ask_questions_tool, propose_plan_tool]

        system_prompt = (
            "Bạn là một chuyên gia điều tra, khảo sát mã nguồn và lập kế hoạch kỹ thuật (Active Discovery Engine).\n"
            f"Nhiệm vụ hiện tại:\n{tasks_str}\n"
            f"Thư mục làm việc: {ws}\n\n"
            "⚠️ QUY TRÌNH KHẢO SÁT CHỦ ĐỘNG VÀ ĐA ĐƯỜNG DẪN (BẮT BUỘC):\n"
            "1. Sử dụng các công cụ `list_directory`, `search_symbols_universal`, `read_files` để thám thính và tìm hiểu "
            "   nguyên nhân gây ra vấn đề trong thư mục dự án.\n"
            "2. Đừng ngần ngại gọi nhiều công cụ thăm dò liên tục để tự xây dựng ngữ cảnh đầy đủ nhất.\n"
            "3. ĐÁNH GIÁ ĐỘ PHỨC TẠP VÀ RA QUYẾT ĐỊNH CHỌN ĐƯỜNG DẪN THỰC THI:\n"
            "   - ĐƯỜNG DẪN A (Thực hiện trực tiếp - DIN): Nếu nguyên nhân cực kỳ đơn giản (ví dụ: chỉ cần sửa đổi "
            "     hoặc bổ sung một vài dòng mã cấu hình đơn giản dưới 10 dòng trong 1 tệp tin), bạn có thể giải thích nguyên nhân "
            "     và không cần gọi đề xuất kế hoạch. Chúng ta sẽ giải quyết nó ở bước tiếp theo.\n"
            "   - ĐƯỜNG DẪN B (Đề xuất kế hoạch - PBE): Nếu lỗi phức tạp, liên quan đến logic nghiệp vụ chính hoặc tác động "
            "     lên nhiều file nguồn, bạn BẮT BUỘC phải gọi công cụ `propose_implementation_plan` để phác thảo Bản kế hoạch "
            "     triển khai (DAG) chi tiết và tạm dừng chờ người dùng duyệt [2].\n"
            "4. Nếu thông tin dự án quá mơ hồ hoặc thiếu file cấu hình thiết yếu, hãy dùng `ask_questions_if_underspecified` để trưng cầu ý kiến người dùng."
        )
        
    else:  # task_type == "development"
        # Đăng ký công cụ can thiệp vật lý (Write-Access Tools)
        read_files = ReadFilesTool(workspace_path=ws)
        write_file = WriteFileTool(workspace_path=ws)
        apply_patch = ApplyPatchTool(workspace_path=ws)
        search_symbols = UniversalSymbolSearchTool(workspace_path=ws)
        list_directory = ListDirectoryTool(workspace_path=ws)
        run_terminal_command = RunTerminalTool(workspace_path=ws)
        read_file_lines = ReadFileLinesTool(workspace_path=ws)
        ask_questions_tool = AskQuestionsTool(workspace_path=ws)
        
        tools = [read_files, write_file, apply_patch, list_directory, run_terminal_command, search_symbols, read_file_lines, ask_questions_tool]

        system_prompt = (
            "Bạn là kỹ sư phần mềm thực thi chuyên nghiệp (Write-Access Mode).\n"
            f"Nhiệm vụ phát triển:\n{tasks_str}\n"
            f"Thư mục làm việc: {ws}\n\n"
            "Hãy áp dụng các bản vá, viết code mới hoặc thực thi kiểm thử tĩnh để hoàn tất kế hoạch đã được phê duyệt.\n"
            "Luôn tuân thủ nguyên tắc Search-and-Replace thông qua `apply_search_replace_patch` đối với các file lớn."
        )

    if workspace_context:
        system_prompt += f"\n\n--- THÔNG TIN NỀN TẢNG THU THẬP ĐƯỢC ---\n{workspace_context}"
    if git_branch and git_branch != "no_git":
        system_prompt += f"\n- Nhánh Git đang hoạt động: `{git_branch}`"
    if registry_context_str:
        system_prompt += registry_context_str

    # Gọi LLM
    model_with_tools = model.bind_tools(tools)
    optimized_history = compact_reading_tool_messages(messages)
    
    input_messages = [SystemMessage(content=system_prompt)]
    if task_type == "development" and error_logs:
        input_messages.append(HumanMessage(content=f"LƯU Ý SỬA LỖI TỪ VÒNG KIỂM THỬ:\n{error_logs}\nHãy sửa triệt để."))
        
    response = model_with_tools.invoke(input_messages + optimized_history)
    response = sanitize_llm_response_content(response)
    
    # ==========================================
    # BƯỚC 3: XỬ LÝ ĐẦU RA AN TOÀN (DEFENSIVE PARSING)
    # ==========================================
    if not response.tool_calls:
        # Nếu đang ở pha phân tích/khảo sát nhưng LLM chọn tự trả lời trực tiếp mà không cần sửa code phức tạp
        if task_type == "analysis":
            findings = []
            if response.content:
                findings = [f"### Báo cáo khảo sát chủ động:\n{response.content}"]
                
            updated_plan = []
            eligible_ids = {t.get("id") if isinstance(t, dict) else getattr(t, "id", None) for t in eligible_tasks}
            for t in plan:
                if isinstance(t, dict):
                    t_copy = dict(t)
                    if t_copy["id"] in eligible_ids:
                        t_copy["status"] = "completed"
                else:
                    t_copy = t.model_copy()
                    if t_copy.id in eligible_ids:
                        t_copy.status = "completed"
                updated_plan.append(t_copy)

            state_updates.update({
                "messages": [response],
                "plan": updated_plan,
                "last_executed_task_ids": list(eligible_ids)
            })
            if findings:
                state_updates["step_findings"] = findings
            return state_updates
        else:
            # Xử lý logic hoàn thành cho pha development (giữ nguyên quy tắc cũ)
            has_executed_action = any(
                getattr(msg, "type", None) == "tool" and msg.name in ["write_file", "apply_search_replace_patch", "run_terminal_command"]
                for msg in reversed(messages)
            )
            content_lower = response.content.lower() if response.content else ""
            explicitly_finished = any(kw in content_lower for kw in ["hoàn thành", "hoàn tất", "done", "finished"])

            if has_executed_action or explicitly_finished:
                updated_plan = []
                eligible_ids = {t.get("id") if isinstance(t, dict) else getattr(t, "id", None) for t in eligible_tasks}
                for t in plan:
                    if isinstance(t, dict):
                        t_copy = dict(t)
                        if t_copy["id"] in eligible_ids:
                            t_copy["status"] = "completed"
                    else:
                        t_copy = t.model_copy()
                        if t_copy.id in eligible_ids:
                            t_copy.status = "completed"
                    updated_plan.append(t_copy)

                state_updates.update({
                    "messages": [response],
                    "plan": updated_plan,
                    "last_executed_task_ids": list(eligible_ids)
                })
                return state_updates
            else:
                warning_feedback = HumanMessage(
                    content="⚠️ Cảnh báo: Bạn đang ở chế độ Phát triển (Development) nhưng chưa thực hiện bất kỳ thay đổi vật lý nào lên file. Hãy dùng write_file hoặc apply_search_replace_patch trước khi hoàn tất."
                )
                state_updates.update({
                    "messages": [response, warning_feedback]
                })
                return state_updates
                
    # Nếu có gọi công cụ, tiếp tục luồng lặp bình thường
    state_updates.update({"messages": [response]})
    return state_updates

def replanner_node(state: AgentState) -> Dict[str, Any]:
    replanning_count = state.get("replanning_count", 0)
    ws = state["workspace_path"]
    plan = state["plan"]
    messages = state["messages"]
    task_type = state.get("task_type", "development")
    workspace_context = state.get("workspace_context", "")
    error_logs = state.get("error_logs", "")
    
    # Xác định xem có phải đang ở pha chuyển tiếp từ thám thính (T_SURVEY) sang phát triển hay không
    is_survey_transition = (
        len(plan) == 1 and 
        (plan[0].id if isinstance(plan[0], Task) else plan[0].get("id")) == "T_SURVEY" and
        (plan[0].status if isinstance(plan[0], Task) else plan[0].get("status")) == "completed"
    )
    
    # Tránh lặp vô hạn nếu vượt ngưỡng hoặc không có lỗi
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
            "⚠️ QUY TẮC THIẾT KẾ KẾ HOẠCH PHÁT TRIỂN & KIỂM THỬ (BẮT BUỘC):\n"
            "1. Đặt `should_modify_plan` là True để áp dụng kế hoạch mới.\n"
            "2. Giữ nguyên nhiệm vụ 'T_SURVEY' với trạng thái là 'completed'.\n"
            "3. Bổ sung các nhiệm vụ mới (ví dụ: T1, T2...) mô tả chính xác các file cần xem, các file cần sửa dựa trên dữ liệu thật thu được từ pha khảo sát.\n"
            "4. TUYỆT ĐỐI BẮT BUỘC phải lập kế hoạch cho nhiệm vụ 'Kiểm thử tích hợp động trên trình duyệt thật' sử dụng công cụ `web_interact_and_test` "
            "để trực tiếp nạp Extension, truy cập trang web đích và xác thực hành vi của Extension làm bước cuối cùng trong kế hoạch!\n"
            "5. Đặt `task_type` là 'development' (vì chúng ta sẽ sửa code và chạy trình duyệt kiểm thử)."
        )
    else:
        system_prompt = (
            "Bạn là một Kiến trúc sư kiêm Điều phối viên dự án phần mềm cấp cao.\n"
            f"Nhiệm vụ: Đánh giá tiến trình thực thi kế hoạch tại thư mục làm việc '{ws}'.\n\n"
            "Hệ thống vừa phát hiện lỗi nghiêm trọng không thể tự gỡ lỗi ở cấp độ cục bộ.\n"
            "Hãy đề xuất một kế hoạch điều chỉnh (được cập nhật trong updated_tasks) để giải quyết triệt để lỗi này.\n\n"
            "⚠️ QUY TẮC CẬP NHẬT KẾ HOẠCH CHO PRODUCTION (BẮT BUỘC):\n"
            "1. Đặt `should_modify_plan` là True và cập nhật danh sách nhiệm vụ trong `updated_tasks` để giải quyết vấn đề.\n"
            "2. ĐỐI VỚI CÁC NHIỆM VỤ ĐÃ HOÀN THÀNH (status: 'completed'): Bắt buộc giữ nguyên ID, mô tả và trạng thái là 'completed'.\n"
            "3. Kế hoạch cập nhật của bạn chỉ tập trung hoàn toàn vào các bước thực thi khảo sát vật lý (analysis) hoặc sửa đổi mã nguồn (development)."
        )
    
    if workspace_context:
        system_prompt += f"\n\n--- NGỮ CẢNH HỆ THỐNG (THONGTIN.md) ---\n{workspace_context}"
        
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
        updated_tasks = getattr(decision, "updated_tasks", plan)
        updated_task_type = getattr(decision, "task_type", task_type)
        
        old_completed_tasks = {
            (t.id if isinstance(t, Task) else t.get("id")): t 
            for t in plan 
            if (t.status if isinstance(t, Task) else t.get("status")) == "completed"
        }
        
        refined_tasks = []
        seen_ids = set()
        for task_data in updated_tasks:
            task_obj = task_data if isinstance(task_data, Task) else Task(**task_data)
            t_id = task_obj.id
            if t_id in seen_ids:
                t_id = f"{t_id}_alt_{len(seen_ids)}"
                task_obj.id = t_id
            seen_ids.add(t_id)
            
            if t_id in old_completed_tasks:
                task_obj.status = "completed"
                old_task = old_completed_tasks[t_id]
                task_obj.description = old_task.description if isinstance(old_task, Task) else old_task.get("description")
            refined_tasks.append(task_obj)
            
        proposal_data = {
            "action": "propose",
            "explanation": explanation,
            "task_type": updated_task_type,
            "tasks": [t.model_dump() if hasattr(t, "model_dump") else t for t in refined_tasks]
        }
        
    except Exception as e:
        # =====================================================================
        # KÍCH HOẠT HỆ THỐNG DỰ PHÒNG CHỦ ĐỘNG (FAIL-SAFE ENGINE)
        # =====================================================================
        # Khi Local LLM qua localhost proxy bị lỗi phân tích cú pháp hoặc mất kết nối,
        # chúng ta tự động dựng lại một Schema PlanUpdate hợp lệ theo hướng kỹ thuật.
        
        fail_safe_explanation = (
            f"⚠️ [Hệ thống tự động kích hoạt chế độ Dự phòng do lỗi gọi LLM Local: {str(e)}]. "
        )
        
        if is_survey_transition:
            # Nếu đang chuyển từ Khảo sát sang Phát triển, bắt buộc phải sinh ra Task sửa code
            fail_safe_explanation += "Tự động thiết lập lộ trình phát triển và kiểm thử tích hợp mặc định."
            fallback_tasks = [
                Task(id="T_SURVEY", description="Khảo sát cấu trúc thư mục và manifest", status="completed"),
                Task(
                    id="T_DEV_FALLBACK", 
                    description="Thực hiện viết/chỉnh sửa mã nguồn trực tiếp trong workspace dựa trên yêu cầu ban đầu của người dùng.", 
                    dependencies=["T_SURVEY"], 
                    status="pending"
                ),
                Task(
                    id="T_TEST_FALLBACK",
                    description="Khởi chạy trình duyệt thật, nạp thử nghiệm Chrome Extension từ ổ đĩa và kiểm tra lỗi console runtime.",
                    dependencies=["T_DEV_FALLBACK"],
                    status="pending"
                )
            ]
            updated_task_type = "development"
        else:
            # Nếu đang chạy sửa lỗi dở dang mà LLM bị sập, giữ nguyên các task cũ để tránh mất mát,
            # đồng thời tiêm thêm một Task mô tả việc sửa lỗi trực tiếp.
            fail_safe_explanation += "Bảo toàn kế hoạch hiện hành và chèn thêm nhiệm vụ sửa đổi trực tiếp."
            fallback_tasks = []
            for t in plan:
                t_obj = t if isinstance(t, Task) else Task(**t)
                fallback_tasks.append(t_obj)
                
            has_pending = any(t.status == "pending" for t in fallback_tasks)
            if not has_pending:
                fallback_tasks.append(
                    Task(
                        id="T_FIX_FALLBACK",
                        description=f"Tiến hành rà soát sửa lỗi biên dịch/runtime phát sinh: {error_logs[:150]}",
                        dependencies=[],
                        status="pending"
                    )
                )
            updated_task_type = task_type

        # Xuất ra dữ liệu có định dạng cấu trúc hoàn hảo như LLM sinh thành công
        proposal_data = {
            "action": "propose",
            "explanation": fail_safe_explanation,
            "task_type": updated_task_type,
            "tasks": [t.model_dump() for t in fallback_tasks]
        }
        
    # Tạo đóng gói phản hồi đồng nhất
    proposal_message = AIMessage(
        content=json.dumps(proposal_data, ensure_ascii=False),
        name="replanner_proposal"
    )
    
    explanation_message = AIMessage(
        content=f"🔄 **[Đề xuất lộ trình hành động]**\n\n{proposal_data['explanation']}\n\nHệ thống đang tiến hành điều phối..."
    )
    
    return {
        "replanning_count": replanning_count + 1,
        "messages": [proposal_message, explanation_message]
    }


def replanner_interrupt_node(state: AgentState) -> Dict[str, Any]:
    messages = state["messages"]
    plan = state["plan"]
    task_type = state.get("task_type", "development")
    
    proposal_msg = None
    for msg in reversed(messages):
        if getattr(msg, "name", None) == "replanner_proposal":
            proposal_msg = msg
            break
            
    if not proposal_msg:
        return {}
        
    try:
        proposal_data = json.loads(proposal_msg.content)
    except Exception:
        return {}
        
    # Xử lý trường hợp chạm giới hạn lập kế hoạch lại (an toàn phòng thủ)
    if proposal_data.get("action") in ["bypass_limit", "bypass_no_error"]:
        # Nếu đã lặp quá 5 lần, giữ nguyên kế hoạch cũ nhưng bắt buộc phải có ít nhất 1 task pending 
        # để tránh việc router đẩy thẳng sang synthesis gây Halt luồng vô ích.
        fallback_tasks = []
        for t in plan:
            fallback_tasks.append(t if isinstance(t, Task) else Task(**t))
            
        has_pending = any(t.status == "pending" for t in fallback_tasks)
        if not has_pending and fallback_tasks:
            # Khôi phục trạng thái của task cuối cùng về pending để tiếp tục sửa chữa
            fallback_tasks[-1].status = "pending"
            
        return {
            "plan": fallback_tasks,
            "error_logs": "",
            "attempts": 0,
            "modified_files": [],
            "messages": [AIMessage(content="🔄 [Bypass Replanner] Đã vượt ngưỡng giới hạn lập kế hoạch. Tiếp tục sửa chữa mã nguồn.")]
        }
        
    payload = {
        "title": "📋 ĐÁNH GIÁ & PHÊ DUYỆT KẾ HOẠCH HÀNH ĐỘNG",
        "explanation": proposal_data["explanation"],
        "proposed_tasks": proposal_data["tasks"],
        "prompt": (
            "Hệ thống đề xuất điều chỉnh lộ trình như trên.\n"
            "- Gửi phản hồi 'yes' hoặc rỗng để ĐỒNG Ý áp dụng kế hoạch mới.\n"
            "- Gửi phản hồi 'skip' hoặc 'no' để BỎ QUA việc lập kế hoạch lại.\n"
        )
    }
    
    # Kích hoạt ngắt đồ thị chờ duyệt (hoặc tự động lấy input nếu chạy CLI không tương tác)
    user_input = interrupt(payload)
    
    if isinstance(user_input, str):
        user_input_clean = user_input.strip().lower()
        
        if user_input_clean in ["skip", "no", "cancel"]:
            # Nếu người dùng từ chối đổi kế hoạch, ta vẫn giữ kế hoạch cũ nhưng phải đảm bảo có task pending
            fallback_tasks = [t if isinstance(t, Task) else Task(**t) for t in plan]
            if not any(t.status == "pending" for t in fallback_tasks) and fallback_tasks:
                fallback_tasks[-1].status = "pending"
            return {
                "plan": fallback_tasks,
                "error_logs": "",           
                "attempts": 0,
                "modified_files": [],
                "messages": [AIMessage(content="⏭️ **[Người dùng bỏ qua kế hoạch mới]** Tiếp tục lộ trình thực thi hiện tại.")]
            }
            
        elif user_input_clean in ["yes", "approve", "ok", ""]:
            refined_tasks = [Task(**t) for t in proposal_data["tasks"]]
            return {
                "plan": refined_tasks,
                "task_type": proposal_data["task_type"],
                "error_logs": "",
                "attempts": 0,
                "modified_files": [],
                "messages": [AIMessage(content="✅ **[Kế hoạch được duyệt]** Áp dụng lộ trình phát triển mới thành công.")]
            }
            
        else:
            # Xử lý JSON tự nhập từ người dùng
            try:
                parsed_tasks = json.loads(user_input)
                if isinstance(parsed_tasks, list):
                    custom_tasks = [Task(**t) for t in parsed_tasks]
                    return {
                        "plan": custom_tasks,
                        "error_logs": "",
                        "attempts": 0,
                        "modified_files": [],
                        "messages": [AIMessage(content="✏️ **[Kế hoạch tùy chỉnh]** Đã áp dụng danh sách nhiệm vụ của bạn.")]
                    }
            except Exception:
                pass
            
    # Mặc định tự động duyệt (Auto-approve) khi chạy không tương tác
    refined_tasks = [Task(**t) for t in proposal_data["tasks"]]
    return {
        "plan": refined_tasks,
        "task_type": proposal_data["task_type"],
        "error_logs": "",
        "attempts": 0,
        "modified_files": [],
        "messages": [AIMessage(content="✅ **[Tự động duyệt]** Đồng ý kế hoạch điều chỉnh.")]
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
    ask_questions_tool = AskQuestionsTool(workspace_path=ws) # ĐÃ THÊM KHỞI TẠO TRONG TOOL_NODE
    
    tools_map = {
        "read_files": read_files,
        "write_file": write_file,
        "apply_search_replace_patch": apply_patch,
        "list_directory": list_directory,
        "run_terminal_command": run_terminal_command,
        "search_symbols_universal": search_symbols,
        "read_file_lines": read_file_lines,
        "web_interact_and_test": web_interact_tool,
        "ask_questions_if_underspecified": ask_questions_tool, # ĐÃ ĐĂNG KÝ VÀO THƯ VIỆN THỰC THI
    }
    
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {}
        
    tool_messages = []
    modified_files = list(state.get("modified_files", []))
    file_registry = dict(state.get("file_registry", {}))
    
    impacted_files = set()
    
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"] or {}
        tool_id = tool_call["id"]
        
        # Chỉ nạp vào Registry các file bị can thiệp bởi công cụ Ghi/Sửa hoặc Đọc toàn bộ
        if tool_name in ["write_file", "apply_search_replace_patch", "read_files"]:
            raw_path = tool_args.get("file_path") or tool_args.get("file_paths")
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
                result = tool_instance.invoke(tool_args)
                if tool_name in ["write_file", "apply_search_replace_patch"] and "Lỗi" not in str(result):
                    if raw_path and not isinstance(raw_path, list):
                        try:
                            safe_path = sanitize_and_resolve_path(ws, raw_path, create_parent=True)
                            if str(safe_path) not in modified_files:
                                modified_files.append(str(safe_path))
                        except Exception:
                            pass
            except (GraphInterrupt, GraphBubbleUp) as g:
                raise g
            except Exception as e:
                result = f"Lỗi thực thi công cụ '{tool_name}': {str(e)}"
                
                
        tool_messages.append(ToolMessage(content=str(result), name=tool_name, tool_call_id=tool_id))
        
    for file_path in impacted_files:
        try:
            safe_path = sanitize_and_resolve_path(ws, file_path, create_parent=False)
            if safe_path.exists() and safe_path.is_file():
                current_content = safe_path.read_text(encoding="utf-8")
                file_registry[file_path] = current_content
        except Exception:
            pass
            
    return {
        "messages": tool_messages,
        "modified_files": modified_files,
        "file_registry": file_registry
    }


def tester_node(state: AgentState) -> Dict[str, Any]:
    modified_files = state.get("modified_files", [])
    attempts = state.get("attempts", 0)
    plan = state["plan"]
    ws = state["workspace_path"]
    last_executed_ids = state.get("last_executed_task_ids", [])
    
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
            for t in plan:
                if isinstance(t, dict):
                    t_copy = dict(t)
                    if t_copy["id"] in last_executed_ids:
                        t_copy["status"] = "pending"
                else:
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
                "error_logs": "",
                "attempts": 0,
                "modified_files": [],
                "messages": [AIMessage(content=f"{warning_msg}❌ Đã vượt quá giới hạn số lần sửa lỗi tự động. Hệ thống sẽ bỏ qua lỗi để tiếp tục tiến trình.")]
            }
                
    success_content = f"{warning_msg}✅ [Vòng kiểm thử thành công] Toàn bộ mã nguồn đã vượt qua kiểm tra tĩnh và biên dịch."
    return {
        "error_logs": "",
        "attempts": 0,
        "modified_files": [],
        "messages": [AIMessage(content=success_content)]
    }


def synthesis_node(state: AgentState) -> Dict[str, Any]:
    ws = state["workspace_path"]
    findings = state.get("step_findings", [])
    git_branch = state.get("git_branch", "no_git")
    
    if not findings:
        return {"messages": [AIMessage(content="Không thu thập được thông tin khảo sát để tổng hợp.")]}
        
    compiled_data = "\n\n---\n\n".join(findings)
    
    synthesis_prompt = (
        "Bạn là một Kiến trúc sư Hệ thống chuyên nghiệp chuyên biên soạn tài liệu.\n"
        "Hãy tổng hợp toàn bộ thông tin khảo sát thô được ghi nhận ở các bước trước thành một tài liệu 'THONGTIN.md' duy nhất.\n"
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
                tools_mgr.write_file("THONGTIN.md", cleaned_md)
                
                git_manager = GitManager(ws)
                git_manager._run_cmd(["git", "add", "THONGTIN.md"], ignore_error=True)
                
                message_content = (
                    "**Tổng hợp tài liệu hoàn tất:** Đã biên dịch tri thức khảo sát, "
                    "lưu vật lý thành tệp `THONGTIN.md` và đưa vào Git staging thành công."
                )
            else:
                message_content = (
                    "**Tổng hợp tài liệu hoàn tất (Chế độ In-Memory):** Tri thức khảo sát "
                    "đã được tổng hợp và nạp trực tiếp vào ngữ cảnh trạng thái đồ thị (`workspace_context`). "
                    "Tệp tin `THONGTIN.md` vật lý **không** được tạo trên ổ đĩa do hệ thống phát hiện không sử dụng Git."
                )
            
            return {
                "workspace_context": cleaned_md,
                "messages": [AIMessage(content=message_content)]
            }
    except Exception as e:
        return {"messages": [AIMessage(content=f"Cảnh báo: Có lỗi xảy ra khi tổng hợp tệp THONGTIN.md: {str(e)}")]}
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