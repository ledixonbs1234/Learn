# oder/nodes.py
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
from venv import logger
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langgraph.types import interrupt
from config import find_project_root_heuristic, model, sanitize_and_resolve_path, fast_model
from mcp_helper import run_agent_with_devtools_mcp
from skills_engine import AgentSkillsEngine
from state import AgentState, PlanUpdate, RuntimeVerificationResult, TaskTriage, Task
from tools import (
    ActivateSkillTool, AskQuestionsTool, GitManager, ReadFileLinesTool, RunSkillScriptTool, SearchKeywordTool, UniversalSymbolSearchTool, WebInteractAndTestTool, WorkspaceTools, 
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


def verify_workspace_safety(workspace_path: str, allow_explicit: bool = False) -> bool:
    """
    Hệ thống phòng thủ an toàn (Security Guardrail):
    Ngăn chặn việc Agent trỏ Workspace vào thư mục nguồn của chính nó,
    trừ khi người dùng chủ động yêu cầu phân tích mã nguồn của chính Agent.
    """
    if allow_explicit:
        return True
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

def triage_node_stateful(state: AgentState, catalog_summary: str) -> TaskTriage:
    """
    LLM Triage thông minh có trạng thái và có tri thức về thư viện Kỹ năng.
    """
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
        "     nhưng không nói ở đâu -> Đặt task_type = 'clarify' để hệ thống hỏi lại đường dẫn mới."
        "⚠️ ÁNH XẠ KỸ NĂNG CHỦ ĐỘNG (BẮT BUỘC):\n"
        "Dưới đây là danh sách các Kỹ năng kỹ thuật (Skills) khả dụng có sẵn trong hệ thống.\n"
        "Nhiệm vụ cực kỳ quan trọng của bạn là đối chiếu yêu cầu hiện tại của người dùng với mô tả và điều kiện kích hoạt (triggers) của từng Kỹ năng dưới đây.\n"
        "Nếu yêu cầu của người dùng khớp với mục đích của kỹ năng nào, bạn BẮT BUỘC phải điền chính xác tên kỹ năng đó (ví dụ: 'test-driven-development') vào trường 'recommended_skills'.\n"
        f"{catalog_summary}"

    )

    structured_llm = model.with_structured_output(TaskTriage, method="function_calling")
    
    triage_output = structured_llm.invoke([
        SystemMessage(content=system_prompt),
        SystemMessage(content=active_session_context),
        *context_messages,
        HumanMessage(content=f"Yêu cầu hiện tại của người dùng: {user_query_text}")
    ])
    
    return triage_output

# oder/nodes.py

def detect_and_triage_node(state: AgentState) -> Dict[str, Any]:
    """
    Nút phân loại và thiết lập môi trường hoạt động thông minh có kế thừa trạng thái.
    CẢI TIẾN: Tự động nạp trước các kỹ năng Grilling & PRD ngay tại đầu nguồn (Triage)
    và cấu hình động nhiệm vụ T_SURVEY để thực thi phỏng vấn trước khi lập kế hoạch.
    """
    messages = state["messages"]
    user_msg = messages[-1]
    user_query_text = get_text_content_safely(user_msg.content)
    
    existing_workspace = state.get("workspace_path", ".")

    # =====================================================================
    # BƯỚC 1: TRÍCH XUẤT ĐƯỜNG DẪN CO ĐỘ ƯU TIÊN (PRECEDENCE)
    # =====================================================================
    detected_path_str = extract_path_from_text(user_query_text)
    if not detected_path_str:
        path_keywords = ["thư mục", "folder", "dự án", "project", "mở", "quét", "ls", "dir", "làm việc tại", "tại", "cd"]
        if any(kw in user_query_text.lower() for kw in path_keywords) or not existing_workspace:
            detected_path_str = resolve_special_system_paths(user_query_text)

    # =====================================================================
    # BƯỚC 2: XÁC ĐỊNH NGỮ CẢNH WORKSPACE TẠM THỜI
    # =====================================================================
    provisional_workspace = existing_workspace
    pivoted_msg = ""

    if detected_path_str:
        try:
            resolved_path = Path(detected_path_str).expanduser().resolve()
            if resolved_path.exists():
                provisional_workspace = str(find_project_root_heuristic(resolved_path))
                if provisional_workspace != existing_workspace and existing_workspace:
                    pivoted_msg = f"🔄 **[Chuyển đổi Workspace]**: Phát hiện yêu cầu chuyển đổi thư mục làm việc sang: `{provisional_workspace}`\n"
        except Exception: pass
    else:
        if existing_workspace:
            pivoted_msg = f"🔄 **[Kế thừa Workspace]**: Sử dụng lại thư mục làm việc hiện hành: `{existing_workspace}`\n"

    # =====================================================================
    # BƯỚC 3: ĐỒNG BỘ TRẠNG THÁI & QUÉT THƯ VIỆN KỸ NĂNG VẬT LÝ
    # =====================================================================
    temp_state = state.copy()
    temp_state["workspace_path"] = provisional_workspace
    
    skills_engine = AgentSkillsEngine(provisional_workspace)
    catalog = skills_engine.scan_catalog()
    
    catalog_summary = ""
    if catalog:
        catalog_summary = "\n=== 📚 DANH MỤC KỸ NĂNG HỆ THỐNG HIỆN CÓ ===\n"
        for item in catalog:
            catalog_summary += f"- Kỹ năng: `{item['name']}`\n  Điều kiện kích hoạt: {item['description']}\n"

    if provisional_workspace and provisional_workspace != existing_workspace:
        temp_ext = find_extension_dir_heuristic(Path(provisional_workspace))
        temp_state["extension_path"] = temp_ext or ""

    # Gọi Triage Supervisor
    try:
        triage_output = triage_node_stateful(temp_state, catalog_summary)
        task_type = triage_output.task_type
        is_simple = triage_output.is_simple
        detailed_analysis = triage_output.detailed_analysis
        recommended_skills = triage_output.recommended_skills or []
    except Exception as e:
        task_type = "clarify" if not provisional_workspace else "analysis"
        is_simple = True
        detailed_analysis = f"Lỗi hệ thống phân loại: {str(e)}"
        recommended_skills = []

    # Rào chắn an toàn
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
    if not verify_workspace_safety(final_workspace, allow_explicit=is_user_explicit):
        return {
            "plan": [], "task_type": "analysis", "is_simple": True,
            "messages": [AIMessage(content="🚨 **[CẢNH BÁO BẢO MẬT]**: Workspace nằm trong thư mục Agent.")]
        }

    # =====================================================================
    # CẢI TIẾN 1: TỰ ĐỘNG NẠP TRƯỚC CÁC KỸ NĂNG PHỤC VỤ PHA KHẢO SÁT (BOOTSTRAPPING)
    # =====================================================================
# BƯỚC 4: TỰ ĐỘNG KHỞI TẠO VÀ NẠP ĐÓNG GÓI BỘ BA KỸ NĂNG (TRIAD BUNDLING)
    # =====================================================================
    active_skills = {}
    skills_engine = AgentSkillsEngine(final_workspace)
    
    # Định nghĩa bộ ba kỹ năng nghiệp vụ bắt buộc đi cùng nhau
    analysis_triad = ["grill-with-docs", "write-a-prd", "domain-modeling"]
    
    # Kiểm tra xem Supervisor có đề xuất bất kỳ kỹ năng nào trong bộ ba này không
    requires_triad = any(skill in recommended_skills for skill in analysis_triad)
    
    if requires_triad:
        # Tự động nạp toàn bộ bộ ba để Agent có đầy đủ quy trình và định dạng tệp tin
        for skill_name in analysis_triad:
            body = skills_engine.load_skill_body(skill_name)
            if body:
                active_skills[skill_name] = body
                
    # Nạp các kỹ năng kỹ thuật khác được đề xuất ngoài bộ ba trên
    for skill_name in recommended_skills:
        if skill_name not in active_skills:
            body = skills_engine.load_skill_body(skill_name)
            if body:
                active_skills[skill_name] = body

    # =====================================================================
    # BƯỚC 5: THIẾT LẬP KẾ HOẠCH DỰA TRÊN KỸ NĂNG ĐỀ XUẤT (DYNAMIC TASK SPEC)
    # =====================================================================
    plan = []
    if is_simple:
        plan = [
            Task(id="T1", description=f"Thực hiện trực tiếp tác vụ tại `{final_workspace}`: {user_query_text}", dependencies=[], status="pending")
        ]
    else:
        # Nếu bộ ba Grilling được kích hoạt, ép buộc Agent phải hoàn tất phỏng vấn và sinh PRD trong pha khảo sát
        if requires_triad:
            survey_desc = (
                f"Sử dụng kỹ năng `grill-with-docs` để thực hiện phiên phỏng vấn/chất vấn không khoan nhượng nhằm "
                f"làm rõ yêu cầu về: {user_query_text}. Đồng thời cập nhật bảng thuật ngữ (`CONTEXT.md`) sử dụng kỹ năng `domain-modeling`, "
                f"tạo các quyết định kiến trúc (ADRs) và đúc kết thành tệp `PRD.md` bằng skill `write-a-prd`."
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

    triage_info_msg = (
        f"📊 **[Hệ thống Phân phối thông minh]**:\n"
        f"{pivoted_msg}"
        f"- **Workspace hoạt động:** `{final_workspace}`\n"
        f"- **Chế độ kiểm soát:** {'Đơn giản (Fast-Track)' if is_simple else 'Phức tạp (Multi-Step Discovery)'}\n"
        f"- **Pha hoạt động khởi động:** `{task_type.upper()}`\n"
        f"{skills_log_msg}\n"
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
        "active_skills": active_skills, # Đã có sẵn chỉ dẫn Grilling trong state!
        "recommended_skills": recommended_skills
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
    """
    [NODE MỚI] Thực hiện rà soát đơn mô hình đối kháng (Step 3: DOUBT trong SKILL.md).
    Chỉ chạy khi mã nguồn đã được sửa đổi và vượt qua các kiểm tra cú pháp tĩnh thành công.
    """
    modified_files = state.get("modified_files", [])
    file_registry = state.get("file_registry", {})
    doubt_attempts = state.get("doubt_attempts", 0)
    
    if not modified_files or doubt_attempts >= 3:
        return {"doubt_findings": ""}
        
    latest_file = modified_files[-1]
    artifact_code = file_registry.get(latest_file, "")
    
    if not artifact_code:
        # Nếu chưa nạp code vào registry, thử đọc từ đĩa vật lý
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
    """
    [NODE MỚI] Nút ngắt tương tác (Human-in-the-Loop) của Doubt-Driven Development.
    Hiển thị các phát hiện lỗi cho người dùng, cho phép kích hoạt thẩm định chéo qua mô hình phụ (Gemini/Codex)
    hoặc phê duyệt/yêu cầu sửa code trực tiếp.
    """
    doubt_findings = state.get("doubt_findings", "")
    modified_files = state.get("modified_files", [])
    
    if not doubt_findings or not modified_files:
        return {}

    latest_file = modified_files[-1]

    # Thiết lập payload giao diện cho interrupt
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

    # KÍCH HOẠT NGẮT ĐỒ THỊ ĐỂ ĐỢI PHẢN HỒI TỪ NGƯỜI DÙNG
    user_input = interrupt(interrupt_payload)
    
    user_action = str(user_input).strip().lower() if user_input else ""

    # Kịch bản 1: Người dùng phê duyệt/Bỏ qua (Bypass)
    if user_action in ["", "yes", "approve", "ok", "skip"]:
        return {
            "error_logs": "",
            "doubt_findings": "", # Xóa log nghi ngờ để đi tiếp
            "messages": [AIMessage(content="✅ **[Doubt Bypassed]** Người dùng đã phê duyệt mã nguồn. Tiến hành hoàn tất tác vụ.")]
        }

    # Kịch bản 2: Yêu cầu thẩm định chéo qua CLI ngoại vi (Cross-Model Escalation)
    if user_action in ["gemini", "codex"]:
        cli_tool = user_action
        cli_executable = "gemini" if cli_tool == "gemini" else "codex"
        
        # Kiểm tra sự tồn tại vật lý của CLI trong môi trường hệ thống
        if not shutil.which(cli_executable):
            feedback_msg = HumanMessage(
                content=f"⚠️ Lỗi: Không tìm thấy thực thi CLI `{cli_executable}` trong biến môi trường PATH của bạn. Vui lòng kiểm tra lại cấu hình."
            )
            return {
                "error_logs": f"Không tìm thấy công cụ ngoại vi `{cli_executable}`",
                "messages": [feedback_msg]
            }

        # Chuẩn bị Prompt Thẩm định đối kháng chéo
        file_registry = state.get("file_registry", {})
        artifact_code = file_registry.get(latest_file, "")
        
        cross_prompt = (
            f"Thẩm định đối kháng chéo (Adversarial Cross-Model Review) cho file {latest_file}.\n"
            "Hãy tìm ra các lỗ hổng, lỗi logic hoặc điểm chưa tối ưu mà mô hình trước đã bỏ qua.\n\n"
            "MÃ NGUỒN:\n"
            f"{artifact_code}"
        )

        # 🛡️ PHÒNG THỦ AN TOÀN TUYỆT ĐỐI CHỐNG SHELL INJECTION (Không truyền trực tiếp qua argument)
        # Ghi prompt đối kháng ra tệp tạm và truyền dữ liệu thông qua Standard Input (stdin)
        try:
            with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8", delete=False) as temp_prompt_file:
                temp_prompt_file.write(cross_prompt)
                temp_prompt_file_path = temp_prompt_file.name

            # Thực thi tiến trình con an toàn với chế độ sandbox hoặc luồng an toàn
            # Cú pháp chạy: cli_executable < temp_prompt_file_path
            with open(temp_prompt_file_path, "r", encoding="utf-8") as stdin_file:
                # Thiết lập biến môi trường UTF-8 đồng bộ cho tiến trình con
                env_copy = os.environ.copy()
                env_copy["PYTHONIOENCODING"] = "utf-8"
                env_copy["PYTHONUTF8"] = "1"
                
                cmd = [cli_executable]
                # Thêm cờ bổ sung nếu là gemini để chạy không tương tác
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

            # Dọn dẹp tệp tạm vật lý ngay lập tức
            Path(temp_prompt_file_path).unlink(missing_ok=True)

            combined_output = (res.stdout or "") + "\n" + (res.stderr or "")
            escalated_findings = combined_output.strip()
            
            # Ghi nhận kết quả rà soát mới và giữ lại trạng thái ngắt để hiển thị tiếp cho người dùng
            return {
                "doubt_findings": f"🛡️ **[KẾT QUẢ THẨM ĐỊNH CHÉO TỪ {cli_executable.upper()}]**:\n\n{escalated_findings}",
                "messages": [AIMessage(content=f"🔍 Kích hoạt thành công rà soát chéo từ {cli_executable.upper()}. Đang chờ ý kiến phê duyệt cuối cùng.")]
            }
            
        except Exception as err:
            return {
                "error_logs": f"Lỗi hệ thống khi khởi chạy Cross-Model: {str(err)}",
                "messages": [AIMessage(content=f"❌ Thao tác gọi mô hình chéo thất bại: {str(err)}")]
            }

    # Kịch bản 3: Phản hồi tự do yêu cầu sửa lỗi (Reconcile Path)
    feedback_message = HumanMessage(
        content=(
            "⚠️ Yêu cầu sửa đổi mã nguồn dựa trên kết quả thẩm định đối kháng:\n"
            f"Ý kiến người dùng: '{user_input}'\n"
            f"Các lỗi cần khắc phục:\n{doubt_findings}"
        )
    )
    
    return {
        "error_logs": f"Cần khắc phục lỗi logic thẩm định: {user_input}",
        "doubt_findings": "", # Xóa log để chuẩn bị lượt kiểm tra mới sau khi sửa xong
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

# =====================================================================
# THÊM NÚT DỌN DẸP NGỮ CẢNH (CONTEXT COMPRESSOR NODE) VÀO CUỐI FILE
# =====================================================================
def context_compressor_node(state: AgentState) -> Dict[str, Any]:
    """
    [NODE THU GỌN CONTEXT CÓ RÀO CHẮN AN TOÀN]
    Chỉ thực hiện xóa tin nhắn (CGC) khi phiên Grilling thực sự diễn ra và thành công.
    Nếu chỉ là khảo sát kỹ thuật thông thường (không grilling), nút này sẽ giữ nguyên
    toàn bộ lịch sử tin nhắn thám thính và chuyển giao nguyên vẹn cho Planner.
    """
    messages = state.get("messages", [])
    ws = state["workspace_path"]
    active_skills = state.get("active_skills", {}) or {}
    workspace_root = Path(ws).expanduser().resolve()
    
    # =====================================================================
    # 1. KIỂM TRA ĐIỀU KIỆN KÍCH HOẠT THỰC TẾ (SAFE-GUARD CHECK)
    # =====================================================================
    # Kiểm tra xem Agent có thực sự gọi công cụ kích hoạt Grilling/PRD hay không
    grilling_activated = "grill-with-docs" in active_skills or "write-a-prd" in active_skills
    
    # 2. ĐỒNG BỘ TOÀN BỘ TÀI LIỆU VẬT LÝ VÀO STATE
    compiled_context_parts = []
    
    # Thử đọc PRD.md hoặc THONGTIN.md
    prd_path = workspace_root / "PRD.md"
    thongtin_path = workspace_root / "THONGTIN.md"
    if prd_path.exists():
        try:
            compiled_context_parts.append(f"### [PRODUCT REQUIREMENTS DOCUMENT (PRD.md)]\n{prd_path.read_text(encoding='utf-8')}")
        except Exception: pass
    elif thongtin_path.exists():
        try:
            compiled_context_parts.append(f"### [THÔNG TIN DỰ ÁN (THONGTIN.md)]\n{thongtin_path.read_text(encoding='utf-8')}")
        except Exception: pass

    # Đọc CONTEXT.md (Glossary)
    context_file_path = workspace_root / "CONTEXT.md"
    if context_file_path.exists():
        try:
            compiled_context_parts.append(f"### [BẢNG THUẬT NGỮ NGHIỆP VỤ (CONTEXT.md)]\n{context_file_path.read_text(encoding='utf-8')}")
        except Exception: pass

    # Đọc danh sách các quyết định kiến trúc (ADRs)
    adr_dir = workspace_root / "docs" / "adr"
    if adr_dir.exists() and adr_dir.is_dir():
        adr_texts = []
        try:
            for adr_file in sorted(adr_dir.glob("*.md")):
                adr_texts.append(f"#### Tệp {adr_file.name}:\n{adr_file.read_text(encoding='utf-8')}")
            if adr_texts:
                compiled_context_parts.append("### [QUYẾT ĐỊNH KIẾN TRÚC (ARCHITECTURAL DECISIONS - ADRs)]\n" + "\n\n".join(adr_texts))
        except Exception: pass

    super_context = "\n\n---\n\n".join(compiled_context_parts)
    if not super_context:
        super_context = state.get("workspace_context", "")

    # =====================================================================
    # 3. ĐIỀU HƯỚNG BẢO VỆ CONTEXT (SAFE-GUARD RULE)
    # =====================================================================
    # Nếu không có grilling thực sự, HOẶC không có file PRD/Glossary vật lý nào được tạo ra:
    # -> BỎ QUA VIỆC XÓA TIN NHẮN để tránh mất dữ liệu khảo sát thô.
    if not grilling_activated or not (prd_path.exists() or context_file_path.exists()):
        return {
            "workspace_context": super_context
            # Không trả về deletion_list, toàn bộ lịch sử tin nhắn được bảo toàn nguyên vẹn
        }

    # =====================================================================
    # 4. THỰC HIỆN DỌN DẸP KHI ĐỦ ĐIỀU KIỆN (CHỈ KHI CÓ GRILLING THÀNH CÔNG)
    # =====================================================================
    if len(messages) <= 2:
        return {"workspace_context": super_context}
        
    deletion_list = []
    for msg in messages[1:]:
        if msg.id:
            deletion_list.append(RemoveMessage(id=msg.id)) # Đánh dấu xóa tin nhắn [2]
            
    clean_checkpoint_msg = AIMessage(
        content=(
            "🔄 **[Hệ thống dọn dẹp Ngữ cảnh & Đồng bộ Domain Model]**:\n"
            "Phát hiện phiên đối thoại chất vấn nghiệp vụ (Grilling) đã diễn ra thành công.\n"
            "Hệ thống đã dọn dẹp lịch sử tin nhắn thô để tiết kiệm token, giải phóng các kỹ năng "
            "khảo sát (Garbage Collection) và đồng bộ hóa tài liệu "
            "(PRD, CONTEXT.md, ADRs) vào bộ nhớ ngữ cảnh của Graph."
        )
    )
    
    # Thực hiện thu gom rác ngữ cảnh cho các kỹ năng đã hoàn thành nhiệm vụ
    cleaned_active_skills = dict(active_skills)
    discovery_triad = ["grill-with-docs", "write-a-prd", "domain-modeling"]
    for skill_name in discovery_triad:
        cleaned_active_skills.pop(skill_name, None)
    
    return {
        "workspace_context": super_context,
        "messages": deletion_list + [clean_checkpoint_msg],
        "active_skills": cleaned_active_skills
    }
# THAY THẾ ĐOẠN CODE TRONG oder/nodes.py BẰNG ĐOẠN DƯỚI ĐÂY

def executor_node(state: AgentState) -> Dict[str, Any]:
    ws = state["workspace_path"]
    plan = state.get("plan", [])
    error_logs = state.get("error_logs", "")
    file_registry = state.get("file_registry", {})
    messages = list(state["messages"])
    task_type = state.get("task_type", "development")
    extension_path = state.get("extension_path", "")
    active_skills = state.get("active_skills", {}) or {}

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

    if not extension_path:
        ext_dir = find_extension_dir_heuristic(Path(ws))
        if ext_dir:
            extension_path = ext_dir
            state_updates["extension_path"] = extension_path

    # Ép kiểu phòng thủ từ dict thô về Task object
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

    skills_engine = AgentSkillsEngine(ws)
    catalog = skills_engine.scan_catalog()
    
    catalog_prompt = ""
    if catalog:
        catalog_prompt = "\n=== 📚 THƯ VIỆN KỸ NĂNG KHẢ DỤNG (TIER 1: CATALOG) ===\n"
        for item in catalog:
            catalog_prompt += f"- **{item['name']}**: {item['description']}\n"

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
        active_skills_prompt = "\n=== ⚡ CÁC KỸ NĂNG ĐANG HOẠT ĐỘNG (TIER 2: INSTRUCTIONS) ===\n"
        for s_name, s_instructions in active_skills.items():
            active_skills_prompt += f"\n--- CHỈ DẪN KỸ NĂNG `{s_name}` ---\n{s_instructions}\n"

    activate_skill_tool = ActivateSkillTool(workspace_path=ws)
    run_skill_script_tool = RunSkillScriptTool(workspace_path=ws)
    search_keyword_tool = SearchKeywordTool(workspace_path=ws)

    if task_type == "analysis":
        # PHA KHẢO SÁT: Đọc hiểu mã nguồn, phỏng vấn nghiệp vụ và thiết lập tài liệu đặc tả (PRD, CONTEXT, ADRs)
        read_files = ReadFilesTool(workspace_path=ws)
        list_directory = ListDirectoryTool(workspace_path=ws)
        search_symbols = UniversalSymbolSearchTool(workspace_path=ws)
        read_file_lines = ReadFileLinesTool(workspace_path=ws)
        ask_questions_tool = AskQuestionsTool(workspace_path=ws)
        
        # Cấp thêm quyền ghi tệp tin tài liệu nghiệp vụ
        write_file = WriteFileTool(workspace_path=ws)
        apply_patch = ApplyPatchTool(workspace_path=ws)
        
        tools = [
            activate_skill_tool, 
            run_skill_script_tool, 
            read_files, 
            list_directory, 
            search_symbols, 
            read_file_lines, 
            ask_questions_tool, 
            search_keyword_tool,
            write_file,
            apply_patch
        ]
        
        system_prompt = (
            "Bạn là một chuyên gia khảo sát mã nguồn, phỏng vấn nghiệp vụ và thiết lập mô hình miền (Active Discovery & Glossary Engine).\n"
            f"Nhiệm vụ hiện tại:\n{tasks_str}\n"
            f"Thư mục làm việc: {ws}\n\n"
            "⚠️ QUY TẮC THIẾT LẬP TÀI LIỆU & KHẢO SÁT (BẮT BUỘC):\n"
            "1. Bạn có quyền đọc mã nguồn và viết/cập nhật các tệp tài liệu đặc tả quan trọng như `CONTEXT.md`, các quyết định kiến trúc (ADRs) trong `docs/adr/`, và `PRD.md`.\n"
            "   TUYỆT ĐỐI KHÔNG sửa đổi các tệp tin mã nguồn chạy thật (code) của ứng dụng trong pha khảo sát này.\n"
            "2. Nếu bạn cần tìm kiếm vị trí của một biến hoặc hàm, hãy ưu tiên dùng `search_keyword`.\n"
            "3. Khi đã xác định được tệp tin cần quan tâm, hãy dùng `read_file_lines` để đọc phân đoạn thay vì đọc cả file lớn.\n"
            "4. Thúc đẩy tiến trình phỏng vấn không khoan nhượng (Grilling): hãy tiếp tục gọi `ask_questions_if_underspecified` nếu các "
            "phương án kỹ thuật chưa được làm rõ tuyệt đối. Chỉ hoàn tất nhiệm vụ khảo sát khi đã ghi đầy đủ tài liệu đặc tả vật lý xuống đĩa.\n"
            "5. Khi hoàn tất toàn bộ đặc tả tài liệu, kết thúc lượt bằng một văn bản tổng hợp kết quả điều tra (không gọi thêm công cụ). Hệ thống sẽ tự động chuyển tiếp tới pha lập kế hoạch."
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
            activate_skill_tool, run_skill_script_tool, read_files, write_file, 
            apply_patch, list_directory, run_terminal_command, search_symbols, 
            read_file_lines, ask_questions_tool, write_and_run_script, search_keyword_tool
        ]
        
        system_prompt = (
            "Bạn là kỹ sư phần mềm thực thi chuyên nghiệp (Write-Access Mode).\n"
            f"Nhiệm vụ phát triển:\n{tasks_str}\n"
            f"Thư mục làm việc: {ws}\n\n"
            "⚠️ HƯỚNG DẪN TIẾT KIỆM TOKEN:\n"
            "Ưu tiên sử dụng `apply_search_replace_patch` thay vì ghi đè lại toàn bộ tệp tin lớn bằng `write_file`.\n"
        )

    system_prompt += catalog_prompt + active_skills_prompt
    if workspace_context:
        system_prompt += f"\n\n--- THÔNG TIN NỀN TẢNG THU THẬP ĐƯỢC ---\n{workspace_context}"
    if git_branch and git_branch != "no_git":
        system_prompt += f"\n- Nhánh Git đang hoạt động: `{git_branch}`"
    if registry_context_str:
        system_prompt += registry_context_str

    model_with_tools = model.bind_tools(tools)
    optimized_history = messages 

    input_messages = [SystemMessage(content=system_prompt)]
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
                    # Hoàn thành nhiệm vụ lính canh T_SURVEY
                    t_copy.status = "completed"
                updated_plan.append(t_copy)

            state_updates.update({
                "messages": [response],
                "plan": updated_plan,
                "last_executed_task_ids": list(eligible_ids),
                "active_skills": active_skills
            })
            if findings:
                state_updates["step_findings"] = findings
            return state_updates
        else:
            has_executed_action = any(
                getattr(msg, "type", None) == "tool" and msg.name in ["write_file", "apply_search_replace_patch", "run_terminal_command", "run_skill_script"]
                for msg in reversed(messages)
            )
            content_lower = response.content.lower() if response.content else ""
            explicitly_finished = any(kw in content_lower for kw in ["hoàn thành", "hoàn tất", "done", "finished"])

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
                    "active_skills": active_skills
                })
                return state_updates
            else:
                warning_feedback = HumanMessage(
                    content="⚠️ Cảnh báo: Bạn chưa thực hiện chỉnh sửa nào lên file hoặc kích hoạt script. Hãy ghi file hoặc vá code trước khi hoàn tất."
                )
                state_updates.update({
                    "messages": [response, warning_feedback],
                    "active_skills": active_skills
                })
                return state_updates
                
    state_updates.update({
        "messages": [response],
        "active_skills": active_skills
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
    recommended_skills = state.get("recommended_skills", [])
    active_skills = dict(state.get("active_skills", {}))
    skills_engine = AgentSkillsEngine(ws)
    loaded_skills_list = []
    
    for skill_name in recommended_skills:
        if skill_name not in active_skills:
            body = skills_engine.load_skill_body(skill_name)
            if body:
                active_skills[skill_name] = body
                loaded_skills_list.append(f"`{skill_name}`")
                
    # Xây dựng Prompt kỹ năng đang hoạt động dành riêng cho Planner
    active_skills_prompt = ""
    if active_skills:
        active_skills_prompt = "\n=== ⚡ CÁC KỸ NĂNG ĐANG HOẠT ĐỘNG (INSTRUCTIONS) ===\n"
        for s_name, s_instructions in active_skills.items():
            active_skills_prompt += f"\n--- CHỈ DẪN KỸ NĂNG `{s_name}` ---\n{s_instructions}\n"
    # Xác định trạng thái chuyển tiếp từ khảo sát sang phát triển
    is_survey_transition = (
        len(plan) == 1 and 
        (plan[0].id if isinstance(plan[0], Task) else plan[0].get("id")) == "T_SURVEY" and
        (plan[0].status if isinstance(plan[0], Task) else plan[0].get("status")) == "completed"
    )
    
    # Kiểm tra nếu không có lỗi phát sinh và không phải là pha chuyển đổi khảo sát
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
    system_prompt += active_skills_prompt 
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
        updated_tasks = getattr(decision, "updated_tasks", [])
        updated_task_type = getattr(decision, "task_type", task_type)
        
        # Nếu LLM phân tích xong và trả về danh sách nhiệm vụ rỗng
        if not updated_tasks or not should_modify:
            proposal_message = AIMessage(
                content="📋 [Hệ thống tự động duyệt qua: Kế hoạch hiện tại đã tối ưu, không cần cập nhật thêm]",
                name="replanner_proposal",
                additional_kwargs={"proposal_payload": {"action": "bypass_no_error", "tasks": []}}
            )
            return {"messages": [proposal_message],"active_skills": active_skills }

        # Chuẩn hóa danh sách task
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
        # Cơ chế dự phòng khẩn cấp (Fail-Safe) khi LLM lỗi
        proposal_data = {
            "action": "bypass_no_error", # Bỏ qua ngắt để không gây treo đồ thị
            "explanation": f"Kích hoạt cơ chế tự phục hồi do lỗi hệ thống: {str(e)}",
            "task_type": task_type,
            "tasks": []
        }
        
    # Tạo đóng gói phản hồi đồng nhất
    # 🛡️ GIẢI PHÁP THEN CHỐT: Ghi nội dung text sạch vào content để hiển thị trên UI,
    # cất cấu trúc JSON kỹ thuật vào additional_kwargs để phục vụ xử lý ngầm.
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
    """
    Nút ngắt duyệt kế hoạch thông minh (Human-in-the-Loop Gate).
    Chỉ kích hoạt ngắt khi có kế hoạch thực sự, ẩn JSON điều khiển khỏi UI chat,
    và tự động phân tích phản hồi tùy chỉnh của người dùng để cập nhật lộ trình.
    """
    messages = state["messages"]
    plan = state["plan"]
    
    # =====================================================================
    # BƯỚC 1: TRÍCH XUẤT PAYLOAD PHÂN TÍCH NỘI BỘ (IDEMPOTENT EXTRACTION)
    # =====================================================================
    proposal_payload = {}
    for msg in reversed(messages):
        # Tìm tin nhắn chứa gói dữ liệu kế hoạch được đóng gói ẩn bởi replanner_node
        if getattr(msg, "name", None) == "replanner_proposal":
            proposal_payload = msg.additional_kwargs.get("proposal_payload", {})
            break
            
    # Nếu không tìm thấy bất kỳ đề xuất nào từ nút replanner trước đó,
    # bảo toàn kế hoạch hiện hành và đi tiếp
    if not proposal_payload:
        logger.warning("[Replanner Gate] Không tìm thấy dữ liệu đề xuất từ replanner_node.")
        return {
            "error_logs": "",
            "attempts": 0,
            "modified_files": []
        }
        
    action = proposal_payload.get("action", "bypass_no_error")
    proposed_tasks = proposal_payload.get("tasks", [])
    
    # =====================================================================
    # BƯỚC 2: TỰ ĐỘNG DUYỆT (AUTO-APPROVE BYPASS GUARD)
    # Ngăn chặn tuyệt đối việc bắt người dùng duyệt bản kế hoạch rỗng
    # =====================================================================
    if action in ["bypass_limit", "bypass_no_error"] or not proposed_tasks:
        # Chuẩn bị lại danh sách nhiệm vụ từ kế hoạch cũ
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
        
    # =====================================================================
    # BƯỚC 3: THIẾT LẬP GIAO DIỆN INTERRUPT PAYLOAD
    # =====================================================================
    # Payload này sẽ được serialization thành JSON và gửi trực tiếp lên giao diện Client/Studio
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
    
    # KÍCH HOẠT NGẮT ĐỒ THỊ (LangGraph pauses here and waits for Command(resume=value))
    user_input = interrupt(interrupt_payload)
    
    # =====================================================================
    # BƯỚC 4: BỘ PHÂN TÍCH PHẢN HỒI ĐA TẦNG (DEFENSIVE RESPONSE PARSER)
    # Chạy khi đồ thị được RESUME. Xử lý tất cả các kịch bản đầu vào từ UI.
    # =====================================================================
    
    # Mặc định hóa phản hồi
    user_input_clean = ""
    if isinstance(user_input, str):
        user_input_clean = user_input.strip().lower()
    
    # --- Kịch bản 4.1: Đồng ý mặc định (Approval Path) ---
    # Người dùng gửi 'yes', 'approve', 'ok', hoặc chỉ ấn Enter (chuỗi rỗng / None)
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
        
    # --- Kịch bản 4.2: Từ chối / Bỏ qua (Skip/Decline Path) ---
    # Người dùng không muốn thay đổi lộ trình, giữ nguyên kế hoạch hiện tại
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
        
    # --- Kịch bản 4.3: Người dùng nhập Kế hoạch tùy chỉnh (Custom Task List Path) ---
    # Giao diện frontend hoặc Studio có thể cho phép người dùng tùy chỉnh danh sách Tasks 
    # và gửi về dưới dạng chuỗi JSON hoặc một cấu trúc dữ liệu mảng trực tiếp.
    custom_tasks_raw = []
    
    # Nếu client gửi về một list/dict trực tiếp
    if isinstance(user_input, list):
        custom_tasks_raw = user_input
    elif isinstance(user_input, dict) and "tasks" in user_input:
        custom_tasks_raw = user_input["tasks"]
    elif isinstance(user_input, str):
        # Thử nghiệm phân tích cú pháp chuỗi JSON nếu người dùng tự nhập tay cấu trúc mảng
        try:
            parsed = json.loads(user_input)
            if isinstance(parsed, list):
                custom_tasks_raw = parsed
            elif isinstance(parsed, dict) and "tasks" in parsed:
                custom_tasks_raw = parsed["tasks"]
        except json.JSONDecodeError:
            pass

    # Nếu phân tích ra được danh sách task tùy chỉnh hợp lệ
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
            
    # --- Kịch bản 4.4: Phản hồi tự do (Free-text Feedback Path) ---
    # Nếu người dùng không nhập đúng từ khóa điều hướng và cũng không phải JSON,
    # mà nhập một phản hồi văn bản tự do (ví dụ: "Hãy làm thêm bước X trước bước Y"),
    # chúng ta sẽ chuyển tiếp ý kiến phản hồi này làm tin nhắn hệ thống đưa ngược về Executor/Replanner
    logger.info(f"[Replanner Gate] Ghi nhận phản hồi văn bản tự do: {user_input}")
    feedback_message = HumanMessage(
        content=(
            "⚠️ Ý kiến điều chỉnh lộ trình từ người dùng:\n"
            f"'{user_input}'\n"
            "Hãy phân tích và cập nhật lại kế hoạch hành động tương ứng dựa trên ý kiến này."
        )
    )
    
    # Hoàn trả lại kế hoạch cũ để Replanner có thể xử lý điều chỉnh lại ở lượt tiếp theo
    fallback_tasks = [t if isinstance(t, Task) else Task(**t) for t in plan]
    return {
        "plan": fallback_tasks,
        "error_logs": f"Người dùng yêu cầu thay đổi lộ trình: {user_input}",
        "attempts": 0,
        "messages": [feedback_message]
    }


def tool_node(state: AgentState) -> Dict[str, Any]:
    """
    Nút thực thi công cụ. Đã nâng cấp cơ chế nén Token trực tiếp cho tệp tin khi ghi nhận vào lịch sử tin nhắn.
    """
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
    # 🌟 VÁ LỖI: Khởi tạo ProposePlanTool cho Node thực thi
    
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
            except Exception as e:
                result = f"Lỗi thực thi công cụ '{tool_name}': {str(e)}"
                
        # 🌟 PHÒNG THỦ TOKEN TRỰC TIẾP: Nén nội dung file ngay tại đầu ra của Tool Message
        if tool_name == "read_files" and "Lỗi" not in str(result):
            found_files = []
            raw_paths = tool_args.get("file_paths")
            if isinstance(raw_paths, list):
                found_files = raw_paths
            elif isinstance(raw_paths, str):
                found_files = [raw_paths]
                
            file_info = f" của tệp {', '.join([f'`{f}`' for f in found_files])}" if found_files else ""
            compacted_result = f"[Đã nạp thành công dữ liệu vật lý{file_info} vào File Registry. Hãy sử dụng cấu trúc mã nguồn cập nhật mới nhất trong System Prompt để làm việc]"
            tool_messages.append(ToolMessage(content=compacted_result, name=tool_name, tool_call_id=tool_id))
        else:
            tool_messages.append(ToolMessage(content=str(result), name=tool_name, tool_call_id=tool_id))
        
    BINARY_EXTENSIONS = {".xlsx", ".xls", ".png", ".jpg", ".jpeg", ".zip", ".pdf", ".exe"}
    
    for file_path in impacted_files:
        try:
            safe_path = sanitize_and_resolve_path(ws, file_path, create_parent=False)
            if safe_path.exists() and safe_path.is_file():
                if safe_path.suffix.lower() in BINARY_EXTENSIONS:
                    continue
                current_content = safe_path.read_text(encoding="utf-8")
                file_registry[file_path] = current_content
        except Exception:
            pass
            
    return {
        "messages": tool_messages,
        "modified_files": modified_files,
        "file_registry": file_registry
    }


def human_interaction_gate_node(state: AgentState) -> Dict[str, Any]:
    """
    Node rào chắn tương tác người dùng chỉ dành cho mục đích làm rõ thông tin (Clarification Questions).
    Đã loại bỏ hoàn toàn phần xử lý kế hoạch thủ công do vai trò này được chuyển giao hoàn toàn cho Replanner.
    """
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

    # Thực hiện ngắt đồ thị để đợi câu trả lời từ người dùng cho các câu hỏi
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
    """
    Nút kiểm thử tĩnh. Đã nâng cấp cơ chế ép kiểu phòng thủ từ Checkpoint.
    """
    modified_files = state.get("modified_files", [])
    attempts = state.get("attempts", 0)
    plan = state["plan"]
    ws = state["workspace_path"]
    last_executed_ids = state.get("last_executed_task_ids", [])
    
    # 🌟 VÁ LỖI: Ép kiểu phòng thủ để tránh lỗi AttributeError từ checkpointer thô
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
                "error_logs": "",
                "attempts": 0,
                "modified_files": [],
                "messages": [AIMessage(content=f"{warning_msg}❌ Đã vượt quá giới hạn số lần sửa lỗi tự động. Bỏ qua để tiến tục.")]
            }
                
    success_content = f"{warning_msg}✅ [Vòng kiểm thử thành công] Toàn bộ mã nguồn đã vượt qua kiểm tra tĩnh."
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