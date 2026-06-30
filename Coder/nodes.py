# nodes.py
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional,  Tuple, List
from concurrent.futures import ThreadPoolExecutor

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from config import find_project_root_heuristic, model, sanitize_and_resolve_path, fast_model
from state import AgentState, PlanUpdate, TaskPlan, TaskTriage, Task
from tools import (
    GitManager, ReadFileLinesTool, UniversalSymbolSearchTool, WebInteractAndTestTool, WorkspaceTools, 
    ReadFilesTool, WriteFileTool, ApplyPatchTool, 
    ListDirectoryTool, RunTerminalTool, get_markdown_language
)
def get_text_content_safely(content: Any) -> str:
    """
    Trích xuất văn bản (text content) an toàn từ tin nhắn của người dùng.
    Hỗ trợ cả trường hợp tin nhắn dạng chuỗi (str) hoặc danh sách đa phương tiện (list).
    """
    if isinstance(content, str):
        return content.strip()
        
    elif isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                # Trường hợp cấu trúc chuẩn: {"type": "text", "text": "..."}
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                # Trường hợp cấu trúc rút gọn: {"text": "..."}
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
        if msg.type == "tool" and msg.name in ["read_files", "read_file_lines", "search_symbols_universal"]:
            content_str = str(msg.content)
            found_files = []
            
            matches_read = re.findall(r"=== TỆP TIN:\s*[`']?([^`'\n]+)[`']?\s*===", content_str)
            if matches_read:
                found_files.extend(matches_read)
                
            matches_lines = re.findall(r"=== NỘI DUNG TỆP KHOANH VÙNG:\s*([^ \n]+)", content_str)
            if matches_lines:
                found_files.extend(matches_lines)
                
            matches_symbols = re.findall(r"📁\s*`([^`\n]+)`", content_str)
            if matches_symbols:
                found_files.extend(matches_symbols)
                
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


# =====================================================================
# CẬP NHẬT DETECT_AND_TRIAGE_NODE SỬ DỤNG FUNCTION ĐỂ DUY TRÌ HIỆU NĂNG
# =====================================================================
def detect_and_triage_node(state: AgentState) -> Dict[str, Any]:
    messages = state["messages"]
    user_msg = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) or getattr(msg, "type", None) == "human":
            user_msg = msg
            break
            
    user_query_text = get_text_content_safely(user_msg.content) if user_msg else ""
    
    # 1. PHÂN TÍCH TĨNH: Tìm kiếm đường dẫn vật lý trong yêu cầu (KHÔNG dùng AI)
    detected_path_str = extract_path_from_text(user_query_text)
    
    workspace_path = None
    is_fallback_workspace = False
    
    if detected_path_str:
        # Nếu tìm thấy đường dẫn cụ thể -> Xác định gốc dự án một cách tự động
        start_path = Path(detected_path_str)
        resolved_root = find_project_root_heuristic(start_path)
        workspace_path = str(resolved_root)
    else:
        # Nếu KHÔNG tìm thấy, hoãn việc thiết lập đường dẫn và đánh dấu chế độ Fallback
        is_fallback_workspace = True
        # Gán tạm thời là thư mục hiện tại để phục vụ chạy phân loại trước
        workspace_path = state.get("workspace_path", ".")
        
    # 2. CHẠY PHÂN LOẠI TRƯỚC (Triage) để hiểu bản chất của tác vụ
    triage_res = triage_node(state)
    
    # Cơ chế phòng vệ (Defensive programming): Đảm bảo triage_res luôn hợp lệ
    if triage_res is None:
        triage_res = {
            "plan": [],
            "task_type": "development",
            "last_executed_task_ids": [],
            "messages": [AIMessage(content="⚠️ Không thể phân loại tác vụ. Đang chạy ở chế độ mặc định.")],
            "replanning_count": 0,
            "modified_files": [],     
            "error_logs": "",         
            "step_findings": [],      
            "is_simple": True
        }
        
    is_simple = triage_res.get("is_simple", False)
    
    # 3. ĐIỀU CHỈNH WORKSPACE DỰA TRÊN KẾT QUẢ PHÂN LOẠI
    if is_fallback_workspace:
        base_path = Path(workspace_path).resolve()
        
        # Nếu là tác vụ đơn giản/hướng ngoại (như web automation hoặc terminal độc lập)
        if is_simple:
            # Chỉ cần khởi tạo và gán thư mục Sandbox "temp/" cách ly, không cần quét toàn bộ mã nguồn làm gì
            temp_dir = base_path / "temp"
            if temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception:
                    pass
            temp_dir.mkdir(parents=True, exist_ok=True)
            workspace_path = str(temp_dir)
            
            # Thêm temp/ vào .gitignore của thư mục chạy để tránh làm bẩn git dự án lớn
            try:
                root_gitignore = base_path / ".gitignore"
                if root_gitignore.exists():
                    gitignore_content = root_gitignore.read_text(encoding="utf-8")
                    if "temp/" not in gitignore_content:
                        with open(root_gitignore, "a", encoding="utf-8") as f:
                            f.write("\n# AI Generated Sandbox\ntemp/\n")
                else:
                    root_gitignore.write_text("# AI Generated Sandbox\ntemp/\n", encoding="utf-8")
            except Exception:
                pass
        else:
            # Nếu là tác vụ lập trình phức tạp nhưng người dùng quên chỉ định đường dẫn
            # Tự động gán gốc dự án mặc định tại Project Root của thư mục hiện tại
            resolved_root = find_project_root_heuristic(base_path)
            workspace_path = str(resolved_root)
            
    # 4. CHUẨN BỊ TIN NHẮN HIỂN THỊ TRỰC QUAN
    detect_messages = []
    if is_fallback_workspace:
        if is_simple:
            detect_messages.append(
                AIMessage(content=f"ℹ️ Phát hiện tác vụ Sandbox độc lập. Kích hoạt môi trường cách ly tại: `{workspace_path}`.")
            )
        else:
            detect_messages.append(
                AIMessage(content=f"🔍 Không chỉ định đường dẫn. Tự động thiết lập gốc dự án làm việc tại: `{workspace_path}`.")
            )
    else:
        detect_messages.append(
            AIMessage(content=f"🔍 Hệ thống đã phát hiện đường dẫn chỉ định và thiết lập workspace tại: `{workspace_path}`.")
        )
        
    merged_messages = detect_messages + triage_res.get("messages", [])
    
    return {
        "workspace_path": workspace_path,
        "plan": triage_res.get("plan", []),
        "task_type": triage_res.get("task_type", "development"),
        "last_executed_task_ids": triage_res.get("last_executed_task_ids", []),
        "replanning_count": triage_res.get("replanning_count", 0),
        "modified_files": triage_res.get("modified_files", []),
        "error_logs": triage_res.get("error_logs", ""),
        "step_findings": triage_res.get("step_findings", []),
        "is_simple": is_simple,
        "messages": merged_messages
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
            
    return {
        "workspace_context": workspace_context,
        "git_branch": git_branch,
        "messages": [AIMessage(content=f"{git_msg}\n{context_msg}")]
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
    
    structured_llm = fast_model.with_structured_output(TaskTriage, method="function_calling")
    
    system_prompt = (
        "Bạn là một điều phối viên Agent thông minh cấp cao (Triage Supervisor).\n"
        "Nhiệm vụ của bạn là phân tích yêu cầu của người dùng để xác định xem yêu cầu đó nên được xử lý qua luồng Fast-Track (Đơn giản) hay luồng Lập kế hoạch (Phức tạp).\n\n"
        "Hãy dựa trên các tiêu chí phân loại nghiêm ngặt sau:\n\n"
        "1. KIỂM TRA TÁC VỤ ĐƠN GIẢN (is_simple = True):\n"
        "   - Tương tác Web trực tiếp: Các yêu cầu truy cập website, nhấp nút, điền form, chụp ảnh màn hình hoặc chạy thử JS trên một trang cụ thể (ví dụ: 'vào trang web abc.com và nhấn...', 'chụp màn hình web...'). Các tác vụ này có thể thực thi trực tiếp bằng công cụ 'web_interact_and_test' mà không cần lập kế hoạch viết mã nguồn.\n"
        "   - Khảo sát hệ thống đơn giản: Đọc nội dung 1-2 tệp tin cụ thể, liệt kê thư mục, hoặc tìm kiếm symbol.\n"
        "   - Thực thi Terminal trực tiếp: Chạy một lệnh terminal đơn lẻ (ví dụ: kiểm tra phiên bản, chạy thử một tệp tin script có sẵn).\n"
        "   - Chỉnh sửa nhỏ: Sửa đổi nhanh chỉ một vài dòng mã hoặc ghi đè một tệp tin ngắn dưới 100 dòng.\n\n"
        "2. KIỂM TRA TÁC VỤ PHỨC TẠP (is_simple = False):\n"
        "   - Phát triển tính năng mới (Feature Development): Yêu cầu viết mới hoặc can thiệp chỉnh sửa logic phức tạp trên nhiều tệp tin nguồn khác nhau.\n"
        "   - Sửa lỗi hệ thống diện rộng (Complex Bug Fixing): Đòi hỏi phải phân tích kiến trúc, tìm kiếm ký hiệu xuyên suốt mã nguồn trước khi đưa ra phương án sửa đổi.\n"
        "   - Phân tích & Viết tài liệu tổng thể dự án: Khảo sát sâu toàn bộ workspace lớn để viết báo cáo kỹ thuật (chọn task_type = 'analysis').\n\n"
        "QUY TẮC CHỌN TASK_TYPE:\n"
        "   - Chọn 'analysis' nếu yêu cầu thuần túy chỉ là đọc hiểu, giải thích cấu trúc mã nguồn hoặc khảo sát dự án (không viết/sửa code trên ổ đĩa).\n"
        "   - Chọn 'development' cho các trường hợp còn lại (có viết code, sửa code, chạy terminal, hoặc tương tác trình duyệt web)."
    )
    
    try:
        triage_output = structured_llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query_text}
        ])
        is_simple = getattr(triage_output, "is_simple", False)
        task_type = getattr(triage_output, "task_type", "development")
    except Exception:
        is_simple = False
        task_type = "development"
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
        
    return {
        "plan": plan,  # Gán kế hoạch đơn bước thay vì danh sách rỗng
        "task_type": task_type,
        "is_simple": is_simple,
        "messages": [
            AIMessage(
                content=f"📊 **[Phân loại tác vụ]**: Hệ thống xác định yêu cầu thuộc diện "
                        f"{'ĐƠN GIẢN (Fast-Track)' if is_simple else 'PHỨC TẠP (Multi-Step Planning)'} | "
                        f"Pha hoạt động: `{task_type.upper()}`."
            )
        ]
    }

def planner_node(state: AgentState) -> Dict[str, Any]:
    ws = state["workspace_path"]
    conversation_history = state["messages"]
    existing_plan = state.get("plan", [])
    workspace_context = state.get("workspace_context", "")
    
    if existing_plan:
        return {
            "modified_files": [],
            "attempts": 0,
            "step_findings": ["__RESET__"],
            "last_executed_task_ids": []
        }
    
    structured_llm = model.with_structured_output(TaskPlan, method="function_calling")
    
    system_prompt = (
        "Bạn là một Kiến trúc sư phần mềm cấp cao. Hãy lập hoặc cập nhật kế hoạch thực hiện "
        f"cho dự án nằm trong thư mục '{ws}'.\n"
    )
    if workspace_context:
        system_prompt += f"\n--- NGỮ CẢNH HỆ THỐNG HIỆN TẠI (THONGTIN.md) ---\n{workspace_context}\n"
        
    system_prompt += (
        "Hãy phân tích kỹ lưỡng toàn bộ lịch sử trò chuyện để sinh ra kế hoạch dạng Đồ thị phụ thuộc (DAG).\n"
        "QUY TẮC QUAN TRỌNG VỀ PHỤ THUỘC (DEPENDENCIES):\n"
        "- Thiết lập `dependencies` của từng nhiệm vụ chính xác để cho phép chạy song song (nếu không liên quan) hoặc nối tiếp (nếu cần sự liên tục).\n"
        "⚠️ LƯU Ý ĐẶC BIỆT QUAN TRỌNG ĐỂ TRÁNH TRÙNG LẶP (BẮT BUỘC):\n"
        "- Tuyệt đối KHÔNG đưa bước 'Tổng hợp', 'Báo cáo', hay 'Viết tệp THONGTIN.md' làm nhiệm vụ trong kế hoạch! "
        "Tác vụ tổng hợp đã được quản lý tự động bởi node 'synthesis' chuyên biệt ở phía sau.\n"
        "- Tuyệt đối KHÔNG đưa bước 'Chạy test suite', 'Kiểm thử logic' (ví dụ: pytest, npm test, dart test) vào kế hoạch! "
        "Tác vụ kiểm thử tĩnh và kiểm tra cú pháp đã được quản lý tự động bởi node 'tester' chuyên biệt ở phía sau.\n"
        "- Kế hoạch của bạn chỉ tập quan tâm hoàn toàn vào các bước thực thi khảo sát vật lý (analysis) hoặc sửa đổi mã nguồn (development)."
    )
    
    try:
        plan_output = structured_llm.invoke([
            {"role": "system", "content": system_prompt},
            *conversation_history
        ])
        plan_tasks = getattr(plan_output, "tasks", [])
        task_type = getattr(plan_output, "task_type", "development")
    except Exception as e:
        print(f"[Cảnh báo] Lỗi parse cấu trúc Planner: {str(e)}. Sử dụng kế hoạch dự phòng đơn bước.")
        user_query = "Thực hiện yêu cầu hiện tại"
        for msg in reversed(conversation_history):
            if isinstance(msg, HumanMessage) or getattr(msg, "type", None) == "human":
                user_query = msg.content
                break
        plan_tasks = [Task(
            id="T1", 
            description=f"Thực hiện trực tiếp nhiệm vụ từ yêu cầu: {user_query}", 
            dependencies=[], 
            status="pending"
        )]
        task_type = "development"
    
    plan_str = "\n".join([
        f"- [{t.id}] {t.description} (Phụ thuộc: {t.dependencies if t.dependencies else 'Không'})" 
        for t in plan_tasks
    ])
    
    return {
        "plan": plan_tasks,
        "task_type": task_type,
        "modified_files": [],
        "attempts": 0,
        "step_findings": ["__RESET__"],
        "last_executed_task_ids": [],
        "messages": [AIMessage(content=f"Đã lập kế hoạch hành động dạng đồ thị phụ thuộc (Loại: {task_type.upper()}):\n{plan_str}")]
    }


def executor_node(state: AgentState) -> Dict[str, Any]:
    ws = state["workspace_path"]
    plan = state["plan"]
    error_logs = state.get("error_logs", "")
    workspace_context = state.get("workspace_context", "")
    file_registry = state.get("file_registry", {})
    messages = list(state["messages"])
    task_type = state.get("task_type", "development")
    
    eligible_tasks = get_eligible_tasks(plan)
    if not eligible_tasks:
        pending_tasks = [t for t in plan if (t.get("status") if isinstance(t, dict) else getattr(t, "status", None)) == "pending"]
        if pending_tasks:
            eligible_tasks = [pending_tasks[0]]
        else:
            return {"messages": [AIMessage(content=f"Đã hoàn thành khảo sát toàn bộ các bước {task_type}.")]}
         
    tasks_str = "\n".join([
        f"- [{getattr(t, 'id', None) or t.get('id')}] {getattr(t, 'description', None) or t.get('description')}"
        for t in eligible_tasks
    ])

    if task_type == "analysis":
        read_files = ReadFilesTool(workspace_path=ws)
        list_directory = ListDirectoryTool(workspace_path=ws)
        search_symbols = UniversalSymbolSearchTool(workspace_path=ws)
        read_file_lines = ReadFileLinesTool(workspace_path=ws)
        tools = [read_files, list_directory, search_symbols, read_file_lines]

        previous_findings_str = ""
        existing_findings = state.get("step_findings", [])
        if existing_findings:
            previous_findings_str = "\n\n--- CÁC KẾT QUẢ KHẢO SÁT BẠN ĐÃ THU THẬP ĐƯỢC Ở CÁC BƯỚC TRƯỚC ---\n" + "\n\n".join(existing_findings)
        
        system_prompt = (
            "Bạn là một kiến trúc sư chuyên khảo sát, đọc hiểu và phân tích cấu trúc mã nguồn (Read-Only Mode).\n"
            f"Nhiệm vụ: Bạn đang thực hiện đồng thời các nhiệm vụ sau:\n{tasks_str}\n"
            f"Thư mục làm việc: {ws}\n"
        )
        if workspace_context:
            system_prompt += f"\n--- TỔNG QUAN VỀ DỰ ÁN (THONGTIN.md) ---\n{workspace_context}\n"
            
        system_prompt += (
            "\nHãy sử dụng các công cụ khảo sát cấu trúc hệ thống.\n"
            "\n⚠️ RÀNG BUỘC PHẠM VI NGHIÊM NGẶT (BẮT BUỘC):\n"
            "1. Bạn chỉ có quyền ĐỌC dữ liệu, tuyệt đối không chỉnh sửa mã nguồn hoặc tự ý tạo tệp tin trong bước này.\n"
            "2. KHÔNG ĐƯỢC PHÉP tự ý định dạng tài liệu báo cáo hoàn chỉnh, tổng hợp tri thức hay viết tệp THONGTIN.md. "
            "Nhiệm vụ của bạn chỉ là thu thập thông tin thô, liệt kê các phát hiện kỹ thuật (raw findings) thực tế. "
            "Việc tổng hợp chúng thành một tài liệu THONGTIN.md hoàn chỉnh là nhiệm vụ ĐỘC QUYỀN của node 'synthesis' tiếp theo."
        )
        if previous_findings_str:
            system_prompt += previous_findings_str

    else:  # development
        registry_context_str = ""
        if file_registry:
            registry_context_str = "\n=== 📦 NỘI DUNG MÃ NGUỒN CẬP NHẬT MỚI NHẤT (SINGLE SOURCE OF TRUTH) ===\n"
            for file_path, content in file_registry.items():
                lang = get_markdown_language(file_path)
                lines = content.splitlines()
                formatted_lines = [f"{idx+1:04d} | {line}" for idx, line in enumerate(lines)]
                
                registry_context_str += (
                    f"\n--- TỆP TIN: `{file_path}` ---\n"
                    f"```{lang}\n" + "\n".join(formatted_lines) + "\n```\n"
                )
                
        web_interact_tool = WebInteractAndTestTool(workspace_path=ws)
        read_files = ReadFilesTool(workspace_path=ws)
        write_file = WriteFileTool(workspace_path=ws)
        apply_patch = ApplyPatchTool(workspace_path=ws)
        search_symbols = UniversalSymbolSearchTool(workspace_path=ws)
        list_directory = ListDirectoryTool(workspace_path=ws)
        run_terminal_command = RunTerminalTool(workspace_path=ws)
        read_file_lines = ReadFileLinesTool(workspace_path=ws)
        
        tools = [read_files, write_file, apply_patch, list_directory, run_terminal_command, search_symbols, read_file_lines, web_interact_tool]
        
        system_prompt = (
            "Bạn là một kỹ sư phần mềm thực thi chuyên nghiệp chuyên sửa lỗi và viết mới mã nguồn (Write-Access Mode).\n"
            f"Nhiệm vụ: Bạn đang thực hiện đồng thời các nhiệm vụ sau:\n{tasks_str}\n"
            f"Thư mục làm việc: {ws}\n"
        )
        if workspace_context:
            system_prompt += f"\n--- TỔNG QUAN VỀ DỰ ÁN (THONGTIN.md) ---\n{workspace_context}\n"
        
        if registry_context_str:
            system_prompt += registry_context_str
            
        system_prompt += (
            "\n⚠️ QUY TẮC PHẠM VI VÀ PHÒNG TRÁNH LỆCH DÒNG (BẮT BUỘC):\n"
            "1. Bạn đã được cung cấp nguồn mã nguồn mới nhất (đã đánh số dòng chi tiết) trong mục 'SINGLE SOURCE OF TRUTH' ở trên.\n"
            "2. ĐÂY LÀ NỘI DUNG MỚI NHẤT VÀ CHÍNH XÁC NHẤT. Hãy luôn sử dụng mốc dòng và nội dung từ mục này để thiết lập khối SEARCH-AND-REPLACE cho công cụ `apply_search_replace_patch`.\n"
            "3. Nếu bạn vừa sửa đổi một file ở bước trước, nội dung file đó trong mục 'SINGLE SOURCE OF TRUTH' đã được cập nhật tự động. Bạn không cần phải gọi lại công cụ đọc trừ khi muốn kiểm tra sâu hơn.\n"
            "\n⚠️ QUY TẮC SỬ DỤNG CÔNG CỤ:\n"
            "1. Đối với file trên 300 dòng: BẮT BUỘC dùng `apply_search_replace_patch` để áp dụng bản vá, cấm ghi đè bừa bãi.\n"
            "2. Công cụ `write_file` chỉ dùng khi tạo mới hoặc sửa các tệp ngắn dưới 300 dòng.\n"
            "3. NGHIÊM CẤM thực hiện chạy các bộ kiểm thử tự động (như pytest, cargo test, dart test, npm test, vitest) bằng công cụ `run_terminal_command`."
        )

    model_with_tools = model.bind_tools(tools)
    optimized_history = compact_reading_tool_messages(messages)
    
    input_messages = [SystemMessage(content=system_prompt)]
    if task_type == "development" and error_logs:
        input_messages.append(HumanMessage(content=f"LƯU Ý SỬA LỖI TỪ LƯỢT CHẠY TRƯỚC:\n{error_logs}\nHãy sửa triệt để."))
        
    response = model_with_tools.invoke(input_messages + optimized_history)
    response = sanitize_llm_response_content(response)
    
    if not response.tool_calls:
        findings = []
        if task_type == "analysis" and response.content:
            tasks_ids = ", ".join([str(getattr(t, "id", None) or t.get('id')) for t in eligible_tasks])
            findings = [f"### Kết quả khảo sát các nhiệm vụ ({tasks_ids}):\n{response.content}"]
            
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

        ret_dict = {
            "messages": [response],
            "plan": updated_plan,
            "last_executed_task_ids": list(eligible_ids)
        }
        if task_type == "analysis" and findings:
            ret_dict["step_findings"] = findings
            
        return ret_dict
    else:
        return {"messages": [response]}


def replanner_node(state: AgentState) -> Dict[str, Any]:
    replanning_count = state.get("replanning_count", 0)
    
    if replanning_count >= 5:
        return {
            "messages": [AIMessage(content="🚨 **[Hệ thống tự động dừng]** Phát hiện vòng lặp lập kế hoạch quá nhiều lần. Chuyển tiếp tới bước tiếp theo.")]
        }
    ws = state["workspace_path"]
    plan = state["plan"]
    messages = state["messages"]
    task_type = state.get("task_type", "development")
    workspace_context = state.get("workspace_context", "")
    error_logs = state.get("error_logs", "")
    
    if not error_logs:
        return {
            "plan": plan,
            "task_type": task_type,
            "messages": [AIMessage(content="🔄 [Bypass Replanner] Không phát hiện lỗi phát sinh. Tiếp tục thực hiện kế hoạch.")]
        }
    
    plan_str = "\n".join([
        f"- [{t.id if isinstance(t, Task) else t.get('id')}] {t.description if isinstance(t, Task) else t.get('description')} "
        f"(Trạng thái: {t.status if isinstance(t, Task) else t.get('status')}, "
        f"Phụ thuộc: {t.dependencies if isinstance(t, Task) else t.get('dependencies')})"
        for t in plan
    ])
    
    system_prompt = (
        "Bạn là một Kiến trúc sư kiêm Điều phối viên dự án phần mềm cấp cao.\n"
        f"Nhiệm vụ: Đánh giá tiến trình thực thi kế hoạch tại thư mục làm việc '{ws}'.\n"
        "Hãy phân tích kỹ các tin nhắn hội thoại và kết quả thực thi các công cụ gần nhất để quyết định xem kế hoạch hiện tại có cần thích ứng hay không.\n\n"
        "⚠️ QUY TẮC CẬP NHẬT KẾ HOẠCH CHO PRODUCTION (BẮT BUỘC):\n"
        "1. Nếu phát hiện thấy lỗi phát sinh (lỗi cú pháp, lỗi test, lỗi logic), file bị thiếu, hoặc cần thêm các bước khảo sát/phát triển bổ sung, "
        "hãy đặt `should_modify_plan` là True và cập nhật danh sách nhiệm vụ trong `updated_tasks` để giải quyết vấn đề.\n"
        "2. Nếu tiến trình diễn ra hoàn hảo không có lỗi và không cần bổ sung gì, hãy đặt `should_modify_plan` là False.\n"
        "3. ĐỐI VỚI CÁC NHIỆM VỤ ĐÃ HOÀN THÀNH (status: 'completed'): Bắt buộc giữ nguyên ID, mô tả và trạng thái là 'completed'. Tuyệt đối không xóa hoặc reset trạng thái của chúng trừ khi cần thực hiện lại từ đầu.\n"
        "4. Đảm bảo các nhiệm vụ mới (nếu có) được đặt ID mới (ví dụ: T1.1, T3_fix) và thiết lập quan hệ phụ thuộc `dependencies` chính xác.\n"
        "5. RẤT QUAN TRỌNG: Nếu kế hoạch chuyển từ giai đoạn khảo sát (analysis) sang phát triển/sửa đổi code (development), hãy cập nhật giá trị `task_type` tương ứng sang 'development'.\n\n"
        "⚠️ QUY TẮC CẤM ĐOÁN ĐẶC BIỆT QUAN TRỌNG ĐỂ TRÁNH TRÙNG LẶP (BẮT BUỘC):\n"
        "- Tuyệt đối KHÔNG đưa bước 'Tổng hợp', 'Báo cáo', hay 'Viết tệp THONGTIN.md' làm nhiệm vụ trong kế hoạch! "
        "Tác vụ tổng hợp đã được quản lý tự động bởi node 'synthesis' chuyên biệt ở phía sau.\n"
        "- Tuyệt đối KHÔNG đưa bước 'Chạy test suite', 'Kiểm thử logic' (ví dụ: pytest, npm test, chạy file test_*.py) vào kế hoạch! "
        "Tác vụ kiểm thử tĩnh và kiểm tra động đã được quản lý tự động bởi node 'tester' chuyên biệt ở phía sau sau khi toàn bộ kế hoạch phát triển hoàn tất.\n"
        "- Kế hoạch cập nhật của bạn chỉ tập trung hoàn toàn vào các bước thực thi khảo sát vật lý (analysis) hoặc sửa đổi mã nguồn (development)."
    )
    
    if workspace_context:
        system_prompt += f"\n\n--- NGỮ CẢNH HỆ THỐNG (THONGTIN.md) ---\n{workspace_context}"
        
    user_prompt = f"Kế hoạch hiện tại:\n{plan_str}\n\n"
    if error_logs:
        user_prompt += f"🚨 PHÁT HIỆN LỖI KIỂM TRA/BIÊN DỊCH CẦN SỬA ĐỔI KẾ HOẠCH:\n{error_logs}\n\n"
        
    user_prompt += "Hãy đưa ra phân tích và cập nhật kế hoạch phù hợp thông qua cuộc gọi hàm."
    
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
            
        if should_modify and refined_tasks:
            new_plan_str = "\n".join([
                f"- [{t.id}] {t.description} (Trạng thái: {t.status}, Phụ thuộc: {t.dependencies})" 
                for t in refined_tasks
            ])
            return {
                "plan": refined_tasks,
                "task_type": updated_task_type,
                "replanning_count": replanning_count + 1,
                "messages": [AIMessage(content=f"🔄 **[Điều chỉnh kế hoạch]** {explanation}\n\nKế hoạch hành động mới (Chuyển pha: {updated_task_type.upper()}):\n{new_plan_str}")]
            }
        else:
            return {
                "plan": refined_tasks,
                "task_type": updated_task_type,
                "messages": [AIMessage(content=f"✅ **[Giữ nguyên lộ trình]** {explanation or 'Tiến trình thực thi đang bám sát kế hoạch ban đầu.'}")]
            }
            
    except Exception as e:
        print(f"[Cảnh báo] Lỗi Re-planner: {str(e)}. Tiếp tục sử dụng kế hoạch hiện có.")
        return {
            "messages": [AIMessage(content="⚠️ Không thể tự động điều chỉnh kế hoạch do sự cố phân tích cấu trúc từ LLM. Tiếp tục bám sát kế hoạch cũ.")]
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
    tools_map = {
        "read_files": read_files,
        "write_file": write_file,
        "apply_search_replace_patch": apply_patch,
        "list_directory": list_directory,
        "run_terminal_command": run_terminal_command,
        "search_symbols_universal": search_symbols,
        "read_file_lines": read_file_lines,
        "web_interact_and_test": web_interact_tool
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