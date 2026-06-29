# nodes.py
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from config import model, sanitize_and_resolve_path,fast_model
from state import AgentState, PlanUpdate, WorkspaceDetection, TaskPlan, TaskTriage, Task, WorkspaceDiscoveryState
from tools import (
    GitManager, ReadFileLinesTool, UniversalSymbolSearchTool, WorkspaceTools, 
    ReadFilesTool, WriteFileTool, ApplyPatchTool, 
    ListDirectoryTool, RunTerminalTool,
    get_current_working_directory, check_path_exists, find_project_root, get_markdown_language
)
workspace_discovery_subgraph = None

def sanitize_llm_response_content(response: AIMessage) -> AIMessage:
    """
    Hậu xử lý tin nhắn của LLM: Loại bỏ triệt để các khối <thinking>...</thinking>
    hoặc <thought>...</thought> nếu mô hình tự ý sinh ra để tiết kiệm token cho các vòng sau.
    """
    if not response or not isinstance(response, AIMessage):
        return response
        
    content_str = response.content
    if isinstance(content_str, str) and content_str.strip():
        # Xóa sạch thẻ <thinking>...</thinking> và <thought>...</thought> kèm nội dung bên trong
        cleaned_content = re.sub(r"<thinking>.*?</thinking>", "", content_str, flags=re.DOTALL)
        cleaned_content = re.sub(r"<thought>.*?</thought>", "", cleaned_content, flags=re.DOTALL)
        
        # Cập nhật lại nội dung sạch cho tin nhắn
        response.content = cleaned_content.strip()
        
    return response

def compact_reading_tool_messages(messages: List[BaseMessage]) -> List[BaseMessage]:
    """
    Quét qua lịch sử tin nhắn và thu gọn nội dung của các ToolMessage đọc file.
    Thay thế hàng ngàn dòng code thô bằng một thông báo định danh siêu nhẹ,
    giúp giảm 95% lượng token lãng phí mà không phá vỡ tính toàn vẹn của chuỗi Tool Call.
    """
    compacted_messages = []
    
    for msg in messages:
        # Chỉ can thiệp vào các ToolMessage thuộc nhóm đọc/khảo sát file
        if msg.type == "tool" and msg.name in ["read_files", "read_file_lines", "search_symbols_universal"]:
            content_str = str(msg.content)
            found_files = []
            
            # 1. Trích xuất tên file từ cấu trúc đầu ra của read_files
            matches_read = re.findall(r"=== TỆP TIN:\s*[`']?([^`'\n]+)[`']?\s*===", content_str)
            if matches_read:
                found_files.extend(matches_read)
                
            # 2. Trích xuất tên file từ cấu trúc đầu ra của read_file_lines
            matches_lines = re.findall(r"=== NỘI DUNG TỆP KHOANH VÙNG:\s*([^ \n]+)", content_str)
            if matches_lines:
                found_files.extend(matches_lines)
                
            # 3. Trích xuất tên file từ cấu trúc đầu ra của search_symbols_universal
            matches_symbols = re.findall(r"📁\s*`([^`\n]+)`", content_str)
            if matches_symbols:
                found_files.extend(matches_symbols)
                
            # Tạo chuỗi thông tin định danh ngắn gọn
            file_info = f" của tệp {', '.join([f'`{f}`' for f in found_files])}" if found_files else ""
            
            # Tạo ToolMessage mới kế thừa nguyên vẹn ID nhưng có nội dung tối giản cực hạn
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
# =====================================================================
# CÁC HÀM TIỆN ÝCH BỔ TRỢ CHO BỘ KIỂM THỬ ĐA NGÔN NGỮ THÍCH ỨNG
# =====================================================================
def clear_compiler_cache(workspace_path: Path, ext: str):
    """
    Dọn dẹp vật lý các tệp tin cache biên dịch tạm thời trên đĩa cứng 
    để đảm bảo kết quả kiểm thử hoàn toàn là của mã nguồn mới nhất.
    """
    try:
        if ext == ".py":
            # Dọn dẹp __pycache__ và các tệp .pyc cục bộ
            for pycache in workspace_path.rglob("__pycache__"):
                if pycache.is_dir():
                    shutil.rmtree(pycache, ignore_errors=True)
            for pyc in workspace_path.rglob("*.pyc"):
                if pyc.is_file():
                    pyc.unlink(missing_ok=True)
                    
        elif ext in [".ts", ".tsx", ".js", ".jsx"]:
            # Dọn dẹp cache của TypeScript nếu có
            ts_cache = workspace_path / "node_modules" / ".cache"
            if ts_cache.exists():
                shutil.rmtree(ts_cache, ignore_errors=True)
                
        elif ext == ".rs":
            # Dọn dẹp cache thô của Cargo để cargo check luôn chạy thực tế
            # Lưu ý: Không chạy 'cargo clean' vì sẽ tốn rất nhiều thời gian compile lại từ đầu
            pass
            
    except Exception as e:
        print(f"[Cảnh báo] Không thể dọn dẹp cache biên dịch: {str(e)}")
def find_nearest_config(start_path: Path, config_name: str, max_depth: int = 5) -> Optional[Path]:
    """
    Quét ngược lên các thư mục cha từ start_path để tìm tệp cấu hình (ví dụ: package.json, Cargo.toml).
    Giúp xác định đúng gốc của sub-project trong monorepo hoặc dự án phân tầng.
    """
    current = start_path.resolve()
    if current.is_file():
        current = current.parent
        
    for _ in range(max_depth):
        target = current / config_name
        if target.exists() and target.is_file():
            return current
        if current.parent == current: # Đã chạm gốc hệ thống ổ đĩa
            break
        current = current.parent
    return None


def execute_validation_cmd(cmd: List[str], cwd: Path, timeout: int = 30) -> Tuple[int, str]:
    """
    Thực thi lệnh kiểm thử đa nền tảng an toàn. 
    Tự động xử lý bẫy đuôi tệp tin lệnh (.cmd/.bat) trên Windows.
    """
    executable = cmd[0]
    is_windows = platform.system() == "Windows"
    
    # Chuẩn hóa lệnh cho Windows (.cmd, .bat, .exe)
    resolved_executable = shutil.which(executable)
    if not resolved_executable and is_windows:
        for ext in [".cmd", ".bat", ".exe"]:
            if shutil.which(executable + ext):
                cmd[0] = executable + ext
                resolved_executable = shutil.which(cmd[0])
                break
                
    # Nếu thiếu công cụ, trả về mã lỗi đặc biệt (-99) để soft-bypass
    if not resolved_executable:
        return (-99, f"Cảnh báo: Trình biên dịch/phân tích '{executable}' chưa được cài đặt trên hệ thống.")
        
    try:
        
        env_copy = os.environ.copy()
        env_copy["PYTHONIOENCODING"] = "utf-8"
        env_copy["PYTHONUTF8"] = "1"
        # Thực thi lệnh trên terminal
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
    """
    Nén log lỗi để giữ lại các thông tin giá trị nhất cho LLM, loại bỏ các dòng log thông tin rác.
    """
    lines = raw_logs.splitlines()
    filtered_lines = []
    
    # Danh sách các từ khóa báo lỗi phổ biến của các trình biên dịch khác nhau
    error_keywords = ["error", "fail", "exception", "cause", "unhandled", "invalid", "undefined"]
    
    for line in lines:
        clean_line = line.strip()
        if not clean_line:
            continue
            
        # Ưu tiên các dòng chứa từ khóa báo lỗi hoặc chỉ định số dòng (ví dụ: :25:10 hoặc .dart:40)
        has_error_kw = any(kw in clean_line.lower() for kw in error_keywords)
        has_line_indicator = ":" in clean_line or ".dart" in clean_line or ".py" in clean_line or ".ts" in clean_line
        
        if has_error_kw or has_line_indicator:
            filtered_lines.append(line)
            
    if not filtered_lines:
        # Nếu bộ lọc quá chặt làm mất hết thông tin, trả về 20 dòng đầu và cuối để đảm bảo an toàn
        if len(lines) > 40:
            return "\n".join(lines[:20] + ["... [Đã lược bớt các dòng ở giữa] ..."] + lines[-20:])
        return raw_logs
        
    return "\n".join(filtered_lines)


# ==========================================
# NÚT WRAPPER CHO SUBGRAPH (STATE ISOLATION)
# ==========================================
def detect_workspace_wrapper_node(state: AgentState) -> Dict[str, Any]:
    """
    Node wrapper kích hoạt Subgraph một cách độc lập.
    Giúp cô lập lịch sử hội thoại, tránh đưa các message gọi tool khảo sát hệ thống vào Parent State.
    """
    if workspace_discovery_subgraph is None:
        raise RuntimeError("workspace_discovery_subgraph chưa được liên kết.")
    
    # Lấy tin nhắn cuối cùng của người dùng để làm đầu vào cho Subgraph
    user_msg = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage) or getattr(msg, "type", None) == "human":
            user_msg = msg
            break
    if not user_msg:
        user_msg = HumanMessage(content="Xác định thư mục làm việc hiện tại.")

    # Khởi tạo trạng thái riêng biệt cho Subgraph
    sub_state = {
        "messages": [user_msg],
        "workspace_path": state.get("workspace_path", ".")
    }
    
    # Thực thi Subgraph một cách cô lập
    result = workspace_discovery_subgraph.invoke(sub_state)
    
    # Chỉ trích xuất kết quả cần thiết trả lại cho Parent Graph
    final_path = result.get("workspace_path", ".")
    final_msg = AIMessage(content=f"🔍 Định vị không gian làm việc thành công: `{final_path}`")
    
    return {
        "workspace_path": final_path,
        "messages": [final_msg] # Chỉ trả về 1 tin nhắn sạch sẽ, loại bỏ toàn bộ tool_calls nháp
    }


# Hàm tiện ích nội bộ lọc các task đủ điều kiện (DAG)
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


# ==========================================
# CÁC NODES PHỤC VỤ ĐỒ THỊ CON DÒ TÌM WORKSPACE
# ==========================================
def discovery_agent_node(state: WorkspaceDiscoveryState) -> Dict[str, Any]:
    """Node trí tuệ của Subgraph: Dùng công cụ khảo sát hệ thống để khóa mục tiêu workspace."""
    discovery_tools = [get_current_working_directory, check_path_exists, find_project_root]
    model_with_tools = fast_model.bind_tools(discovery_tools + [WorkspaceDetection])
    
    system_prompt = (
        "Bạn là một Agent chuyên nghiệp định vị thư mục làm việc (Workspace).\n"
        "Nhiệm vụ: Sử dụng các công cụ hệ thống để định vị chính xác đường dẫn vật lý tuyệt đối của thư mục dự án.\n"
        "⚠️ QUY TẮC CẤM ĐOÁN MÒ:\n"
        "1. Bạn tuyệt đối không được đoán mò đường dẫn. Hãy sử dụng các công cụ kiểm tra để khảo sát thực tế.\n"
        "2. Đầu tiên, hãy gọi `get_current_working_directory` để biết môi trường Agent đang đứng.\n"
        "3. Nếu người dùng nhập đường dẫn (ví dụ: 'Desktop', '~/Desktop'), hãy gọi `check_path_exists` để kiểm định vật lý.\n"
        "4. Nếu muốn tìm gốc dự án hiện tại, hãy sử dụng `find_project_root`.\n"
        "5. Khi đã định vị và xác minh chắc chắn đường dẫn tồn tại, hãy gọi công cụ kết thúc `WorkspaceDetection`."
    )
    
    messages = list(state["messages"])
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=system_prompt)] + messages
        
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}


def discovery_tool_node(state: WorkspaceDiscoveryState) -> Dict[str, Any]:
    """Node thực thi các công cụ khảo sát hệ thống thực tế cho Subgraph."""
    last_msg = state["messages"][-1]
    if not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
        return {}
        
    tools_map = {
        "get_current_working_directory": get_current_working_directory,
        "check_path_exists": check_path_exists,
        "find_project_root": find_project_root
    }
    
    tool_messages = []
    for tool_call in last_msg.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"] or {}
        tool_id = tool_call["id"]
        
        tool_instance = tools_map.get(tool_name)
        if not tool_instance:
            result = f"Lỗi: Không tìm thấy công cụ khảo sát hệ thống '{tool_name}'."
        else:
            try:
                result = tool_instance.invoke(tool_args)
            except Exception as e:
                result = f"Lỗi thực thi công cụ '{tool_name}': {str(e)}"
                
        tool_messages.append(ToolMessage(content=str(result), name=tool_name, tool_call_id=tool_id))
        
    return {"messages": tool_messages}


def discovery_finalize_node(state: WorkspaceDiscoveryState) -> Dict[str, Any]:
    """Node hoàn tất: Trích xuất kết quả cuối cùng từ cuộc gọi WorkspaceDetection và đóng gói."""
    workspace_path = "."
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tool_call in msg.tool_calls:
                if tool_call["name"] == "WorkspaceDetection":
                    workspace_path = tool_call["args"].get("workspace_path", ".")
                    break
                    
    final_path = str(Path(workspace_path).expanduser().resolve())
    return {
        "workspace_path": final_path,
        "finished": True,
        "messages": [AIMessage(content=f"🔍 Hệ thống đã xác minh thực tế và thiết lập workspace tại: `{final_path}`")]
    }


# ==========================================
# CÁC NODES PHỤC VỤ ĐỒ THỊ CHÍNH (MAIN GRAPH)
# ==========================================
def context_loader_node(state: AgentState) -> Dict[str, Any]:
    """Node tải ngữ cảnh: Chỉ thực hiện I/O đọc tệp tin THONGTIN.md để lấy thông tin hệ thống."""
    ws = state["workspace_path"]
    workspace_context = ""
    thongtin_path = Path(ws) / "THONGTIN.md"
    if thongtin_path.exists():
        try:
            workspace_context = thongtin_path.read_text(encoding="utf-8")
        except Exception as e:
            workspace_context = f"Lỗi khi đọc file THONGTIN.md: {str(e)}"
            
    return {
        "workspace_context": workspace_context,
        "git_branch": "no_git",  # Mặc định là no_git để an toàn [1]
        "messages": [AIMessage(content="📋 Đã tải xong ngữ cảnh thông tin dự án từ tệp `THONGTIN.md`.")]
    }


def triage_node(state: AgentState) -> Dict[str, Any]:
    """Node phân loại (Triage): Gọi LLM tách biệt để định hướng nhiệm vụ đơn giản / phức tạp."""
    messages = state["messages"]
    user_msg = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) or getattr(msg, "type", None) == "human":
            user_msg = msg
            break
            
    if not user_msg:
        user_msg = messages[0]
        
    user_query = user_msg.content
    structured_llm = fast_model.with_structured_output(TaskTriage, method="function_calling")
    
    system_prompt = (
        "Bạn là một điều phối viên Agent thông minh. Hãy phân tích yêu cầu của người dùng "
        "để xác định xem đây là một yêu cầu đơn giản (chỉ cần chỉnh sửa trực tiếp 1-2 file) "
        "hay một yêu cầu phức tạp (cần lên kế hoạch khảo sát, phát triển nhiều bước)."
    )
    
    try:
        triage_output = structured_llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ])
        is_simple = getattr(triage_output, "is_simple", False)
        task_type = getattr(triage_output, "task_type", "development")
    except Exception:
        is_simple = False
        task_type = "development"
        
    plan_tasks = []
    messages_to_append = []
    
    if is_simple:
        plan_tasks = [Task(
            id="T1", 
            description=f"Xử lý trực tiếp yêu cầu của người dùng: {user_query}", 
            dependencies=[], 
            status="pending"
        )]
        messages_to_append.append(AIMessage(content="📋 Kích hoạt chế độ **Fast-Track (Nhiệm vụ đơn giản)**. Bỏ qua bước lập kế hoạch chi tiết."))
    else:
        messages_to_append.append(AIMessage(content="📋 Nhận diện tác vụ phức tạp, chuẩn bị chuyển tiếp tới Planner."))
        
    return {
        "plan": plan_tasks,
        "task_type": task_type,
        "last_executed_task_ids": [],
        "messages": messages_to_append,
        "replanning_count": 0,
        "modified_files": [],     
        "error_logs": "",         
        "step_findings": [],      
        "is_simple": is_simple,  # <--- TRẢ VỀ GIÁ TRỊ STATE ĐỂ LƯU TRỮ
    }


def git_setup_node(state: AgentState) -> Dict[str, Any]:
    """Node thiết lập Git: Kiểm tra chủ động sự tồn tại của .git. Nếu không có, bypass an toàn [1]."""
    ws = state["workspace_path"]
    git_dir = Path(ws) / ".git"
    
    if not git_dir.exists():
        return {
            "git_branch": "no_git",
            "messages": [AIMessage(content="ℹ️ Không phát hiện Git repository. Kích hoạt chế độ Sửa đổi trực tiếp (Bypass Git) [1].")]
        }
        
    try:
        git_manager = GitManager(ws)
        branch = git_manager.init_and_prepare_branch()
        return {
            "git_branch": branch,
            "messages": [AIMessage(content=f"Đã cấu hình nhánh Git hoạt động: `{branch}`")]
        }
    except Exception as e:
        return {
            "git_branch": "no_git",
            "messages": [AIMessage(content=f"⚠️ Có lỗi xảy ra khi nạp Git ({str(e)}). Tự động chuyển sang chế độ Sửa đổi trực tiếp [1].")]
        }


# =====================================================================
# CHỈNH SỬA 1: Cấu trúc lại Planner_Node để tối ưu hóa Hợp đồng Nhiệm vụ
# =====================================================================
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
        "- Kế hoạch của bạn chỉ tập trung hoàn toàn vào các bước thực thi khảo sát vật lý (analysis) hoặc sửa đổi mã nguồn (development)."
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

# =====================================================================
# HỢP NHẤT: Executor duy nhất tự động thích ứng theo Trạng thái (Polymorphic)
# =====================================================================
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

    # ⚙️ PHÂN TÁCH CẤU HÌNH DỰA TRÊN TASK TYPE TRONG CÙNG MỘT NODE
    if task_type == "analysis":
        # Công cụ chỉ đọc dành cho Khảo sát
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
        # Công cụ có quyền ghi dành cho Phát triển
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
                
        read_files = ReadFilesTool(workspace_path=ws)
        write_file = WriteFileTool(workspace_path=ws)
        apply_patch = ApplyPatchTool(workspace_path=ws)
        search_symbols = UniversalSymbolSearchTool(workspace_path=ws)
        list_directory = ListDirectoryTool(workspace_path=ws)
        run_terminal_command = RunTerminalTool(workspace_path=ws)
        read_file_lines = ReadFileLinesTool(workspace_path=ws)
        
        tools = [read_files, write_file, apply_patch, list_directory, run_terminal_command, search_symbols, read_file_lines]
        
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



# =====================================================================
# CHỈNH SỬA 4: Ngăn chặn lấn ranh trong các lần điều chỉnh Kế hoạch (Replanner)
# =====================================================================
def replanner_node(state: AgentState) -> Dict[str, Any]:
    """
    Node trí tuệ trung gian: Đánh giá tiến độ thực tế, phát hiện rào cản 
    và tiến hành cập nhật/thích ứng đồ thị nhiệm vụ (DAG) theo thời gian thực.
    """
    replanning_count = state.get("replanning_count", 0)
    
    # 🛡️ CHỐT CHẶN BẢO VỆ: Nếu Re-plan quá 5 lần, dừng lại để tránh cạn kiệt tài nguyên
    if replanning_count >= 5:
        return {
            "messages": [AIMessage(content="🚨 **[Hệ thống tự động dừng]** Phát hiện vòng lặp lập kế hoạch quá nhiều lần (Vượt giới hạn 5 lần). Chuyển tiếp tới bước tiếp theo để tránh lặp vô hạn.")]
        }
    ws = state["workspace_path"]
    plan = state["plan"]
    messages = state["messages"]
    task_type = state.get("task_type", "development")
    workspace_context = state.get("workspace_context", "")
    error_logs = state.get("error_logs", "")
    
    # Định dạng trực quan danh sách nhiệm vụ hiện tại để LLM dễ phân tích
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
    
    # Ép cấu trúc đầu ra bằng function calling tương thích local model
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


# nodes.py (Cập nhật tool_node để tự động cập nhật registry)
def tool_node(state: AgentState) -> Dict[str, Any]:
    ws = state["workspace_path"]
    read_files = ReadFilesTool(workspace_path=ws)
    write_file = WriteFileTool(workspace_path=ws)
    apply_patch = ApplyPatchTool(workspace_path=ws)
    list_directory = ListDirectoryTool(workspace_path=ws)
    run_terminal_command = RunTerminalTool(workspace_path=ws)
    search_symbols = UniversalSymbolSearchTool(workspace_path=ws)
    read_file_lines = ReadFileLinesTool(workspace_path=ws)
    
    tools_map = {
        "read_files": read_files,
        "write_file": write_file,
        "apply_search_replace_patch": apply_patch,
        "list_directory": list_directory,
        "run_terminal_command": run_terminal_command,
        "search_symbols_universal": search_symbols,
        "read_file_lines": read_file_lines
    }
    
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {}
        
    tool_messages = []
    modified_files = list(state.get("modified_files", []))
    file_registry = dict(state.get("file_registry", {}))
    
    # Tập hợp các file bị tác động (cả đọc lẫn ghi) trong lượt này
    impacted_files = set()
    
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"] or {}
        tool_id = tool_call["id"]
        
        # Trích xuất đường dẫn file từ đối số của tool
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
        
    # ĐỒNG BỘ HÓA REGISTRY: Đọc thực tế trên đĩa cứng để cập nhật nội dung mới nhất
    for file_path in impacted_files:
        try:
            safe_path = sanitize_and_resolve_path(ws, file_path, create_parent=False)
            if safe_path.exists() and safe_path.is_file():
                # Đọc nội dung thực tế trên đĩa
                current_content = safe_path.read_text(encoding="utf-8")
                # Đồng bộ vào registry
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
    
    # Phân loại tệp đã sửa và tiến hành DỌN DẸP CACHE trước khi test
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
            import sys # 🌟 THÊM MỚI: Import sys để lấy trình thông dịch hiện tại
            for f in files:
                if f.name.startswith("test_") or f.name.endswith("_test.py"):
                    # 🌟 THAY ĐỔI: Dùng sys.executable thay vì "python" để giữ đúng môi trường venv
                    code, output = execute_validation_cmd([sys.executable, str(f)], workspace_root)
                    if code == -99:
                        warnings.append(output)
                    elif code != 0:
                        errors.append(f"❌ [Lỗi Thực Thi Unit Test Python] tại tệp `{f.name}`:\n{clean_compiler_logs(output)}")
                else:
                    # Tương tự cho lệnh kiểm tra cú pháp py_compile
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
        response = model.invoke([
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
            
            # =====================================================================
            # THAY ĐỔI: Chỉ ghi file vật lý khi môi trường có Git hoạt động
            # =====================================================================
            if git_branch != "no_git":
                # Thư mục có Git: Cho phép ghi file vật lý xuống đĩa cứng
                tools_mgr = WorkspaceTools(ws)
                tools_mgr.write_file("THONGTIN.md", cleaned_md)
                
                git_manager = GitManager(ws)
                git_manager._run_cmd(["git", "add", "THONGTIN.md"], ignore_error=True)
                
                message_content = (
                    "**Tổng hợp tài liệu hoàn tất:** Đã biên dịch tri thức khảo sát, "
                    "lưu vật lý thành tệp `THONGTIN.md` và đưa vào Git staging thành công."
                )
            else:
                # Chế độ Không-Git: Chỉ lưu trữ ảo trong State để tránh tạo file rác bừa bãi
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
            "messages": [AIMessage(content="Đã hoàn thành toàn bộ yêu cầu của bạn. Chế độ Không-Git được kích hoạt, bỏ qua commit [1].")]
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