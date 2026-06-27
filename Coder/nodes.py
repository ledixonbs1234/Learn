# nodes.py
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from config import model, sanitize_and_resolve_path
from state import AgentState, WorkspaceDetection, TaskPlan, TaskTriage, Task, WorkspaceDiscoveryState
from tools import (
    GitManager, ReadFileLinesTool, UniversalSymbolSearchTool, WorkspaceTools, 
    ReadFilesTool, WriteFileTool, ApplyPatchTool, 
    ListDirectoryTool, RunTerminalTool,
    get_current_working_directory, check_path_exists, find_project_root
)
workspace_discovery_subgraph = None
# =====================================================================
# CÁC HÀM TIỆN ÍCH BỔ TRỢ CHO BỘ KIỂM THỬ ĐA NGÔN NGỮ THÍCH ỨNG
# =====================================================================

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
    
    # 🛡️ VÁ ĐIỂM KHUYẾT 2: Chuẩn hóa lệnh cho Windows (.cmd, .bat, .exe)
    resolved_executable = shutil.which(executable)
    if not resolved_executable and is_windows:
        for ext in [".cmd", ".bat", ".exe"]:
            if shutil.which(executable + ext):
                cmd[0] = executable + ext
                resolved_executable = shutil.which(cmd[0])
                break
                
    # 🛡️ VÁ ĐIỂM KHUYẾT 3: Nếu thiếu công cụ, trả về mã lỗi đặc biệt (-99) để soft-bypass
    if not resolved_executable:
        return (-99, f"Cảnh báo: Trình biên dịch/phân tích '{executable}' chưa được cài đặt trên hệ thống.")
        
    try:
        # Thực thi lệnh trên terminal
        res = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
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
# CÁC NODES PHỤC VỤ ĐỒ THỊ CON DÒ TÌM WORKSPACE [1.2.2]
# ==========================================
def discovery_agent_node(state: WorkspaceDiscoveryState) -> Dict[str, Any]:
    """Node trí tuệ của Subgraph: Dùng công cụ khảo sát hệ thống để khóa mục tiêu workspace."""
    discovery_tools = [get_current_working_directory, check_path_exists, find_project_root]
    model_with_tools = model.bind_tools(discovery_tools + [WorkspaceDetection])
    
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
                    
    # Giải mã và lấy đường dẫn tuyệt đối chuẩn xác [2]
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
    structured_llm = model.with_structured_output(TaskTriage, method="function_calling")
    
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
        "messages": messages_to_append
    }


def git_setup_node(state: AgentState) -> Dict[str, Any]:
    """Node thiết lập Git: Kiểm tra chủ động sự tồn tại của .git. Nếu không có, bypass an toàn [1]."""
    ws = state["workspace_path"]
    git_dir = Path(ws) / ".git"
    
    # GIẢI PHÁP: Nếu chưa có thư mục .git, bypass toàn bộ [1]
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


# ==========================================
# Cập nhật planner_node với cơ chế bảo vệ (Fallback)
# ==========================================
def planner_node(state: AgentState) -> Dict[str, Any]:
    ws = state["workspace_path"]
    conversation_history = state["messages"]
    existing_plan = state.get("plan", [])
    workspace_context = state.get("workspace_context", "")
    
    # Nếu đã có sẵn kế hoạch từ bước Triage (Fast-track), bỏ qua việc lập kế hoạch
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
        "- Tuyệt đối KHÔNG đưa bước 'Tổng hợp và báo cáo' hoặc 'Viết tệp THONGTIN.md' làm nhiệm vụ cuối cùng trong kế hoạch!\n"
        "Kế hoạch của bạn chỉ tập trung hoàn toàn vào các bước thực thi khảo sát vật lý (analysis) hoặc sửa đổi mã nguồn (development)."
    )
    
    # SỬA ĐỔI: Thêm khối try-except phòng thủ cho mô hình cục bộ
    try:
        plan_output = structured_llm.invoke([
            {"role": "system", "content": system_prompt},
            *conversation_history
        ])
        plan_tasks = getattr(plan_output, "tasks", [])
        task_type = getattr(plan_output, "task_type", "development")
    except Exception as e:
        # Cơ chế Fallback khi LLM gặp sự cố parse cấu trúc
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
    
def analysis_executor_node(state: AgentState) -> Dict[str, Any]:
    ws = state["workspace_path"]
    plan = state["plan"]
    workspace_context = state.get("workspace_context", "")
    messages = list(state["messages"])
    
    eligible_tasks = get_eligible_tasks(plan)
    if not eligible_tasks:
        pending_tasks = [t for t in plan if (t.get("status") if isinstance(t, dict) else getattr(t, "status", None)) == "pending"]
        if pending_tasks:
            eligible_tasks = [pending_tasks[0]]
        else:
            return {"messages": [AIMessage(content="Đã hoàn thành khảo sát toàn bộ các bước.")]}
         
    tasks_str = "\n".join([
        f"- [{getattr(t, 'id', None) or t.get('id')}] {getattr(t, 'description', None) or t.get('description')}"
        for t in eligible_tasks
    ])
    
    read_files = ReadFilesTool(workspace_path=ws)
    list_directory = ListDirectoryTool(workspace_path=ws)
    search_symbols = UniversalSymbolSearchTool(workspace_path=ws)
    read_file_lines = ReadFileLinesTool(workspace_path=ws)
    tools = [read_files, list_directory,search_symbols,read_file_lines]

    model_with_tools = model.bind_tools(tools)
    
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
        "\nHãy sử dụng các công cụ `read_files` và `list_directory` để đọc hiểu cấu trúc hệ thống.\n"
        "YÊU CẦU QUAN TRỌNG:\n"
        "- Bạn chỉ có quyền ĐỌC dữ liệu, tuyệt đối không chỉnh sửa mã nguồn hoặc tự ý tạo tệp tin trong bước này."
    )
    if previous_findings_str:
        system_prompt += previous_findings_str
    
    response = model_with_tools.invoke([SystemMessage(content=system_prompt)] + messages)
    
    if not response.tool_calls:
        findings = []
        if response.content:
            tasks_ids = ", ".join([str(getattr(t, "id", None) or t.get("id")) for t in eligible_tasks])
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
            
        return {
            "messages": [response],
            "plan": updated_plan,
            "step_findings": findings
        }
    else:
        return {"messages": [response]}


# nodes.py

def development_executor_node(state: AgentState) -> Dict[str, Any]:
    ws = state["workspace_path"]
    plan = state["plan"]
    error_logs = state.get("error_logs", "")
    workspace_context = state.get("workspace_context", "")
    messages = list(state["messages"])
        
    eligible_tasks = get_eligible_tasks(plan)
    if not eligible_tasks:
        pending_tasks = [t for t in plan if (t.get("status") if isinstance(t, dict) else getattr(t, "status", None)) == "pending"]
        if pending_tasks:
            eligible_tasks = [pending_tasks[0]]
        else:
            return {"messages": [AIMessage(content="Đã hoàn thành toàn bộ các bước phát triển.")]}
         
    tasks_str = "\n".join([
        f"- [{getattr(t, 'id', None) or t.get('id')}] {getattr(t, 'description', None) or t.get('description')}"
        for t in eligible_tasks
    ])
    
    read_files = ReadFilesTool(workspace_path=ws)
    write_file = WriteFileTool(workspace_path=ws)
    apply_patch = ApplyPatchTool(workspace_path=ws)
    search_symbols = UniversalSymbolSearchTool(workspace_path=ws)
    list_directory = ListDirectoryTool(workspace_path=ws)
    run_terminal_command = RunTerminalTool(workspace_path=ws)

    read_file_lines = ReadFileLinesTool(workspace_path=ws)
    
    tools = [read_files, write_file, apply_patch, list_directory, run_terminal_command,search_symbols,read_file_lines]
    model_with_tools = model.bind_tools(tools)
    
    system_prompt = (
        "Bạn là một kỹ sư phần mềm thực thi chuyên nghiệp chuyên sửa lỗi và viết mới mã nguồn (Write-Access Mode).\n"
        f"Nhiệm vụ: Bạn đang thực hiện đồng thời các nhiệm vụ sau:\n{tasks_str}\n"
        f"Thư mục làm việc: {ws}\n"
    )
    if workspace_context:
        system_prompt += f"\n--- NGỮ CẢNH HỆ THỐNG HIỆN TẠI (THONGTIN.md) ---\n{workspace_context}\n"
        
    system_prompt += (
        "\nHãy sử dụng các công cụ để giải quyết nhiệm vụ.\n"
        "\n⚠️ QUY TẮC CHỈNH SỬA TỆP (BẮT BUỘC):\n"
        "1. Đối với file trên 300 dòng: BẮT BUỘC dùng `apply_search_replace_patch` để áp dụng bản vá, cấm ghi đè bừa bãi.\n"
        "2. Công cụ `write_file` chỉ dùng khi tạo mới hoặc sửa các tệp ngắn dưới 300 dòng."
    )
    
    input_messages = [SystemMessage(content=system_prompt)]
    if error_logs:
        input_messages.append(HumanMessage(content=f"LƯU Ý SỬA LỖI TỪ LƯỢT CHẠY TRƯỚC:\n{error_logs}\nHãy sửa triệt để."))
        
    response = model_with_tools.invoke(input_messages + messages)
    
    if not response.tool_calls:
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

        # 🛡️ BẢN VÁ: Loại bỏ việc ghi đè "attempts": 0 và "error_logs": "" để bảo vệ bộ đếm của Tester
        return {
            "messages": [response],
            "plan": updated_plan,
            "last_executed_task_ids": list(eligible_ids)
        }
    else:
        return {"messages": [response]}


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
        "search_symbols_universal":search_symbols,
        "read_file_lines": read_file_lines
    }
    
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {}
        
    tool_messages = []
    modified_files = list(state.get("modified_files", []))
    
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"] or {}
        tool_id = tool_call["id"]
        
        tool_instance = tools_map.get(tool_name)
        if not tool_instance:
            result = f"Lỗi: Không tìm thấy công cụ '{tool_name}'."
        else:
            try:
                result = tool_instance.invoke(tool_args)
                if tool_name in ["write_file", "apply_search_replace_patch"]:
                    file_path = tool_args.get("file_path")
                    if file_path:
                        try:
                            safe_path = sanitize_and_resolve_path(ws, file_path, create_parent=True)
                            if str(safe_path) not in modified_files:
                                modified_files.append(str(safe_path))
                        except Exception:
                            pass
            except Exception as e:
                result = f"Lỗi thực thi công cụ '{tool_name}': {str(e)}"
                
        tool_messages.append(ToolMessage(content=str(result), name=tool_name, tool_call_id=tool_id))
        
    return {
        "messages": tool_messages,
        "modified_files": modified_files
    }


def tester_node(state: AgentState) -> Dict[str, Any]:
    modified_files = state.get("modified_files", [])
    attempts = state.get("attempts", 0)
    plan = state["plan"]
    ws = state["workspace_path"]
    last_executed_ids = state.get("last_executed_task_ids", [])
    
    errors = []
    warnings = [] # Chứa các cảnh báo thiếu công cụ để không phạt LLM
    workspace_root = Path(ws).expanduser().resolve()
    
    # Phân loại tệp đã sửa
    files_by_ext: Dict[str, List[Path]] = {}
    for f_path_str in modified_files:
        try:
            p = Path(f_path_str).resolve()
            if p.exists() and p.is_file():
                ext = p.suffix.lower()
                files_by_ext.setdefault(ext, []).append(p)
        except Exception:
            pass

    for ext, files in files_by_ext.items():
        
        # 1. NGÔN NGỮ PYTHON (.py)
        if ext == ".py":
            for f in files:
                code, output = execute_validation_cmd(["python", "-m", "py_compile", str(f)], workspace_root)
                if code == -99:
                    warnings.append(output)
                elif code != 0:
                    errors.append(f"❌ [Lỗi Cú Pháp Python] tại `{f.name}`:\n{clean_compiler_logs(output)}")

        # 2. DART / FLUTTER (.dart)
        elif ext == ".dart":
            for f in files:
                target_dir = find_nearest_config(f, "pubspec.yaml") or workspace_root
                code, output = execute_validation_cmd(["dart", "analyze"], target_dir)
                if code == -99:
                    warnings.append(f"{output} (Bỏ qua kiểm tra tĩnh cho `{f.name}`)")
                elif code != 0:
                    errors.append(f"❌ [Lỗi Dart Analysis] tại sub-project `{target_dir.name}`:\n{clean_compiler_logs(output)}")
                    break

        # 3. TYPESCRIPT / JAVASCRIPT (.ts, .tsx, .js, .jsx)
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

        # 4. RUST (.rs)
        elif ext == ".rs":
            for f in files:
                target_dir = find_nearest_config(f, "Cargo.toml") or workspace_root
                code, output = execute_validation_cmd(["cargo", "check"], target_dir)
                if code == -99:
                    warnings.append(f"{output} (Bỏ qua biên dịch Rust cho `{f.name}`)")
                elif code != 0:
                    errors.append(f"❌ [Lỗi Biên Dịch Rust] tại `{target_dir.name}`:\n{clean_compiler_logs(output)}")
                    break

        # 5. GO (.go)
        elif ext == ".go":
            for f in files:
                target_dir = find_nearest_config(f, "go.mod") or workspace_root
                code, output = execute_validation_cmd(["go", "vet", "./..."], target_dir)
                if code == -99:
                    warnings.append(f"{output} (Bỏ qua kiểm tra tĩnh Go cho `{f.name}`)")
                elif code != 0:
                    errors.append(f"❌ [Lỗi Tĩnh Go Vet] tại `{target_dir.name}`:\n{clean_compiler_logs(output)}")
                    break

    # =====================================================================
    # XỬ LÝ KẾT QUẢ VÀ TRẢ VỀ TRẠNG THÁI (ĐÃ SỬA LỖI TÍCH LŨY FILE)
    # =====================================================================
    warning_msg = ""
    if warnings:
        warning_msg = "⚠️ **Cảnh báo môi trường:**\n" + "\n".join([f"- {w}" for w in warnings]) + "\n\n"

    # Nếu có lỗi cú pháp/biên dịch thực tế từ mã nguồn
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
                "modified_files": [], # 🛡️ VÁ ĐIỂM KHUYẾT 1: Reset danh sách file khi bỏ qua để tránh lỗi tích lũy
                "messages": [AIMessage(content=f"{warning_msg}❌ Đã vượt quá giới hạn số lần sửa lỗi tự động. Hệ thống sẽ bỏ qua lỗi để tiếp tục tiến trình.")]
            }
                
    # Trường hợp thành công hoàn toàn hoặc kích hoạt Soft-Bypass thành công
    success_content = f"{warning_msg}✅ [Vòng kiểm thử thành công] Toàn bộ mã nguồn đã vượt qua kiểm tra tĩnh và biên dịch."
    return {
        "error_logs": "",
        "attempts": 0,
        "modified_files": [], # 🛡️ VÁ ĐIỂM KHUYẾT 1: RESET SẠCH SẼ ĐỂ NHIỆM VỤ TIẾP THEO KHÔNG BỊ TRÙNG LẶP KIỂM TRA
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
        "Hãy tổng hợp thông tin khảo sát sau thành một tài liệu 'THONGTIN.md' duy nhất, thật cô đọng và súc tích.\n"
        "TUYỆT ĐỐI KHÔNG chèn mã nguồn dài dòng."
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
            
            tools_mgr = WorkspaceTools(ws)
            tools_mgr.write_file("THONGTIN.md", cleaned_md)
            
            # CHỈ CHẠY LỆNH GIT KHI DỰ ÁN SỬ DỤNG GIT [1]
            if git_branch != "no_git":
                git_manager = GitManager(ws)
                git_manager._run_cmd(["git", "add", "THONGTIN.md"], ignore_error=True)
            
            return {
                "workspace_context": cleaned_md,
                "messages": [AIMessage(content="**Tổng hợp tài liệu hoàn tất:** Đã biên dịch tri thức khảo sát thành tệp `THONGTIN.md` thành công tại gốc dự án.")]
            }
    except Exception as e:
        return {"messages": [AIMessage(content=f"Cảnh báo: Có lỗi xảy ra khi tổng hợp tệp THONGTIN.md: {str(e)}")]}
    return {}


def commit_node(state: AgentState) -> Dict[str, Any]:
    ws = state["workspace_path"]
    task_type = state.get("task_type", "development")
    git_branch = state.get("git_branch", "no_git")
    
    # GIẢI PHÁP: Nếu không dùng Git, kết thúc êm đẹp, bỏ qua Git Commit [1]
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