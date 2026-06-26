# nodes.py
import subprocess
from pathlib import Path
from typing import Dict, Any, Union, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from config import model, sanitize_and_resolve_path
from state import AgentState, WorkspaceDetection, TaskPlan, TaskTriage, Task
from tools import (
    GitManager, WorkspaceTools, 
    ReadFilesTool, WriteFileTool, ApplyPatchTool, 
    ListDirectoryTool, RunTerminalTool, check_path_exists, find_project_root, get_current_working_directory
)

# Hàm tiện ích nội bộ để lọc các task đủ điều kiện thực thi (DAG)
def get_eligible_tasks(plan: List[Any]) -> List[Any]:
    completed_ids = set()
    for t in plan:
        # Kiểm tra kiểu dữ liệu một cách minh bạch
        t_id = t.get("id") if isinstance(t, dict) else getattr(t, "id", None)
        t_status = t.get("status") if isinstance(t, dict) else getattr(t, "status", None)
        
        if t_status == "completed":
            completed_ids.add(t_id)
            
    eligible = []
    for t in plan:
        t_status = t.get("status") if isinstance(t, dict) else getattr(t, "status", None)
        # Sử dụng toán tử ternary thay vì toán tử 'or' để tránh Falsy trap của danh sách rỗng []
        t_deps = t.get("dependencies", []) if isinstance(t, dict) else (getattr(t, "dependencies", None) or [])
        
        if t_status == "pending":
            if all(dep in completed_ids for dep in t_deps):
                eligible.append(t)
    return eligible


def detect_workspace_node(state: AgentState) -> Dict[str, Any]:
    # 1. Nếu state đã có sẵn workspace được định nghĩa trước, sử dụng luôn
    if state.get("workspace_path"):
        resolved_ws = str(Path(state["workspace_path"]).expanduser().resolve())
        return {
            "workspace_path": resolved_ws,
            "messages": [AIMessage(content=f"Sử dụng workspace cấu hình sẵn: `{resolved_ws}`")]
        }

    # 2. Tìm tin nhắn gần nhất của người dùng
    user_msg = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage) or getattr(msg, "type", None) == "human":
            user_msg = msg
            break
            
    if not user_msg:
        user_msg = state["messages"][-1]

    # 3. Chuẩn bị danh sách công cụ dò tìm và cấu trúc đầu ra kết thúc (WorkspaceDetection)
    discovery_tools = [get_current_working_directory, check_path_exists, find_project_root]
    
    # Ràng buộc cả công cụ dò tìm và mô hình Pydantic đầu ra vào LLM
    model_with_tools = model.bind_tools(discovery_tools + [WorkspaceDetection])
    
    system_prompt = (
        "Bạn là một Agent chuyên nghiệp chịu trách nhiệm dò tìm và thiết lập thư mục làm việc (Workspace) chính xác trên máy tính.\n"
        "Nhiệm vụ của bạn là phân tích yêu cầu của người dùng để xác định thư mục họ muốn xử lý.\n\n"
        "⚠️ QUY TẮC TUÂN THỦ THỰC TẾ (CHỐNG ĐOÁN MÒ):\n"
        "1. Bạn TUYỆT ĐỐI KHÔNG ĐƯỢC tự ý đoán mò đường dẫn. Hãy sử dụng các công cụ được cung cấp để khảo sát hệ thống.\n"
        "2. Đầu tiên, hãy gọi `get_current_working_directory` để biết máy chủ của bạn đang chạy ở thư mục nào.\n"
        "3. Nếu người dùng nhắc tới một đường dẫn (ví dụ: 'Desktop', '~/Desktop', 'my-project'), hãy gọi `check_path_exists` để kiểm tra xem nó có tồn tại vật lý và đường dẫn tuyệt đối của nó là gì.\n"
        "4. Nếu người dùng muốn làm việc với dự án hiện hành, hãy gọi `find_project_root` để tìm đúng thư mục gốc chứa các tệp cấu hình của dự án.\n"
        "5. Sau khi đã xác minh chắc chắn đường dẫn tồn tại vật lý, hãy gọi công cụ `WorkspaceDetection` với đường dẫn tuyệt đối đã được xác minh đó để hoàn tất nhiệm vụ."
    )
    
    local_history = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_msg.content)
    ]
    
    # Vòng lặp tối đa 5 bước để LLM dò tìm thực tế
    max_steps = 5
    for _ in range(max_steps):
        response = model_with_tools.invoke(local_history)
        local_history.append(response)
        
        # Nếu LLM trả lời trực tiếp mà không gọi công cụ nào, thoát vòng lặp
        if not response.tool_calls:
            break
            
        # Kiểm tra xem LLM đã gọi công cụ "WorkspaceDetection" (đã ra quyết định cuối) chưa
        finish_call = None
        for tool_call in response.tool_calls:
            if tool_call["name"] == "WorkspaceDetection":
                finish_call = tool_call
                break
                
        if finish_call:
            # LLM đã tìm ra và kiểm chứng đường dẫn thành công
            raw_detected_path = finish_call["args"].get("workspace_path", ".")
            final_ws = str(Path(raw_detected_path).expanduser().resolve())
            return {
                "workspace_path": final_ws,
                "messages": [AIMessage(content=f"🔍 Hệ thống đã xác minh thực tế và thiết lập workspace tại: `{final_ws}`")]
            }
            
        # Nếu LLM gọi các công cụ khảo sát hệ thống khác, chúng ta thực thi ngay lập tức
        tool_messages = []
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]
            
            if tool_name == "get_current_working_directory":
                result = get_current_working_directory.invoke(tool_args)
            elif tool_name == "check_path_exists":
                result = check_path_exists.invoke(tool_args)
            elif tool_name == "find_project_root":
                result = find_project_root.invoke(tool_args)
            else:
                result = f"Lỗi: Không tìm thấy công cụ khảo sát '{tool_name}'."
                
            tool_messages.append(ToolMessage(content=str(result), name=tool_name, tool_call_id=tool_id))
            
        local_history.extend(tool_messages)
        
    # Luồng dự phòng an toàn (Fallback) nếu vòng lặp bị lỗi hoặc LLM không chịu đưa ra quyết định
    fallback_ws = str(Path(".").resolve())
    return {
        "workspace_path": fallback_ws,
        "messages": [AIMessage(content=f"⚠️ Không thể dò tìm tự động bằng công cụ. Thiết lập workspace dự phòng tại: `{fallback_ws}`")]
    }


def git_setup_node(state: AgentState) -> Dict[str, Any]:
    ws = state["workspace_path"]
    
    # Kiểm tra xem có thư mục .git trong workspace hay không
    git_dir = Path(ws) / ".git"
    if not git_dir.exists():
        # Nếu chưa có Git, thiết lập nhánh là "no_git" và bỏ qua toàn bộ việc thiết lập
        return {
            "git_branch": "no_git",
            "messages": [AIMessage(content="ℹ️ Không phát hiện Git repository trong workspace này. Hệ thống sẽ bỏ qua các bước quản lý phiên bản (Git).")]
        }

    try:
        git_manager = GitManager(ws)
        branch = git_manager.init_and_prepare_branch()
        return {
            "git_branch": branch,
            "messages": [AIMessage(content=f"Đã cấu hình nhánh Git hoạt động: `{branch}`")]
        }
    except Exception as e:
        # Cơ chế fallback nếu có lỗi bất ngờ xảy ra khi tương tác với Git
        return {
            "git_branch": "no_git",
            "messages": [AIMessage(content=f"⚠️ Có lỗi xảy ra với Git ({str(e)}). Chuyển sang chế độ không dùng Git.")]
        }


def context_loader_node(state: AgentState) -> Dict[str, Any]:
    ws = state["workspace_path"]
    
    messages = state["messages"]
    user_msg = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) or getattr(msg, "type", None) == "human":
            user_msg = msg
            break
            
    if not user_msg:
        user_msg = messages[0]
        
    user_query = user_msg.content
    
    workspace_context = ""
    thongtin_path = Path(ws) / "THONGTIN.md"
    if thongtin_path.exists():
        try:
            workspace_context = thongtin_path.read_text(encoding="utf-8")
        except Exception as e:
            workspace_context = f"Lỗi khi đọc file THONGTIN.md: {str(e)}"
            
    structured_llm = model.with_structured_output(TaskTriage, method="function_calling")
    
    system_prompt = (
        "Bạn là một điều phối viên Agent thông minh. Hãy phân tích yêu cầu của người dùng "
        "để xác định xem đây là một yêu cầu đơn giản (chỉ cần chỉnh sửa trực tiếp 1-2 file, không cần lập kế hoạch chi tiết) "
        "hay một yêu cầu phức tạp (cần lên kế hoạch khảo sát, phát triển nhiều bước).\n"
        "Ví dụ về tác vụ ĐƠN GIẢN: Thêm một trường/thuộc tính mới vào model, sửa lỗi giao diện nhỏ, "
        "ẩn/hiện một phần tử dựa trên điều kiện đầu vào của người dùng."
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
        messages_to_append.append(AIMessage(content=f"📋 Đã tải ngữ cảnh `THONGTIN.md`. Kích hoạt chế độ **Fast-Track (Nhiệm vụ đơn giản)**. Bỏ qua bước lập kế hoạch chi tiết."))
    else:
        messages_to_append.append(AIMessage(content="📋 Đã tải ngữ cảnh `THONGTIN.md`. Nhận diện tác vụ phức tạp, chuyển tiếp tới Planner để thiết kế kế hoạch."))
        
    return {
        "workspace_context": workspace_context,
        "plan": plan_tasks,
        "task_type": task_type,
        "last_executed_task_ids": [],
        "messages": messages_to_append
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
        "  Ví dụ:\n"
        "  + Nhiệm vụ T1 (không có dep) và T5 (không có dep) có thể chạy song song.\n"
        "  + Nhiệm vụ T2 phụ thuộc T1 (dependencies=['T1']) -> phải chạy T1 trước.\n"
        "⚠️ LƯU Ý ĐẶC BIỆT QUAN TRỌNG ĐỂ TRÁNH TRÙNG LẶP (BẮT BUỘC):\n"
        "- Tuyệt đối KHÔNG đưa bước 'Tổng hợp và báo cáo' hoặc 'Viết tệp THONGTIN.md' làm nhiệm vụ cuối cùng trong kế hoạch!\n"
        "- Lý do: Hệ thống đã thiết kế một node chuyên biệt 'synthesis_node' độc lập để tự động làm việc này ở cuối đồ thị.\n"
        "Kế hoạch của bạn chỉ tập trung hoàn toàn vào các bước thực thi khảo sát vật lý (analysis) hoặc sửa đổi mã nguồn (development)."
    )
    
    plan_output = structured_llm.invoke([
        {"role": "system", "content": system_prompt},
        *conversation_history
    ])
    
    plan_tasks = getattr(plan_output, "tasks", [])
    task_type = getattr(plan_output, "task_type", "development")
    
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
    
    # 1. Tìm các task đủ điều kiện chạy song song/tuần tự dựa trên dependencies
    eligible_tasks = get_eligible_tasks(plan)
    if not eligible_tasks:
        pending_tasks = [
            t for t in plan 
            if (t.get("status") if isinstance(t, dict) else getattr(t, "status", None)) == "pending"
        ]
        if pending_tasks:
            # Cơ chế an toàn giải quyết deadlock nếu có cấu hình dependencies lỗi từ LLM
            eligible_tasks = [pending_tasks[0]]
        else:
            return {
                "messages": [AIMessage(content="Đã hoàn thành khảo sát toàn bộ các bước.")]
            }
         
    tasks_str = "\n".join([
        f"- [{getattr(t, 'id', None) or t.get('id')}] {getattr(t, 'description', None) or t.get('description')}"
        for t in eligible_tasks
    ])
    
    read_files = ReadFilesTool(workspace_path=ws)
    list_directory = ListDirectoryTool(workspace_path=ws)
    terminal = RunTerminalTool(workspace_path=ws)

    tools = [read_files, list_directory,terminal]
    model_with_tools = model.bind_tools(tools)
    
    previous_findings_str = ""
    existing_findings = state.get("step_findings", [])
    if existing_findings:
        previous_findings_str = "\n\n--- CÁC KẾT QUẢ KHẢO SÁT BẠN ĐÃ THU THẬP ĐƯỢC Ở CÁC BƯỚC TRƯỚC ---\n" + "\n\n".join(existing_findings)
    
    system_prompt = (
        "Bạn là một kiến trúc sư chuyên khảo sát, đọc hiểu và phân tích cấu trúc mã nguồn (Read-Only Mode).\n"
        f"Nhiệm vụ: Bạn đang thực hiện đồng thời các nhiệm vụ song song/tuần tự sau:\n{tasks_str}\n"
        f"Thư mục làm việc: {ws}\n"
    )
    if workspace_context:
        system_prompt += f"\n--- TỔNG QUAN VỀ DỰ ÁN (THONGTIN.md) ---\n{workspace_context}\n"
        
    system_prompt += (
        "\nHãy sử dụng các công cụ `read_files` và `list_directory` để đọc hiểu cấu trúc hệ thống.\n"
        "\n Nếu cần thêm thông tin mà 2 công cụ trên không hỗ trợ hãy dùng run_terminal_command \n"
        "YÊU CẦU QUAN TRỌNG:\n"
        "- Bạn chỉ có quyền ĐỌC dữ liệu, tuyệt đối không chỉnh sửa mã nguồn hoặc tự ý tạo tệp tin trong bước này.\n"
        "- Trình bày chi tiết, chuyên nghiệp về kết quả phát hiện được của bạn ở tin nhắn phản hồi cuối cùng.\n"
        "- Dựa trên ngữ cảnh khảo sát từ các bước trước (nếu có) để không thực hiện lại các thao tác tìm kiếm thừa."
    )
    if previous_findings_str:
        system_prompt += previous_findings_str
    
    response = model_with_tools.invoke([SystemMessage(content=system_prompt)] + messages)
    
    # CHUYỂN DỊCH THỜI ĐIỂM HOÀN THÀNH: Đánh dấu hoàn thành khi không còn yêu cầu gọi công cụ
    if not response.tool_calls:
        findings = []
        if response.content:
            tasks_ids = ", ".join([str(getattr(t, "id", None) or t.get("id")) for t in eligible_tasks])
            findings = [f"### Kết quả khảo sát các nhiệm vụ ({tasks_ids}):\n{response.content}"]
            
        # Cập nhật trạng thái completed cho các task vừa hoàn thành trong state
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
        return {
            "messages": [response]
        }


def development_executor_node(state: AgentState) -> Dict[str, Any]:
    ws = state["workspace_path"]
    plan = state["plan"]
    error_logs = state.get("error_logs", "")
    workspace_context = state.get("workspace_context", "")
    messages = list(state["messages"])
        
    # 1. Tìm các task đủ điều kiện chạy song song/tuần tự dựa trên dependencies
    eligible_tasks = get_eligible_tasks(plan)
    if not eligible_tasks:
        pending_tasks = [
            t for t in plan 
            if (t.get("status") if isinstance(t, dict) else getattr(t, "status", None)) == "pending"
        ]
        if pending_tasks:
            # Cơ chế an toàn tránh deadlock
            eligible_tasks = [pending_tasks[0]]
        else:
            return {
                "messages": [AIMessage(content="Đã hoàn thành toàn bộ các bước phát triển.")]
            }
         
    tasks_str = "\n".join([
        f"- [{getattr(t, 'id', None) or t.get('id')}] {getattr(t, 'description', None) or t.get('description')}"
        for t in eligible_tasks
    ])
    
    read_files = ReadFilesTool(workspace_path=ws)
    write_file = WriteFileTool(workspace_path=ws)
    apply_patch = ApplyPatchTool(workspace_path=ws)
    list_directory = ListDirectoryTool(workspace_path=ws)
    run_terminal_command = RunTerminalTool(workspace_path=ws)
    
    tools = [read_files, write_file, apply_patch, list_directory, run_terminal_command]
    model_with_tools = model.bind_tools(tools)
    
    system_prompt = (
        "Bạn là một kỹ sư phần mềm thực thi chuyên nghiệp chuyên sửa lỗi và viết mới mã nguồn (Write-Access Mode).\n"
        f"Nhiệm vụ: Bạn đang thực hiện đồng thời các nhiệm vụ song song/tuần tự sau:\n{tasks_str}\n"
        f"Thư mục làm việc: {ws}\n"
    )
    if workspace_context:
        system_prompt += f"\n--- NGỮ CẢNH HỆ THỐNG HIỆN TẠI (THONGTIN.md) ---\n{workspace_context}\n"
        
    system_prompt += (
        "\nHãy sử dụng các công cụ `read_files`, `write_file`, `apply_search_replace_patch`, `list_directory` và `run_terminal_command` để giải quyết nhiệm vụ.\n"
        "\n⚠️ QUY TẮC QUAN TRỌNG VỀ SỬA ĐỔI FILE (BẮT BUỘC TUÂN THỦ):\n"
        "1. Đối với các file có sẵn trong hệ thống có độ dài TRÊN 300 DÒNG (hoặc tệp tin có kích thước lớn):\n"
        "   - Bạn TUYỆT ĐỐI KHÔNG ĐƯỢC sử dụng công cụ `write_file` để ghi đè lại toàn bộ tệp.\n"
        "   - Thay vào đó, bạn BẮT BUỘC PHẢI sử dụng công cụ `apply_search_replace_patch` để áp dụng bản vá cục bộ (Search-and-Replace).\n"
        "   - Điều này giúp tránh lỗi ngắt quãng đầu ra (output truncation) và giữ tính toàn vẹn của mã nguồn hiện hành.\n"
        "2. Công cụ `write_file` CHỈ được dùng để:\n"
        "   - Tạo mới các tệp tin chưa từng tồn tại trong workspace.\n"
        "   - Sửa đổi các tệp cực kỳ ngắn (dưới 300 dòng).\n"
        "3. Khi sử dụng `apply_search_replace_patch`:\n"
        "   - Hãy chắc chắn rằng khối văn bản SEARCH trùng khớp 100% từng ký tự, dấu cách và thụt lề dòng (indentation) với tệp gốc.\n"
        "\nYÊU CẦU PHÁT TRIỂN KHÁC:\n"
        "- Khi chỉnh sửa tệp đã có, luôn luôn dùng `read_files` đọc nội dung trước để hiểu ngữ cảnh và tránh làm mất mã nguồn cũ.\n"
        "- Bạn có thể cài đặt thư viện, build dự án, hoặc chạy kiểm thử bằng công cụ `run_terminal_command`.\n"
        "- Mô tả thật chi tiết các hành động, giải pháp và vị trí các dòng code bạn đã cập nhật ở tin nhắn phản hồi cuối cùng."
    )
    
    input_messages = [SystemMessage(content=system_prompt)]
    if error_logs:
        input_messages.append(HumanMessage(content=f"LƯU Ý SỬA LỖI TỪ LƯỢT CHẠY TRƯỚC (HỆ THỐNG KIỂM TRA BÁO LỖI):\n{error_logs}\nHãy tập trung khắc phục lỗi này triệt để."))
        
    response = model_with_tools.invoke(input_messages + messages)
    
    # CHUYỂN DỊCH THỜI ĐIỂM HOÀN THÀNH: Đóng trạng thái khi không gọi tool nữa
    if not response.tool_calls:
        # Đánh dấu hoàn thành các task vừa thực thi
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
            "last_executed_task_ids": list(eligible_ids), # Lưu lại danh sách ID để tester rollback chính xác nếu lỗi
            "attempts": 0,                        
            "error_logs": ""                      
        }
    else:
        return {
            "messages": [response]
        }


def tool_node(state: AgentState) -> Dict[str, Any]:
    ws = state["workspace_path"]
    read_files = ReadFilesTool(workspace_path=ws)
    write_file = WriteFileTool(workspace_path=ws)
    apply_patch = ApplyPatchTool(workspace_path=ws)
    list_directory = ListDirectoryTool(workspace_path=ws)
    run_terminal_command = RunTerminalTool(workspace_path=ws)
    
    tools_map = {
        "read_files": read_files,
        "write_file": write_file,
        "apply_search_replace_patch": apply_patch,
        "list_directory": list_directory,
        "run_terminal_command": run_terminal_command
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
    last_executed_ids = state.get("last_executed_task_ids", [])
    errors = []
    
    for file_path in modified_files:
        if file_path.endswith(".py"):
            res = subprocess.run(["python", "-m", "py_compile", file_path], capture_output=True, text=True)
            if res.returncode != 0:
                errors.append(f"Lỗi cú pháp tại {Path(file_path).name}:\n{res.stderr.strip()}")
                
    if errors:
        combined_error = "\n".join(errors)
        if attempts < 2:
            # Sửa đổi quan trọng: Thay vì giảm index, ta rollback trạng thái các task vừa chạy về 'pending'
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
                "messages": [AIMessage(content=f"⚠️ Phát hiện lỗi kiểm tra tại các nhiệm vụ vừa thực thi ({', '.join(last_executed_ids)}):\n{combined_error}\nHệ thống đưa các bước này về 'pending' để chuẩn bị quay lại sửa đổi.")]
            }
        else:
            return {
                "error_logs": "",
                "attempts": 0,
                "messages": [AIMessage(content="❌ Đã vượt quá 3 lần tự động sửa lỗi. Bỏ qua lỗi này để tiếp tục.")]
            }
                
    return {
        "error_logs": "",
        "attempts": 0,
        "messages": [AIMessage(content="✅ Toàn bộ các bước phát triển vừa qua đã vượt qua vòng kiểm tra thành công.")]
    }


def synthesis_node(state: AgentState) -> Dict[str, Any]:
    ws = state["workspace_path"]
    findings = state.get("step_findings", [])
    
    if not findings:
        return {"messages": [AIMessage(content="Không thu thập được thông tin khảo sát để tổng hợp.")]}
        
    compiled_data = "\n\n---\n\n".join(findings)
    
    synthesis_prompt = (
        "Bạn là một Kiến trúc sư Hệ thống cao cấp chuyên biên soạn tài liệu kỹ thuật.\n"
        "Nhiệm vụ của bạn là đọc toàn bộ lịch sử khảo sát TUẦN TỰ bên dưới và tổng hợp thành một tài liệu 'THONGTIN.md' duy nhất.\n\n"
        "LƯU Ý QUAN TRỌNG: Do quá trình khảo sát diễn ra tuần tự, các bước phân tích sau có khả năng mở rộng, sửa chữa "
        "hoặc làm rõ thêm các nhận định của các bước phân tích trước đó. Hãy liên kết, đối chiếu logic một cách thống nhất, "
        "tránh lặp thông tin hoặc tạo ra mâu thuẫn trong báo cáo cuối cùng.\n\n"
        "⚠️ YÊU CẦU CỰC KỲ QUAN TRỌNG ĐỂ TRÁNH LỖI TRÀN TOKEN ĐẦU RA (OUTPUT TRUNCATION):\n"
        "- Hãy viết tài liệu thật SÚC TÍCH, CÔ ĐỌNG, tập trung vào cấu trúc, kiến trúc hệ thống và dependencies.\n"
        "- TUYỆT ĐỐI KHÔNG sao chép hay chèn các đoạn mã nguồn dài dòng.\n"
        "- KHÔNG chia nhỏ tài liệu thành các phần rác như 'Phần 1/3'. Tài liệu phải hoàn chỉnh trong lượt sinh này.\n"
        "- Sử dụng bảng biểu súc tích thay vì viết các đoạn mô tả dài dòng.\n\n"
        "Tài liệu phải bao gồm các mục tiêu chuẩn:\n"
        "1. Tổng quan & Môi trường phát triển.\n"
        "2. Cấu trúc thư mục chính.\n"
        "3. Danh sách Dependencies chính và vai trò.\n"
        "4. Mô tả các Module chính, Scrapers & Services.\n"
        "5. Đánh giá nhanh (Điểm mạnh, điểm yếu, đề xuất nâng cấp)."
    )
    
    try:
        response = model.invoke([
            SystemMessage(content=synthesis_prompt),
            HumanMessage(content=f"Dưới đây là toàn bộ thông tin thu thập tuần tự từ hệ thống:\n\n{compiled_data}\n\nHãy tạo tệp 'THONGTIN.md' hoàn chỉnh.")
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
            
            # CHỈ CHẠY LỆNH GIT KHI WORKSPACE CÓ SỬ DỤNG GIT
            git_branch = state.get("git_branch", "no_git")
            if git_branch != "no_git":
                git_manager = GitManager(ws)
                git_manager._run_cmd(["git", "add", "THONGTIN.md"], ignore_error=True)
            
            return {
                "workspace_context": cleaned_md,
                "messages": [AIMessage(content="**Tổng hợp tài liệu hoàn tất:** Hệ thống đã biên dịch thành tệp `THONGTIN.md` thành công tại thư mục gốc của dự án.")]
            }
    except Exception as e:
        return {
            "messages": [AIMessage(content=f"Cảnh báo: Có lỗi xảy ra trong quá trình tổng hợp tệp THONGTIN.md: {str(e)}")]
        }
    return {}


def commit_node(state: AgentState) -> Dict[str, Any]:
    ws = state["workspace_path"]
    task_type = state.get("task_type", "development")
    git_branch = state.get("git_branch", "no_git")
    
    # Nếu đang ở chế độ không dùng Git, bỏ qua và báo cáo kết quả
    if git_branch == "no_git":
        return {
            "messages": [AIMessage(content="Đã hoàn thành toàn bộ yêu cầu của bạn. Chế độ Không-Git được kích hoạt, các thay đổi vật lý đã được lưu trực tiếp xuống đĩa.")]
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
        
    return {
        "messages": [AIMessage(content=msg)]
    }