# oder/routers.py
from typing import Literal
from langchain_core.messages import AIMessage
from state import AgentState, Task

def executor_router(state: AgentState) -> Literal["tool_node", "tester", "replanner", "synthesis"]:
    messages = state["messages"]
    if not messages:
        if state.get("is_simple"):
            return "tester" if state.get("task_type") == "development" else "synthesis"
        return "replanner"
          
    last_message = messages[-1]
    # Trường hợp 1: Agent đang gọi công cụ
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_node"
        
    # Trường hợp 2: Agent báo cáo đã xong lượt chạy hiện tại
    if state.get("task_type") == "development" and state.get("modified_files"):
        return "tester"
        
    # Trường hợp 3: Tác vụ đơn giản không sửa code
    if state.get("is_simple"):
        return "synthesis"
        
    # Trường hợp 4: Tác vụ phân tích phức tạp hoặc task không thay đổi code (VD: T_SURVEY đã hoàn tất)
    return "replanner"


def tester_router(state: AgentState) -> Literal["executor", "chrome_extension_debugger", "replanner", "commit"]:
    error = state.get("error_logs", "")
    attempts = state.get("attempts", 0)
    is_simple = state.get("is_simple", False)
    extension_path = state.get("extension_path", "")
    
    # 1. Nếu có lỗi kiểm tra tĩnh (cú pháp/biên dịch) và chưa quá 3 lần thử -> Quay lại sửa code [2]
    if error and attempts < 3:
        return "executor"
        
    # 2. Nếu kiểm tra tĩnh ĐÃ THÀNH CÔNG và đây là dự án Chrome Extension: Chuyển sang kiểm thử động
    if not error and extension_path:
        return "chrome_extension_debugger"
        
    # 3. Các trường hợp thông thường khác
    if is_simple:
        return "commit"
        
    return "replanner"

def debugger_router(state: AgentState) -> Literal["executor", "replanner", "synthesis"]:
    runtime_error = state.get("error_logs", "")
    is_simple = state.get("is_simple", False)
    
    # Nếu phát hiện lỗi crash runtime của Extension -> Đưa thông tin lỗi quay lại để sửa đổi
    if runtime_error:
        return "executor" if is_simple else "replanner"
        
    # Nếu chạy mượt mà không có lỗi: Tiếp tục kế hoạch
    if is_simple:
        return "synthesis"
        
    return "replanner"
def replanner_router(state: AgentState) -> Literal["executor", "synthesis"]:
    plan = state["plan"]
    
    pending_tasks = []
    for t in plan:
        status = t.status if isinstance(t, Task) else t.get("status")
        if status == "pending":
            pending_tasks.append(t)
            
    if pending_tasks:
        return "executor"
    else:
        return "synthesis"


def tool_router(state: AgentState) -> Literal["executor"]:
    return "executor"