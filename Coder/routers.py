# oder/routers.py
from typing import Literal
from langchain_core.messages import AIMessage
from state import AgentState, Task

def context_loader_router(state: AgentState) -> Literal["planner", "executor"]:
    # Nếu là tác vụ đơn giản hoặc đã có kế hoạch từ trước, chuyển sang Executor
    if state.get("is_simple") or state.get("plan"):
        return "executor"
    return "planner"


def planner_router(state: AgentState) -> Literal["executor"]:
    return "executor"


def executor_router(state: AgentState) -> Literal["tool_node", "tester", "replanner", "synthesis"]:
    messages = state["messages"]
    if not messages:
        if state.get("is_simple"):
            return "tester" if state.get("task_type") == "development" else "synthesis"
        return "replanner"
          
    last_message = messages[-1]
    # Trường hợp 1: Agent đang gọi công cụ (đọc/ghi file, tương tác web...)
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_node"
        
    # Trường hợp 2: Agent báo cáo đã xong lượt chạy hiện tại
    # CHỈ chạy kiểm thử nếu đây là task phát triển VÀ có tệp tin thực sự bị sửa đổi
    if state.get("task_type") == "development" and state.get("modified_files"):
        return "tester"
        
    # Trường hợp 3: Tác vụ đơn giản không sửa code (ví dụ: tác vụ đọc hiểu đơn thuần)
    if state.get("is_simple"):
        return "synthesis"
        
    # Trường hợp 4: Tác vụ phân tích phức tạp hoặc task không thay đổi code
    return "replanner"


def tester_router(state: AgentState) -> Literal["executor", "replanner", "commit"]:
    error = state.get("error_logs", "")
    attempts = state.get("attempts", 0)
    is_simple = state.get("is_simple", False)
    
    # Lỗi phát sinh và vẫn còn lượt tự sửa đổi -> Quay lại Executor
    if error and attempts < 3:
        return "executor"
        
    # Nếu là tác vụ đơn giản (Fast-Track), bypass hoàn toàn bước lập kế hoạch (Replanner)
    # Chuyển thẳng tới commit/END để kết thúc phiên chạy một cách nhanh nhất
    if is_simple:
        return "commit"
        
    # Tác vụ phức tạp, chuyển sang Replanner để phân tích tổng thể kế hoạch DAG
    return "replanner"


def replanner_router(state: AgentState) -> Literal["executor", "synthesis"]:
    plan = state["plan"]
    
    # Kiểm tra xem còn nhiệm vụ nào ở trạng thái chờ thực thi không
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