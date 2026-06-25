# routers.py
from typing import Literal
from langchain_core.messages import AIMessage
from state import AgentState

def start_router(state: AgentState) -> Literal["detect_workspace", "planner"]:
    if not state.get("workspace_path"):
        return "detect_workspace"
    return "planner"


def planner_router(state: AgentState) -> Literal["analysis_executor", "development_executor"]:
    task_type = state.get("task_type", "development")
    if task_type == "analysis":
        print("--> Kích hoạt khảo sát TUẦN TỰ để tối ưu hóa ngữ cảnh và tính liên kết giữa các bước.")
        return "analysis_executor"
        
    print("--> Kích hoạt phát triển TUẦN TỰ để tránh xung đột ghi đè.")
    return "development_executor"


def analysis_router(state: AgentState) -> Literal["tool_node", "analysis_executor", "synthesis"]:
    messages = state["messages"]
    if not messages:
        return "synthesis"
        
    last_message = messages[-1]
    
    # 1. Nếu tin nhắn AI yêu cầu gọi Tool -> Route đến tool_node để thực thi
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_node"
        
    # 2. Nếu AI trả lời bình thường (không gọi Tool) -> Hoàn thành bước hiện tại
    current_idx = state["current_step_idx"]
    plan = state["plan"]
    
    # So sánh current_idx + 1 với tổng số bước trong kế hoạch
    if current_idx + 1 < len(plan):
        return "analysis_executor"
    return "synthesis"


def development_router(state: AgentState) -> Literal["tool_node", "development_executor", "tester"]:
    messages = state["messages"]
    if not messages:
        return "tester"
        
    last_message = messages[-1]
    
    # 1. Nếu tin nhắn AI yêu cầu gọi Tool -> Route đến tool_node để thực thi
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_node"
        
    # 2. Nếu AI trả lời bình thường (không gọi Tool) -> Hoàn thành bước hiện tại
    current_idx = state["current_step_idx"]
    plan = state["plan"]
    
    if current_idx + 1 < len(plan):
        return "development_executor"
    return "tester"


def tester_router(state: AgentState) -> Literal["development_executor", "commit"]:
    error = state.get("error_logs", "")
    attempts = state.get("attempts", 0)
    if error and attempts < 3:
        return "development_executor"
    return "commit"


def tool_router(state: AgentState) -> Literal["analysis_executor", "development_executor"]:
    """
    ROUTER ĐIỀU HƯỚNG: Đưa Agent trở lại Executor tương ứng tùy theo tác vụ sau khi Tool đã chạy xong.
    """
    task_type = state.get("task_type", "development")
    if task_type == "analysis":
        return "analysis_executor"
    return "development_executor"