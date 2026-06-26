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
        print("--> Kích hoạt khảo sát đồ thị nhiệm vụ (DAG) để tối ưu hóa hiệu năng.")
        return "analysis_executor"
        
    print("--> Kích hoạt phát triển đồ thị nhiệm vụ (DAG) để sửa đổi tối ưu.")
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
    # Kiểm duyệt xem còn bất kỳ task nào chưa hoàn thành hay không
    plan = state["plan"]
    pending_tasks = [t for t in plan if (getattr(t, "status", None) or t.get("status")) == "pending"]
    
    if pending_tasks:
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
    plan = state["plan"]
    pending_tasks = [t for t in plan if (getattr(t, "status", None) or t.get("status")) == "pending"]
    
    if pending_tasks:
        return "development_executor"
    return "tester"


def tester_router(state: AgentState) -> Literal["development_executor", "commit"]:
    error = state.get("error_logs", "")
    attempts = state.get("attempts", 0)
    if error and attempts < 3:
        return "development_executor"
    return "commit"


def tool_router(state: AgentState) -> Literal["analysis_executor", "development_executor"]:
    task_type = state.get("task_type", "development")
    if task_type == "analysis":
        return "analysis_executor"
    return "development_executor"