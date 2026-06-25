# routers.py
from typing import Literal
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


def analysis_router(state: AgentState) -> Literal["analysis_executor", "synthesis"]:
    current_idx = state["current_step_idx"]
    plan = state["plan"]
    if current_idx < len(plan):
        return "analysis_executor"
    return "synthesis"


def development_router(state: AgentState) -> Literal["development_executor", "tester"]:
    current_idx = state["current_step_idx"]
    plan = state["plan"]
    if current_idx < len(plan):
        return "development_executor"
    return "tester"


def tester_router(state: AgentState) -> Literal["development_executor", "commit"]:
    error = state.get("error_logs", "")
    attempts = state.get("attempts", 0)
    if error and attempts < 3:
        return "development_executor"
    return "commit"