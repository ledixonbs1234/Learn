# routers.py
from typing import Literal
from langchain_core.messages import AIMessage
from state import AgentState, Task

# ==========================================
# ĐỊNH TUYẾN CHO ĐỒ THỊ CHÍNH (MAIN GRAPH)
# ==========================================

def context_loader_router(state: AgentState) -> Literal["planner", "executor"]:
    # Nếu là tác vụ đơn giản (Fast-Track) HOẶC đã có sẵn kế hoạch từ trước, chuyển thẳng sang Executor
    if state.get("is_simple") or state.get("plan"):
        return "executor"
    return "planner"


def planner_router(state: AgentState) -> Literal["executor"]:
    return "executor"


def executor_router(state: AgentState) -> Literal["tool_node", "replanner", "tester", "synthesis"]:
    messages = state["messages"]
    if not messages:
        if state.get("is_simple"):
            return "tester" if state.get("task_type") == "development" else "synthesis"
        return "replanner"
          
    last_message = messages[-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_node"
        
    if state.get("is_simple"):
        return "tester" if state.get("task_type") == "development" else "synthesis"
    return "replanner"


def replanner_router(state: AgentState) -> Literal["executor", "synthesis", "tester"]:
    plan = state["plan"]
    task_type = state.get("task_type", "development")
    
    pending_tasks = []
    for t in plan:
        status = t.status if isinstance(t, Task) else t.get("status")
        if status == "pending":
            pending_tasks.append(t)
            
    if pending_tasks:
        return "executor"
    else:
        if task_type == "analysis":
            return "synthesis"
        return "tester"


def tester_router(state: AgentState) -> Literal["replanner", "executor", "commit"]:
    error = state.get("error_logs", "")
    attempts = state.get("attempts", 0)
    
    if error and attempts < 3:
        if state.get("is_simple"):
            return "executor"
        return "replanner"
    return "commit"


def tool_router(state: AgentState) -> Literal["executor"]:
    return "executor"