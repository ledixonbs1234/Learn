# routers.py
from typing import Literal
from langchain_core.messages import AIMessage
from state import AgentState, WorkspaceDiscoveryState, Task

# ==========================================
# ĐỊNH TUYẾN CHO ĐỒ THỊ CON DÒ TÌM WORKSPACE
# ==========================================
def discovery_router(state: WorkspaceDiscoveryState) -> Literal["discovery_tool_node", "discovery_finalize_node"]:
    messages = state["messages"]
    if not messages:
        return "discovery_finalize_node"
        
    last_message = messages[-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            if tool_call["name"] == "WorkspaceDetection":
                return "discovery_finalize_node"
        return "discovery_tool_node"
        
    return "discovery_finalize_node"


# ==========================================
# ĐỊNH TUYẾN CHO ĐỒ THỊ CHÍNH (MAIN GRAPH)
# ==========================================


def context_loader_router(state: AgentState) -> Literal["planner", "executor"]:
    if state.get("plan"):
        return "executor"
    return "planner"


def planner_router(state: AgentState) -> Literal["executor"]:
    return "executor"


# HỢP NHẤT: Định tuyến cho Executor duy nhất
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


# HỢP NHẤT: Định tuyến quay lại sau khi cập nhật kế hoạch thích ứng
def replanner_router(state: AgentState) -> Literal["executor", "synthesis", "tester"]:
    plan = state["plan"]
    task_type = state.get("task_type", "development")
    
    # Lọc tìm các nhiệm vụ chưa hoàn thành (pending)
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