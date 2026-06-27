# routers.py
from typing import Literal
from langchain_core.messages import AIMessage
from state import AgentState, WorkspaceDiscoveryState

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

# SỬA ĐỔI: Sử dụng start_router chuẩn để bypass qua Subgraph nếu đã có workspace_path
def start_router(state: AgentState) -> Literal["detect_workspace", "context_loader"]:
    if not state.get("workspace_path"):
        return "detect_workspace"
    return "context_loader"


# THÊM MỚI: Định tuyến sau git_setup giúp bỏ qua Planner nếu đã có sẵn kế hoạch từ bước Triage (Fast-Track)
def git_setup_router(state: AgentState) -> Literal["planner", "analysis_executor", "development_executor"]:
    # Nếu đã có sẵn kế hoạch (ví dụ từ Triage chế độ đơn giản)
    if state.get("plan"):
        task_type = state.get("task_type", "development")
        if task_type == "analysis":
            return "analysis_executor"
        return "development_executor"
    return "planner"


def planner_router(state: AgentState) -> Literal["analysis_executor", "development_executor"]:
    task_type = state.get("task_type", "development")
    if task_type == "analysis":
        print("--> Kích hoạt khảo sát đồ thị nhiệm vụ (DAG).")
        return "analysis_executor"
        
    print("--> Kích hoạt phát triển đồ thị nhiệm vụ (DAG).")
    return "development_executor"


def analysis_router(state: AgentState) -> Literal["tool_node", "analysis_executor", "synthesis"]:
    messages = state["messages"]
    if not messages:
        return "synthesis"
          
    last_message = messages[-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_node"
        
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
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_node"
        
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