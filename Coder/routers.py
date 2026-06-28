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
def start_router(state: AgentState) -> Literal["detect_workspace", "context_loader"]:
    if not state.get("workspace_path"):
        return "detect_workspace"
    return "context_loader"


def git_setup_router(state: AgentState) -> Literal["planner", "analysis_executor", "development_executor"]:
    if state.get("plan"):
        task_type = state.get("task_type", "development")
        if task_type == "analysis":
            return "analysis_executor"
        return "development_executor"
    return "planner"


def planner_router(state: AgentState) -> Literal["analysis_executor", "development_executor"]:
    task_type = state.get("task_type", "development")
    if task_type == "analysis":
        return "analysis_executor"
    return "development_executor"


# 🔄 ĐIỀU CHỈNH: analysis_router chuyển tiếp tới replanner thay vì synthesis trực tiếp
def analysis_router(state: AgentState) -> Literal["tool_node", "replanner"]:
    messages = state["messages"]
    if not messages:
        return "replanner"
          
    last_message = messages[-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_node"
        
    return "replanner"


# 🔄 ĐIỀU CHỈNH: development_router chuyển tiếp tới replanner thay vì tester trực tiếp
def development_router(state: AgentState) -> Literal["tool_node", "replanner"]:
    messages = state["messages"]
    if not messages:
        return "replanner"
        
    last_message = messages[-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_node"
        
    return "replanner"


# 🔄 THÊM MỚI: Định tuyến thông minh sau khi lập kế hoạch thích ứng
def replanner_router(state: AgentState) -> Literal["analysis_executor", "development_executor", "synthesis", "tester"]:
    plan = state["plan"]
    task_type = state.get("task_type", "development")
    
    # Lọc tìm các nhiệm vụ chưa hoàn thành (pending)
    pending_tasks = []
    for t in plan:
        status = t.status if isinstance(t, Task) else t.get("status")
        if status == "pending":
            pending_tasks.append(t)
            
    if pending_tasks:
        # Nếu vẫn còn tác vụ chưa làm, chuyển tiếp về executor tương ứng để làm tiếp
        if task_type == "analysis":
            return "analysis_executor"
        return "development_executor"
    else:
        # Nếu toàn bộ kế hoạch đã hoàn tất thành công
        if task_type == "analysis":
            return "synthesis"
        return "tester"


# 🔄 ĐIỀU CHỈNH: tester_router gửi lỗi về replanner để lập kế hoạch sửa đổi (Self-Healing)
def tester_router(state: AgentState) -> Literal["replanner", "commit"]:
    error = state.get("error_logs", "")
    attempts = state.get("attempts", 0)
    
    if error and attempts < 3:
        # Thay vì gửi thẳng về development_executor, chúng ta cho qua replanner để phân tích lỗi lầm
        return "replanner"
    return "commit"


def tool_router(state: AgentState) -> Literal["analysis_executor", "development_executor"]:
    task_type = state.get("task_type", "development")
    if task_type == "analysis":
        return "analysis_executor"
    return "development_executor"