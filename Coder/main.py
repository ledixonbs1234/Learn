# main.py
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from state import AgentState, WorkspaceDiscoveryState
import nodes
import routers

# ==========================================
# 1. KHỞI TẠO VÀ BIÊN DỊCH ĐỒ THỊ CON (SUBGRAPH)
# ==========================================
sub_builder = StateGraph(WorkspaceDiscoveryState)

sub_builder.add_node("discovery_agent_node", nodes.discovery_agent_node)
sub_builder.add_node("discovery_tool_node", nodes.discovery_tool_node)
sub_builder.add_node("discovery_finalize_node", nodes.discovery_finalize_node)

sub_builder.add_edge(START, "discovery_agent_node")
sub_builder.add_conditional_edges(
    "discovery_agent_node",
    routers.discovery_router,
    {
        "discovery_tool_node": "discovery_tool_node",
        "discovery_finalize_node": "discovery_finalize_node"
    }
)
sub_builder.add_edge("discovery_tool_node", "discovery_agent_node")
sub_builder.add_edge("discovery_finalize_node", END)

workspace_discovery_subgraph = sub_builder.compile()
nodes.workspace_discovery_subgraph = workspace_discovery_subgraph


# ==========================================
# 2. KHỞI TẠO ĐỒ THỊ CHÍNH (PARENT GRAPH)
# ==========================================
builder = StateGraph(AgentState)

# Đăng ký các Nodes (Hợp nhất các executor thành một node duy nhất)
builder.add_node("detect_and_triage", nodes.detect_and_triage_node) 
builder.add_node("context_loader", nodes.context_loader_node)
builder.add_node("git_setup", nodes.git_setup_node)
builder.add_node("planner", nodes.planner_node)
builder.add_node("executor", nodes.executor_node)  # <--- HỢP NHẤT LÀM 1 NODE DUY NHẤT
builder.add_node("tool_node", nodes.tool_node) 
builder.add_node("replanner", nodes.replanner_node)
builder.add_node("tester", nodes.tester_node)
builder.add_node("synthesis", nodes.synthesis_node)
builder.add_node("commit", nodes.commit_node)

# Định nghĩa các Cạnh nối (Edges) và Định tuyến (Routers)

builder.add_edge(START, "detect_and_triage")
builder.add_edge("detect_and_triage", "git_setup")
builder.add_edge("git_setup", "context_loader")

builder.add_conditional_edges(
    "context_loader",
    routers.context_loader_router,
    {
        "planner": "planner",
        "executor": "executor"
    }
)

builder.add_conditional_edges(
    "planner",
    routers.planner_router,
    {
        "executor": "executor"
    }
)

# Định tuyến từ Executor tới Tool hoặc các nút Đánh giá kế hoạch
builder.add_conditional_edges(
    "executor",
    routers.executor_router,
    {
        "tool_node": "tool_node",                  
        "replanner": "replanner",
        "tester": "tester",
        "synthesis": "synthesis"
    }
)

# Chuyển hướng từ Tool Node quay lại Executor
builder.add_conditional_edges(
    "tool_node",
    routers.tool_router,
    {
        "executor": "executor"
    }
)

# Định tuyến sau replanner đi tiếp dựa vào cấu trúc DAG cập nhật
builder.add_conditional_edges(
    "replanner",
    routers.replanner_router,
    {
        "executor": "executor",
        "synthesis": "synthesis",
        "tester": "tester"
    }
)

# Tester thất bại sẽ đẩy ngược về replanner để sửa kế hoạch hoặc đẩy về executor nếu là Fast-Track
builder.add_conditional_edges(
    "tester",
    routers.tester_router,
    {
        "replanner": "replanner", 
        "executor": "executor",
        "commit": "commit"                              
    }
)

builder.add_edge("synthesis", "commit")
builder.add_edge("commit", END)

memory = MemorySaver()
app = builder.compile(checkpointer=memory)

if __name__ == "__main__":
    print("Biên dịch đồ thị LangGraph tối giản trực quan thành công.")