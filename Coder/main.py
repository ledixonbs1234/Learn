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

# Biên dịch Subgraph thành một thực thể thực thi độc lập
workspace_discovery_subgraph = sub_builder.compile()
# Xuất subgraph để nodes.py có thể gọi thủ công (Isolate State)
nodes.workspace_discovery_subgraph = workspace_discovery_subgraph


# ==========================================
# 2. KHỞI TẠO ĐỒ THỊ CHÍNH (PARENT GRAPH)
# ==========================================
builder = StateGraph(AgentState)

# Đăng ký các Nodes (Đã chuyển detect_workspace thành node wrapper cô lập trạng thái)
builder.add_node("detect_workspace", nodes.detect_workspace_wrapper_node) 
builder.add_node("context_loader", nodes.context_loader_node)
builder.add_node("triage", nodes.triage_node)                       
builder.add_node("git_setup", nodes.git_setup_node)
builder.add_node("planner", nodes.planner_node)
builder.add_node("analysis_executor", nodes.analysis_executor_node)       
builder.add_node("development_executor", nodes.development_executor_node) 
builder.add_node("tool_node", nodes.tool_node) 
builder.add_node("tester", nodes.tester_node)
builder.add_node("synthesis", nodes.synthesis_node)
builder.add_node("commit", nodes.commit_node)

# Định nghĩa các Cạnh nối (Edges) và Đinh tuyến (Routers)

# SỬA ĐỔI: Kích hoạt định tuyến thông minh ngay từ điểm START
builder.add_conditional_edges(
    START,
    routers.start_router,
    {
        "detect_workspace": "detect_workspace",
        "context_loader": "context_loader"
    }
)

builder.add_edge("detect_workspace", "context_loader")
builder.add_edge("context_loader", "triage")
builder.add_edge("triage", "git_setup")

# SỬA ĐỔI: Định tuyến sau git_setup để bỏ qua Planner nếu đang ở chế độ Fast-Track
builder.add_conditional_edges(
    "git_setup",
    routers.git_setup_router,
    {
        "planner": "planner",
        "analysis_executor": "analysis_executor",
        "development_executor": "development_executor"
    }
)

builder.add_conditional_edges(
    "planner",
    routers.planner_router,
    {
        "analysis_executor": "analysis_executor",      
        "development_executor": "development_executor" 
    }
)

builder.add_conditional_edges(
    "analysis_executor",
    routers.analysis_router,
    {
        "tool_node": "tool_node",                  
        "analysis_executor": "analysis_executor", 
        "synthesis": "synthesis"                  
    }
)
builder.add_edge("synthesis", "commit")

builder.add_conditional_edges(
    "development_executor",
    routers.development_router,
    {
        "tool_node": "tool_node",                           
        "development_executor": "development_executor", 
        "tester": "tester"                              
    }
)

# Chuyển hướng từ Tool Node quay lại Executor tương ứng dựa vào task_type
builder.add_conditional_edges(
    "tool_node",
    routers.tool_router,
    {
        "analysis_executor": "analysis_executor",
        "development_executor": "development_executor"
    }
)

builder.add_conditional_edges(
    "tester",
    routers.tester_router,
    {
        "development_executor": "development_executor", 
        "commit": "commit"                              
    }
)

builder.add_edge("commit", END)

# Kích hoạt lưu trữ phiên làm việc qua MemorySaver
memory = MemorySaver()
app = builder.compile(checkpointer=memory)

if __name__ == "__main__":
    print("Khởi tạo và biên dịch đồ thị phân tầng LangGraph tối ưu thành công.")