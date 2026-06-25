# main.py
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from state import AgentState
import nodes
import routers

# ==========================================
# KHỞI TẠO ĐỒ THỊ LANGGRAPH
# ==========================================
builder = StateGraph(AgentState)

# 1. Đăng ký các Nodes từ file nodes.py
builder.add_node("detect_workspace", nodes.detect_workspace_node)
builder.add_node("git_setup", nodes.git_setup_node)
builder.add_node("planner", nodes.planner_node)
builder.add_node("analysis_executor", nodes.analysis_executor_node)       
builder.add_node("development_executor", nodes.development_executor_node) 
builder.add_node("tester", nodes.tester_node)
builder.add_node("synthesis", nodes.synthesis_node)
builder.add_node("commit", nodes.commit_node)

# 2. Định nghĩa các Cạnh nối (Edges) & Định tuyến (Routers) từ routers.py
builder.add_conditional_edges(
    START, 
    routers.start_router, 
    {"detect_workspace": "detect_workspace", "planner": "planner"}
)
builder.add_edge("detect_workspace", "git_setup")
builder.add_edge("git_setup", "planner")

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
        "analysis_executor": "analysis_executor", 
        "synthesis": "synthesis"                  
    }
)
builder.add_edge("synthesis", "commit")

builder.add_conditional_edges(
    "development_executor",
    routers.development_router,
    {
        "development_executor": "development_executor", 
        "tester": "tester"                              
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

# 3. Kích hoạt lưu trữ phiên làm việc qua MemorySaver
memory = MemorySaver()
app = builder.compile(checkpointer=memory)

# ==========================================
# KHU VỰC THỬ NGHIỆM ĐỒ THỊ (NẾU CẦN CHẠY)
# ==========================================
if __name__ == "__main__":
    print("Khởi tạo và biên dịch đồ thị LangGraph thành công.")
    # Bạn có thể thực hiện chạy thử app.invoke(...) tại đây