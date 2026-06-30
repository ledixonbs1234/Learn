# oder/main.py
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from state import AgentState
import nodes
import routers

builder = StateGraph(AgentState)

# Đăng ký các Nodes (ĐÃ LOẠI BỎ PLANNER NODE)
builder.add_node("detect_and_triage", nodes.detect_and_triage_node) 
builder.add_node("context_loader", nodes.context_loader_node)
builder.add_node("executor", nodes.executor_node)  
builder.add_node("tool_node", nodes.tool_node) 

builder.add_node("replanner", nodes.replanner_node)
builder.add_node("replanner_interrupt", nodes.replanner_interrupt_node) 

builder.add_node("tester", nodes.tester_node)
builder.add_node("synthesis", nodes.synthesis_node)
builder.add_node("commit", nodes.commit_node)

# Thiết lập các cạnh nối chính (ĐÃ ĐƠN GIẢN HÓA TUYẾN TÍNH BAN ĐẦU)
builder.add_edge(START, "detect_and_triage")
builder.add_edge("detect_and_triage", "context_loader")

# Thay vì dùng Router điều kiện cũ, ta nối thẳng sang Executor để khởi động pha Khảo sát!
builder.add_edge("context_loader", "executor")

# Định tuyến từ Executor
builder.add_conditional_edges(
    "executor",
    routers.executor_router,
    {
        "tool_node": "tool_node",                  
        "tester": "tester",
        "replanner": "replanner",
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

# Replanner sau khi đề xuất xong sẽ chạy qua Node Interrupt để tạm dừng đồ thị
builder.add_edge("replanner", "replanner_interrupt") 

# Node Interrupt sau khi được resume mới chạy Router để định tuyến tiếp
builder.add_conditional_edges(
    "replanner_interrupt",                      
    routers.replanner_router,
    {
        "executor": "executor",
        "synthesis": "synthesis"
    }
)

# Định tuyến sau khi chạy Tester
builder.add_conditional_edges(
    "tester",
    routers.tester_router,
    {
        "executor": "executor",
        "replanner": "replanner",   
        "commit": "commit"                              
    }
)

builder.add_edge("synthesis", "commit")
builder.add_edge("commit", END)

memory = MemorySaver()
app = builder.compile(checkpointer=memory)