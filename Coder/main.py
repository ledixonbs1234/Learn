# oder/main.py
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from state import AgentState
import nodes
import routers

builder = StateGraph(AgentState)

# 1. ĐĂNG KÝ CÁC NÚT (NODES)
builder.add_node("detect_and_triage", nodes.detect_and_triage_node) 
builder.add_node("context_loader", nodes.context_loader_node)
builder.add_node("executor", nodes.executor_node)  
builder.add_node("tool_node", nodes.tool_node) 

builder.add_node("chrome_extension_debugger", nodes.chrome_extension_debugger_node) # Đăng ký Debugger Node

builder.add_node("replanner", nodes.replanner_node)
builder.add_node("replanner_interrupt", nodes.replanner_interrupt_node) 

builder.add_node("tester", nodes.tester_node)
builder.add_node("synthesis", nodes.synthesis_node)
builder.add_node("commit", nodes.commit_node)

# 2. THIẾT LẬP CÁC CẠNH NỐI CHÍNH (EDGES)
builder.add_edge(START, "detect_and_triage")
builder.add_edge("detect_and_triage", "context_loader")
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

# Định tuyến sau khi chạy Tester (Tích hợp luồng Debugger ở đây)
builder.add_conditional_edges(
    "tester",
    routers.tester_router,
    {
        "executor": "executor",
        "chrome_extension_debugger": "chrome_extension_debugger", # Chuyển sang kiểm thử động
        "replanner": "replanner",   
        "commit": "commit"                              
    }
)

# Định tuyến sau khi chạy Debugger
builder.add_conditional_edges(
    "chrome_extension_debugger",
    routers.debugger_router,
    {
        "executor": "executor",         # Nếu là task đơn giản bị lỗi -> Sửa trực tiếp
        "replanner": "replanner",       # Nếu là task phức tạp bị lỗi -> Lập kế hoạch lại
        "synthesis": "synthesis"       # Nếu thành công -> Tổng hợp tài liệu
    }
)

# Luồng Replanner và Interrupt
builder.add_edge("replanner", "replanner_interrupt") 

builder.add_conditional_edges(
    "replanner_interrupt",                      
    routers.replanner_router,
    {
        "executor": "executor",
        "synthesis": "synthesis"
    }
)

builder.add_edge("synthesis", "commit")
builder.add_edge("commit", END)

memory = MemorySaver()
app = builder.compile(checkpointer=memory)