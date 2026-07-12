# oder/main.py
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Nhập đầy đủ 3 Schema từ state.py
from state import AgentState, AgentInputState, AgentOutputState
import nodes
import routers

# Áp dụng cấu hình phân tách Input và Output cho StateGraph
builder = StateGraph(
    state_schema=AgentState,
    input=AgentInputState,
    output=AgentOutputState
)

# =====================================================================
# 1. ĐĂNG KÝ CÁC NÚT HOẠT ĐỘNG (NODES)
# =====================================================================
builder.add_node("detect_and_triage", nodes.detect_and_triage_node) 
builder.add_node("executor", nodes.executor_node)  
builder.add_node("tool_node", nodes.tool_node) 
builder.add_node("human_interaction_gate", nodes.human_interaction_gate_node)
builder.add_node("chrome_extension_debugger", nodes.chrome_extension_debugger_node)
builder.add_node("replanner", nodes.replanner_node)
builder.add_node("replanner_interrupt", nodes.replanner_interrupt_node) 
builder.add_node("tester", nodes.tester_node)
builder.add_node("synthesis", nodes.synthesis_node)
builder.add_node("commit", nodes.commit_node)
builder.add_node("context_compressor", nodes.context_compressor_node)
builder.add_node("fluxmem_distillation", nodes.fluxmem_distillation_node)

# ĐĂNG KÝ CÁC NÚT THẨM ĐỊNH ĐỐI KHÁNG
builder.add_node("doubt_reviewer", nodes.doubt_reviewer_node)
builder.add_node("doubt_gate", nodes.doubt_gate_node)

# =====================================================================
# 2. THIẾT LẬP CÁC CẠNH NỐI CHÍNH (EDGES & ROUTERS)
# =====================================================================
builder.add_edge(START, "detect_and_triage")
builder.add_edge("detect_and_triage", "executor")

# Định tuyến từ Executor
builder.add_conditional_edges(
    "executor",
    routers.executor_router,
    {
        "executor": "executor",
        "tool_node": "tool_node",                  
        "tester": "tester",
        "replanner": "replanner",
        "synthesis": "synthesis",
        "context_compressor": "context_compressor" 
    }
)
builder.add_edge("context_compressor", "replanner")
builder.add_edge("fluxmem_distillation", "context_compressor")

# Định tuyến từ Tool Node
builder.add_conditional_edges(
    "tool_node", 
    routers.tool_router, 
    {
        "executor": "executor",
        "human_interaction_gate": "human_interaction_gate",
        "fluxmem_distillation": "fluxmem_distillation"
    }
)
builder.add_edge("human_interaction_gate", "executor")

# Định tuyến từ Tester
builder.add_conditional_edges(
    "tester", 
    routers.tester_router, 
    {
        "executor": "executor",
        "chrome_extension_debugger": "chrome_extension_debugger",
        "doubt_reviewer": "doubt_reviewer",
        "replanner": "replanner",   
        "commit": "commit"                              
    }
)

# Định tuyến từ Chrome Extension Debugger
builder.add_conditional_edges(
    "chrome_extension_debugger", 
    routers.debugger_router, 
    {
        "executor": "executor",
        "doubt_reviewer": "doubt_reviewer",
        "replanner": "replanner",
        "synthesis": "synthesis"
    }
)

# ĐỊNH TUYẾN SAU KHI PHÂN TÍCH ĐỐI KHÁNG XONG (DOUBT REVIEWER)
builder.add_conditional_edges(
    "doubt_reviewer",
    routers.doubt_router,
    {
        "doubt_gate": "doubt_gate",
        "synthesis": "synthesis",
        "commit": "commit"
    }
)

# ĐỊNH HƯỚNG TỪ NÚT NGẮT ĐỐI KHÁNG (DOUBT GATE)
builder.add_conditional_edges(
    "doubt_gate",
    lambda state: "executor" if state.get("error_logs") else ("synthesis" if state.get("plan") else "commit"),
    {
        "executor": "executor",
        "synthesis": "synthesis",
        "commit": "commit"
    }
)

# Luồng lập kế hoạch lại (Replanner & Interrupt)
builder.add_edge("replanner", "replanner_interrupt") 
builder.add_conditional_edges("replanner_interrupt", routers.replanner_router, {
    "executor": "executor",
    "synthesis": "synthesis"
})

builder.add_edge("synthesis", "commit")
builder.add_edge("commit", END)

memory = MemorySaver()
app = builder.compile(checkpointer=memory)