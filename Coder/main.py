# main.py (Tệp tin cấu hình hoàn chỉnh đồ thị tích hợp luồng Doubt-Driven)

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from state import AgentState
import nodes
import routers

builder = StateGraph(AgentState)

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
# 🌟 ĐĂNG KÝ CÁC NÚT THẨM ĐỊNH ĐỐI KHÁNG MỚI (DOUBT-DRIVEN WORKFLOW)
builder.add_node("doubt_reviewer", nodes.doubt_reviewer_node)
builder.add_node("doubt_gate", nodes.doubt_gate_node)

# =====================================================================
# 2. THIẾT LẬP CÁC CẠNH NỐI CHÍNH (EDGES & ROUTERS)
# =====================================================================
builder.add_edge(START, "detect_and_triage")
builder.add_edge("detect_and_triage", "executor")

# Định tuyến từ Executor (Kiểm tra xem cần gọi Tool, Test tĩnh hay Lập kế hoạch)
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
# Cập nhật định tuyến từ Tool Node qua Gate tương tác trung gian
builder.add_conditional_edges(
    "tool_node", 
    routers.tool_router, 
    {
        "executor": "executor",
        "human_interaction_gate": "human_interaction_gate"
    }
)
builder.add_edge("human_interaction_gate", "executor")

# Định tuyến từ Tester (Kiểm tra tĩnh thành công sẽ rẽ hướng sang Doubt Reviewer)
builder.add_conditional_edges(
    "tester", 
    routers.tester_router, 
    {
        "executor": "executor",
        "chrome_extension_debugger": "chrome_extension_debugger",
        "doubt_reviewer": "doubt_reviewer",     # 🌟 Rẽ hướng sang đối kháng
        "replanner": "replanner",   
        "commit": "commit"                              
    }
)

# Định tuyến từ Chrome Extension Debugger (Kiểm tra động thành công rẽ sang Doubt Reviewer)
builder.add_conditional_edges(
    "chrome_extension_debugger", 
    routers.debugger_router, 
    {
        "executor": "executor",
        "doubt_reviewer": "doubt_reviewer",     # 🌟 Rẽ hướng sang đối kháng
        "replanner": "replanner",
        "synthesis": "synthesis"
    }
)

# 🌟 ĐỊNH TUYẾN SAU KHI PHÂN TÍCH ĐỐI KHÁNG XONG (DOUBT REVIEWER)
builder.add_conditional_edges(
    "doubt_reviewer",
    routers.doubt_router,
    {
        "doubt_gate": "doubt_gate",             # Đi tới Nút Ngắt tương tác nếu có nghi ngờ
        "synthesis": "synthesis",               # Đi tiếp tới đóng gói nếu sạch lỗi logic
        "commit": "commit"                      # Đi thẳng tới commit (với tác vụ đơn giản)
    }
)

# 🌟 ĐỊNH HƯỚNG TỪ NÚT NGẮT ĐỐI KHÁNG (DOUBT GATE)
# Sử dụng một hàm lambda để xác định: Nếu tồn tại error_logs (do người dùng yêu cầu sửa lỗi),
# đồ thị sẽ quay ngược về 'executor'. Ngược lại, đi tiếp tới khâu đóng gói hoặc commit.
builder.add_conditional_edges(
    "doubt_gate",
    lambda state: "executor" if state.get("error_logs") else ("synthesis" if state.get("plan") else "commit"),
    {
        "executor": "executor",                 # Quay lại sửa code dựa trên phản hồi nghi ngờ
        "synthesis": "synthesis",               # Tiến tới đóng gói tài liệu tổng hợp
        "commit": "commit"                      # Commit trực tiếp (với tác vụ đơn giản)
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

# Sử dụng Memory để lưu lại checkpoint, phục vụ việc hồi phục sau khi interrupt()
memory = MemorySaver()
app = builder.compile(checkpointer=memory)