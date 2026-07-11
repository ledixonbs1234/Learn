# oder/routers.py
import json
from typing import Literal
from langchain_core.messages import AIMessage, HumanMessage
from state import AgentState, Task

def executor_router(state: AgentState) -> Literal["executor", "tool_node", "tester", "replanner", "synthesis", "context_compressor"]:
    messages = state["messages"]
    plan = state.get("plan", [])
    task_type = state.get("task_type", "development")
    
    if not messages:
        if state.get("is_simple"):
            return "tester" if task_type == "development" else "synthesis"
        return "replanner"
          
    last_message = messages[-1]
    
    # 1. Agent đang gọi công cụ (ví dụ: đọc file, tìm kiếm)
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_node"
        
    # Check if the last message is a warning retry feedback
    if isinstance(last_message, HumanMessage) and "⚠️ Cảnh báo: Bạn chưa thực hiện chỉnh sửa" in str(last_message.content):
        return "executor"

    # =====================================================================
    # 2. CHỈNH SỬA TẠI ĐÂY: CHUYỂN PHA SANG ĐIỂM DỌN DẸP NGỮ CẢNH TRƯỚC KHI LẬP KẾ HOẠCH
    # =====================================================================
    # Kiểm tra xem nhiệm vụ khảo sát lính canh T_SURVEY đã hoàn thành chưa [1]
    is_survey_transition = (
        len(plan) == 1 and 
        (plan[0].id if isinstance(plan[0], Task) else plan[0].get("id")) == "T_SURVEY" and
        (plan[0].status if isinstance(plan[0], Task) else plan[0].get("status")) == "completed"
    )
    
    # Nếu T_SURVEY đã hoàn tất hoặc Executor ở pha analysis vừa dừng lại không gọi tool nữa
    if is_survey_transition or (task_type == "analysis" and not (isinstance(last_message, AIMessage) and last_message.tool_calls)):
        return "context_compressor" # Thay vì chuyển thẳng tới "replanner", ta chuyển sang "context_compressor"

    has_pending_tasks = any(
        (t.status if isinstance(t, Task) else t.get("status")) == "pending" 
        for t in plan
    )   

    # 3. Agent báo cáo đã xong lượt chạy và có file bị sửa đổi ở pha Development
    if task_type == "development" and state.get("modified_files") and not has_pending_tasks:
        return "tester"
        
    # 4. Tác vụ đơn giản không sửa code
    if state.get("is_simple"):
        return "synthesis"
        
    # 5. Nếu không còn nhiệm vụ tồn đọng -> Kết thúc
    if not has_pending_tasks:
        return "synthesis"
        
    # 6. Nếu vẫn còn nhiệm vụ tồn đọng -> Chuyển tiếp tới replanner để cập nhật hoặc điều phối tiếp
    return "replanner"

def tester_router(state: AgentState) -> Literal["executor", "chrome_extension_debugger", "replanner", "doubt_reviewer", "commit"]:
    """
    Định tuyến từ Nút Kiểm thử tĩnh.
    Nếu kiểm thử thành công, đồ thị bắt buộc đi qua Nút Phản biện đối kháng (doubt_reviewer).
    """
    error = state.get("error_logs", "")
    attempts = state.get("attempts", 0)
    is_simple = state.get("is_simple", False)
    extension_path = state.get("extension_path", "")
    
    # 1. Nếu có lỗi kiểm tra tĩnh (cú pháp/biên dịch) và chưa quá 3 lần thử -> Quay lại sửa code [1]
    if error and attempts < 3:
        return "executor"
        
    # 2. Nếu kiểm tra tĩnh ĐÃ THÀNH CÔNG:
    if not error:
        # Nếu là Chrome Extension: Chuyển sang kiểm thử động (runtime) trước
        if extension_path:
            return "chrome_extension_debugger"
        # Dự án thông thường: Chuyển thẳng sang bước Hoài nghi đối kháng
        return "doubt_reviewer"
        
    # 3. Các trường hợp lỗi nhưng đã vượt quá 3 lần thử sửa tự động
    if is_simple:
        return "commit"
        
    return "replanner"
# 🌟 ĐỊNH TUYẾN MỚI THUỘC LUỒNG THẨM ĐỊNH ĐỐI KHÁNG (DOUBT FLOW)
def doubt_router(state: AgentState) -> Literal["executor", "doubt_gate", "synthesis", "commit"]:
    """
    Định tuyến sau khi Nút Hoài nghi đối kháng (doubt_reviewer) hoàn tất rà soát.
    """
    findings = state.get("doubt_findings", "")
    is_simple = state.get("is_simple", False)
    
    # 1. Nếu không phát hiện nghi ngờ lỗi logic nào nghiêm trọng:
    if not findings:
        # Tác vụ đơn giản -> Đi tới commit. Tác vụ phức tạp -> Tổng hợp tài liệu.
        return "commit" if is_simple else "synthesis"
        
    # 2. Nếu phát hiện lỗ hổng/nghi ngờ -> Chuyển sang Nút Ngắt để hỏi ý kiến người dùng
    return "doubt_gate"
def debugger_router(state: AgentState) -> Literal["executor", "replanner", "doubt_reviewer", "synthesis"]:
    """
    Định tuyến từ Nút Kiểm thử động (Chrome DevTools).
    """
    runtime_error = state.get("error_logs", "")
    is_simple = state.get("is_simple", False)
    
    # Nếu phát hiện lỗi crash runtime của Extension -> Đưa thông tin lỗi quay lại để sửa đổi
    if runtime_error:
        return "executor" if is_simple else "replanner"
        
    # Nếu chạy mượt mà không có lỗi: Bắt buộc chuyển sang bước Hoài nghi đối kháng trước khi đóng gói
    return "doubt_reviewer"
def replanner_router(state: AgentState) -> Literal["executor", "synthesis"]:
    plan = state["plan"]
    
    pending_tasks = []
    for t in plan:
        status = t.status if isinstance(t, Task) else t.get("status")
        if status == "pending":
            pending_tasks.append(t)
            
    if pending_tasks:
        return "executor"
    else:
        return "synthesis"


def tool_router(state: AgentState) -> Literal["executor", "human_interaction_gate", "context_compressor"]:
    """
    Định tuyến sau khi chạy công cụ.
    - Chuyển hướng sang 'context_compressor' ngay lập tức nếu vừa hoàn thành một Task để dọn dẹp RAM [1].
    - Chuyển sang 'human_interaction_gate' nếu cần hỏi ý kiến người dùng.
    - Mặc định quay về 'executor' để tiếp tục xử lý công cụ khác.
    """
    messages = state["messages"]
    
    # Kiểm tra xem cuộc gọi công cụ gần nhất có chứa complete_agent_task không
    last_ai_message = None
    for msg in reversed(messages):
        if msg.type == "ai" or isinstance(msg, AIMessage):
            last_ai_message = msg
            break
            
    if last_ai_message and getattr(last_ai_message, "tool_calls", None):
        if any(tc["name"] == "complete_agent_task" for tc in last_ai_message.tool_calls):
            # Kích hoạt dọn dẹp và nén ngữ cảnh khẩn cấp trước khi đi tiếp [1]
            return "context_compressor"
    
    # Kiểm tra yêu cầu tương tác người dùng
    for msg in reversed(messages):
        if getattr(msg, "type", None) != "tool":
            break
        if msg.name == "ask_questions_if_underspecified":
            try:
                data = json.loads(msg.content)
                if isinstance(data, dict) and data.get("status") == "requires_human_response":
                    return "human_interaction_gate"
            except Exception:
                pass
                
    return "executor"

