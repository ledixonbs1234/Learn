import os
import sys
from typing import Annotated, Dict, List, Any, Literal, Sequence, TypedDict

from langchain_core.messages import (
    BaseMessage, 
    HumanMessage, 
    AIMessage, 
    SystemMessage, 
    ToolMessage,
    RemoveMessage
)
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    base_url="http://localhost:20128/v1",
    api_key="khongco",
    model="kiro",
    temperature=0.1
)

class AgentMemoryState(TypedDict):
    messages:Annotated[Sequence[BaseMessage],add_messages]
    
    episodic_logs: List[Dict[str,Any]]
    long_term_lessons:List[str]
    current_code:str
    target_file:str
    
    
@tool
def run_python_code(code: str) -> str:
    """Chạy thử đoạn mã Python và trả về kết quả hoặc lỗi chi tiết từ compiler."""
    # Quy chuẩn nội bộ giả định: Nghiêm cấm sử dụng hàm print() thông thường để log,
    # phải sử dụng thư viện logging chuẩn của dự án.
    if "print(" in code:
        return "COMPILER ERROR: Vi phạm quy chuẩn nội bộ! Không được dùng hàm 'print()'. Hãy sử dụng 'logging.info()' hoặc 'logging.error()'."
    
    # Giả lập chạy code thành công hoặc lỗi logic khác
    if "logger.info" in code:
        return "SUCCESS: Chương trình chạy hoàn tất không có lỗi."
    else:
        return "COMPILER ERROR: Code không chứa cấu trúc logging chuẩn. Không thể phê duyệt."


@tool
def save_lesson_to_long_term_memory(lesson: str) -> str:
    """Lưu trữ một bài học kinh nghiệm sâu sắc vào bộ nhớ dài hạn để các lần chạy sau không mắc lại lỗi này."""
    # Trong thực tế, bạn sẽ ghi nhận bài học này vào một file JSON, Database hoặc Vector DB.
    # Ở đây chúng ta trả về một thông báo xác nhận để Agent biết nó đã ghi nhớ thành công.
    return f"SUCCESS: Bài học '{lesson}' đã được lưu trữ vĩnh viễn vào bộ nhớ dài hạn."

tools = [run_python_code,save_lesson_to_long_term_memory]

model = model.bind_tools(tools)

def condense_memory_node(state: AgentMemoryState) ->AgentMemoryState:
    """
    Node này có nhiệm vụ kiểm duyệt lịch sử tin nhắn.
    Nếu số lượng tin nhắn vượt quá 8, nó sẽ tự động giữ lại tin nhắn đầu tiên (yêu cầu),
    tin nhắn cuối cùng, và thực hiện xóa các ToolMessage và AIMessage trung gian bằng RemoveMessage
    để giải phóng bộ nhớ, tránh trôi ngữ cảnh.
    """
    messages = state['messages']
    updates={}

    if len(messages) > 8:
        messages_to_remove = []
        # Giữ lại tin nhắn đầu tiên (yêu cầu người dùng) và 2 tin nhắn gần nhất
        # Các tin nhắn ở giữa sẽ bị đánh dấu xóa bằng RemoveMessage
        for msg in messages[1:-2]:
            if msg.id:
                messages_to_remove.append(RemoveMessage(id =msg.id))

        if messages_to_remove:
            updates['messages'] = messages_to_remove

    return updates

def code_agent_node(state: AgentMemoryState)->AgentMemoryState:
    """
    Đại lý chịu trách nhiệm viết code. Nó nhận thức được cả 
    Episodic Memory (lỗi vừa gặp) và Long-term Memory (bài học dài hạn).
    """
    
    messages = state['messages']
    long_term = state.get("long_term_lessons",[])
    episodic = state.get('episodic_logs',[])

    
    # Tạo chỉ thị hệ thống kết hợp với các lớp Bộ nhớ
    system_instruction = (
        "Bạn là một AI Developer chuyên nghiệp, có kỹ năng lập trình đỉnh cao.\n"
        "Hãy tuân thủ nghiêm ngặt các quy chuẩn viết code của dự án.\n\n"
        "--- BỘ NHỚ DÀI HẠN (LONG-TERM LESSONS) ---\n"
    )
    
    if long_term:
        for idx, lesson in enumerate(long_term,1):
            system_instruction+=f"{idx}. {lesson}\n"
    else:
        system_instruction += "(Chưa có bài học nào được lưu trữ)\n"
    
    system_instruction += "\n--- BỘ NHỚ TÌNH HUỐNG (EPISODIC LOGS) ---\n"
    
    if episodic:
        for idx,log in enumerate(episodic,1):
            system_instruction += f"Lần thử {idx}: Code đã viết: '{log['code']}' -> Kết quả: {log['feedback']}\n"
    else:
        system_instruction += "(Chưa có lịch sử thử nghiệm trong phiên này)\n"
    
    system_instruction += (
        "\nNhiệm vụ của bạn: Hãy viết mã nguồn đáp ứng yêu cầu của người dùng.\n"
        "Sử dụng công cụ `run_python_code` để kiểm tra mã nguồn trước khi bàn giao.\n"
        "Nếu bạn phát hiện ra một quy chuẩn quan trọng hoặc một bài học xương máu giúp tránh lỗi biên dịch, "
        "hãy chủ động gọi công cụ `save_lesson_to_long_term_memory` để lưu lại."
    )
    
    full_message = [SystemMessage(content=system_instruction)]+messages

    response = model.invoke(full_message)
    return {'messages':[response]}

def execute_tool_node(state:AgentMemoryState)->AgentMemoryState:
    """Node bổ trợ thực thi các Tool Calls từ Agent và cập nhật lại State."""
    messages = state['messages']
    last_message = messages[-1]
    
    tool_output = []
    episodic_updates = []
    new_lesson = list(state.get('long_term_lessons',[]))
    current_code= state.get('current_code',"")

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]
            
            if tool_name == 'run_python_code':
                code_to_run = tool_args.get('code','')
                current_code = code_to_run
                result = run_python_code.invoke(tool_args)
                # Lưu vào Episodic Memory (Bộ nhớ tình huống) để Agent tự theo dõi tiến trình của mình
                episodic_updates.append({
                    'code':code_to_run,
                    'feedback':result
                })
                
                tool_output.append(ToolMessage(content=result,tool_call_id=tool_id))
            
            elif tool_name == "save_lesson_to_long_term_memory":
                lesson_content = tool_args.get('lesson',"")
                result = save_lesson_to_long_term_memory.invoke(tool_args)
# Cập nhật trực tiếp bài học mới vào danh sách Long-term Memory của State hiện tại
                if lesson_content not in new_lesson:
                    new_lesson.append(lesson_content)
                
                tool_output.append(ToolMessage(content=result,tool_call_id = tool_id))
    
    return{
        'messages':tool_output,
        'episodic_logs':state.get('episodic_logs',[])+ episodic_updates,
        'long_term_lessons':new_lesson,
        "current_code":current_code
    }
# ==========================================
# 6. NODE 4: COMPILER & REVIEWER (Kiểm duyệt & Đánh giá)
# ==========================================

def code_reviewer_node(state: AgentMemoryState)->AgentMemoryState:
    """
    Node kiểm duyệt cuối cùng. Đọc kết quả chạy code gần nhất từ episodic_logs.
    Nếu code chạy thành công và không vi phạm quy chuẩn nào, luồng sẽ được duyệt.
    """
    episodic = state.get("episodic_logs", [])
    if not episodic:
        return {"review_status": "REJECTED"}
        
    last_run = episodic[-1]
    if "SUCCESS" in last_run["feedback"]:
        return {"review_status": "APPROVED"}
    else:
        return {"review_status": "REJECTED"}
    
# ==========================================
# 7. ĐIỀU HƯỚNG CÓ ĐIỀU KIỆN (Conditional Edges)
# ==========================================

def route_after_agent(state: AgentMemoryState)->AgentMemoryState:
    """Quyết định đi tiếp tới thực thi công cụ hay chuyển sang bước kiểm duyệt."""
    messages = state["messages"]
    last_message = messages[-1]
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "execute_tools"
    return "reviewer"

def route_after_reviewer(state: dict):
    """Nếu reviewer từ chối, quay lại Node dọn dẹp bộ nhớ và chạy lại. Nếu duyệt thì kết thúc."""
    status = state.get("review_status")
    if status == "APPROVED":
        return END
    return "condense_memory"

builder = StateGraph(AgentMemoryState)

# Thêm các Node vào Đồ thị
builder.add_node("condense_memory", condense_memory_node)
builder.add_node("code_agent", code_agent_node)
builder.add_node("execute_tools", execute_tool_node)
builder.add_node("reviewer", code_reviewer_node)

# Thiết lập các Cạnh kết nối (Edges)
builder.add_edge(START, "condense_memory")
builder.add_edge("condense_memory", "code_agent")

builder.add_conditional_edges(
    "code_agent",
    route_after_agent,
    {
        "execute_tools": "execute_tools",
        "reviewer": "reviewer"
    }
)
builder.add_edge("execute_tools", "code_agent")
builder.add_conditional_edges(
    "reviewer",
    route_after_reviewer,
    {
        "condense_memory": "condense_memory",
        END: END
    }
)

memory = MemorySaver()
app = builder.compile(checkpointer=memory)
