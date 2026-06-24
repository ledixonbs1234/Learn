import os
from typing import Annotated, Sequence, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def bank_transfer(amount: float, recipient:str):
    """Thực hiện chuyển tiền ngân hàng đến người nhận."""
    # Trong ứng dụng thực tế, đây là nơi gọi API của ngân hàng để trừ tiền thật
    print(f"\n⚡ [HỆ THỐNG] ĐANG THỰC HIỆN CHUYỂN {amount} VND ĐẾN {recipient} THẬT...")
    return f"Giao dịch thành công: Đã chuyển {amount} VND tới tài khoản của {recipient}."

tools = [bank_transfer]

model = ChatOpenAI(
    base_url="http://localhost:20128/v1",
    api_key="khongco",
    model="kiro",
    temperature=0.1
).bind_tools(tools)

def agent_node(state:AgentState):
    print("\n--- 🤖 [NODE: AGENT] ĐANG SUY NGHĨ VÀ PHÂN TÍCH ---")
    response = model.invoke(state["messages"])
    return {"messages": [response]}

tool_node = ToolNode(tools)

def should_continue(state:AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return 'tools'
    print("   [Routing] Không có yêu cầu gọi Tool. Kết thúc.")
    return END

builder = StateGraph(AgentState)
builder.add_node("agent",agent_node)
builder.add_node('tools',tool_node)
builder.add_edge(START,'agent')
builder.add_edge('tools','agent')

builder.add_conditional_edges(
    "agent",
    should_continue,
    {
       "tools":'tools',
       END:END 
    }
)

memory = MemorySaver()

app = builder.compile(
    checkpointer=memory,
    interrupt_before=['tools']
)

def main():
    config = {"configurable": {"thread_id": "sensitive_payment_session"}}
    
    user_query = "Hãy chuyển ngay 500,000 VND tới tài khoản của anh Nguyễn Văn A hộ tôi."
    print(f"Yêu cầu của người dùng: '{user_query}'")
    
    # --- BƯỚC 1: KHỞI CHẠY ĐỒ THỊ ---
    print("\n[Hệ thống] Khởi động Agent...")
    for event in app.stream({"messages": [HumanMessage(content=user_query)]}, config, stream_mode="values"):
        pass

    # --- BƯỚC 2: KIỂM TRA TRẠNG THÁI KHI BỊ CHẶN LẠI ---
    state_snapshot = app.get_state(config)
    last_message = state_snapshot.values["messages"][-1]
    
    if not last_message.tool_calls:
        print("Agent không yêu cầu chuyển tiền. Chương trình kết thúc.")
        return
    tool_call = last_message.tool_calls[0]
    tool_name = tool_call['name']
    tool_args = tool_call['args']
    tool_call_id = tool_call['id']
    print("\n================ 🚨 CẢNH BÁO GIAO DỊCH NHẠY CẢM ================")
    print(f"Agent đang yêu cầu thực thi Tool: '{tool_name}'")
    print(f"Thông số giao dịch: {tool_args}")
    print("================================================================")
    
    # --- BƯỚC 3: CON NGƯỜI QUYẾT ĐỊNH ---
    print("\nBạn có cho phép thực hiện giao dịch này không?")
    print("1. ĐỒNG Ý (Approve) - Cho phép chuyển tiền thật.")
    print("2. TỪ CHỐI (Reject) - Hủy giao dịch và yêu cầu Agent giải thích.")
    choice = input("Nhập lựa chọn của bạn (1 hoặc 2): ").strip()
    
    if choice ==1:
        print("\n[Hệ thống] Bạn đã phê duyệt giao dịch. Đang chạy tiếp...")
        for event in app.stream(None, config, stream_mode="values"):
            pass
    else:
        # NẾU TỪ CHỐI: Áp dụng kỹ thuật Bypass Node (Bỏ qua Node)
        print("\n[Hệ thống] Bạn đã TỪ CHỐI giao dịch. Tiến hành bypass tool...")
        
        rejection_message = ToolMessage(
            content="Giao dịch bị từ chối: Người dùng không phê duyệt yêu cầu chuyển tiền này do lý do an toàn bảo mật.",
            tool_call_id = tool_call_id
        )
        
         # KỸ THUẬT QUAN TRỌNG: 
        # Cập nhật State dưới danh nghĩa node "tools" (as_node="tools")
        # Việc này báo cho đồ thị biết rằng Node "tools" coi như đã thực thi xong và trả về kết quả từ chối này.
        app.update_state(config,
                         {
                             "messages":[rejection_message],
                             
                         },
                         as_node='tools')
        for event in app.stream(None, config, stream_mode="values"):
            pass
        
    final_state = app.get_state(config)
    print("\n================ KẾT QUẢ CUỐI CÙNG XUẤT RA CHO USER ================")
    print(final_state.values["messages"][-1].content)
    print("====================================================================")

if __name__ == "__main__":
    main()
