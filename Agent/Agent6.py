from typing import Annotated, Sequence, TypedDict
from langchain_core import tools
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_tavily import TavilySearch
from langgraph.graph import END, START,StateGraph
from langgraph.graph.message import  add_messages
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
import os

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage],add_messages]

os.environ['TAVILY_API_KEY'] ='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'


config = {"configurable":{"thread_id":"session_1"}}
search_tool = TavilySearch(max_results =3)

@tool
def check_inventory(product_name:str)->str:
    """Kiểm tra số lượng tồn kho của 1 sản phẩm cụ thể"""
    if "iphone" in product_name.lower():
        return "Sản phẩm hiện giờ còn 5 chiếc trong kho"
    return f"Sản phẩm {product_name} hiện đã hết hàng"

tools = [check_inventory]

model = ChatOpenAI(
    base_url="http://localhost:20128/v1",
    api_key="khongco",
    model="kiro",
    temperature=0.7
).bind_tools(tools)

def agent(state:AgentState):
    print("\n--- 🤖 AGENT ĐANG SUY NGHĨ VÀ GỌI LLM ---")
    response = model.invoke(state['messages'])
    return {"messages":[response]}

tool_node = ToolNode(tools)

def human_review(state:AgentState):
    print("\n--- 👁️ HỆ THỐNG ĐÃ ĐI QUA BƯỚC DUYỆT (KẾT THÚC) ---")
    pass

def should_continue(state:AgentState):
    messages = state["messages"]
    last_message = messages[-1]

    if last_message.tool_calls:
        print(f"   [Routing] Phát hiện Tool Call: {last_message.tool_calls[0]['name']}")
        return "tools"
    print("   [Routing] Agent đã có câu trả lời cuối cùng. Chuyển sang bước kiểm duyệt...")
    return "human_review"


graph = StateGraph(AgentState)
graph.add_node("agent",agent)
graph.add_node('tools',tool_node)
graph.add_node('human_review',human_review)
graph.add_edge('tools', 'agent')       # Từ tools quay lại agent
graph.add_edge('human_review', END) 
graph.add_edge(START,'agent')
graph.add_conditional_edges('agent',
                            should_continue,
                            {
                                "tools":"tools",
                                "human_review":"human_review"
                            })
memory = MemorySaver()

app = graph.compile(checkpointer=memory,interrupt_before=['human_review'])
try:
    # Xuất đồ thị thành định dạng ảnh PNG
    png_bytes = app.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(png_bytes)
    print("Đã vẽ sơ đồ thành công! Bạn hãy mở file 'graph.png' trong thư mục để xem.")
except Exception as e:
    print(f"Không thể xuất ảnh PNG: {e}")

#Chay

def main():
    config = {'configurable':{'thread_id':"session_1"}}
    user_query = "Tôi muốn mua điện thoại iPhone, trong kho còn hàng không và tôi có nên mua lúc này không?"
    print(f"Yêu cầu của người dùng: '{user_query}'")

    print("\n[Hệ thống] Đang khởi động Agent...")
    for event in app.stream({"messages":[HumanMessage(content=user_query)]},config,stream_mode='values'):
        pass
    state_snapshot = app.get_state(config)

    last_message = state_snapshot.values["messages"][-1]
    
    print("\n==================================================")
    print("🤖 CÂU TRẢ LỜI NHÁP CỦA AGENT:")
    print(last_message.content)
    print("==================================================")

    print("\nBạn muốn xử lý thế nào?")
    print("1. Đồng ý và cho xuất câu trả lời này.")
    print("2. Chỉnh sửa lại câu trả lời trước khi xuất.")
    choice = input("Nhập lựa chọn của bạn (1 hoặc 2): ").strip()    

    if choice == "2":
        new_content = input("\nNhập nội dung câu trả lời mới đã chỉnh sửa của bạn:\n> ")
        edited_message = AIMessage(content=new_content,id=last_message.id)

        app.update_state(config,{"messages":[edited_message]},as_node="agent")
        print("\n[Hệ thống] Đã chỉnh sửa thành công nội dung câu trả lời trong State!")
    else:
        print("\n[Hệ thống] Bạn đã phê duyệt câu trả lời gốc.")

    print("\n[Hệ thống] Đang tiếp tục chạy đồ thị để hoàn tất...")
    for event in app.stream(None, config, stream_mode="values"):
        pass

    final_state = app.get_state(config)
    final_response = final_state.values['messages'][-1].content
    print("\n================ KẾT QUẢ CUỐI CÙNG XUẤT RA CHO USER ================")
    print(final_response)
    print("====================================================================")

if __name__ == "__main__":
    main()

