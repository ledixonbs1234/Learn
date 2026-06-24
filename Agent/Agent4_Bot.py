from typing import Annotated, Sequence, TypedDict

from langchain_core import tools
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END
from langgraph.graph.message import StateGraph, add_messages
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

# đây là global
document_content = ""

class AgentState(TypedDict):
    messages:Annotated[Sequence[BaseMessage],add_messages]


@tool
def update(content:str) -> str:
    """Cập nhật tài liệu với nội dung đã cung cấp"""
    global document_content
    document_content = content
    return f"Tài liệu đã cập nhật thành công! Tài liệu hiện tại là :\n{document_content}"

@tool
def save(filename:str)->str:
    """Lưu nội dung hiện tại tới file text và hoàn thành tiến trình
    Args:
        filename: Tên của file text
    """
    global document_content
    if(not filename.endswith('.txt')):
        filename = f"{filename}.txt"

    try:
        with open(filename,'w',encoding="utf-8") as file:
            file.write(document_content)
        print(f"\n Tài liệu đã được lưu vào {filename}")
        return f"\n Tài liệu đã được lưu vào {filename}"
    
    except Exception as e:
        return f" Lỗi khi lưu tài liệu : {str(e)}"

tools = [update,save]     


model = ChatOpenAI(
    base_url="http://localhost:20128/v1",
    api_key="khongco",
    model="kiro",
    temperature=0.7
).bind_tools(tools)

def our_agent(state:AgentState)->AgentState:
    system_prompt = SystemMessage(content=f"""
Bạn là Drafter, một trợ lý viết lách đắc lực. Bạn sẽ giúp người dùng cập nhật và [chỉnh sửa tài liệu...]

- Nếu người dùng muốn cập nhật hoặc chỉnh sửa nội dung, hãy sử dụng công cụ 'update' với [nội dung] đầy đủ...
- Nếu người dùng muốn lưu và hoàn tất, bạn cần sử dụng công cụ 'save'.
- Hãy đảm bảo luôn hiển thị trạng thái hiện tại của tài liệu sau khi chỉnh sửa.

Nội dung hiện tại của tài liệu là: {document_content}
""")
    
    if not state['messages']:
        user_input = ' Tôi sẵn sàng giúp bạn cập nhật tài liệu, Bạn muốn tạo như thế nào?'
        user_message = HumanMessage(content=user_input)
    else:
        user_input = input("\n Bạn muốn tạo tài liệu như thế nào: ")
        print(f"\n User : {user_input}")
        user_message = HumanMessage(content=user_input)
    all_message = [system_prompt]+list(state["messages"]) + [user_message]    
    response = model.invoke(all_message)
    
    print(f"\n AI :{response.content} ")
    if hasattr(response,"tool_calls") and response.tool_calls:
        print(f"USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")
    return {"messages":list(state['messages'])+ [user_message,response]}

def should_continue(state:AgentState)->AgentState:
    """tiep tucj hay hoan thanh"""

    messages = state["messages"]
    if not messages:
        return "continue"
    
    for message in reversed(messages):
        if (isinstance(message,ToolMessage) and
            "saved" in message.content.lower() and
            "document" in message.content.lower()):
            return "end"
    return "continue"

def print_messages(messages):
    """Ham In Ra"""
    if not messages:
        return
    
    for message in messages[-3:]:
        if isinstance(message,ToolMessage):
            print(f"\n TOOL Result:{message.content}")

graph = StateGraph(AgentState)
graph.add_node("agent",our_agent)
graph.add_node("tools",ToolNode(tools))
graph.set_entry_point("agent")
graph.add_edge("agent","tools")
graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue":"agent",
        "end":END
    }
)
app =graph.compile()
try:
    # Xuất đồ thị thành định dạng ảnh PNG
    png_bytes = app.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(png_bytes)
    print("Đã vẽ sơ đồ thành công! Bạn hãy mở file 'graph.png' trong thư mục để xem.")
except Exception as e:
    print(f"Không thể xuất ảnh PNG: {e}")
def run_document_agent():
    print("\n ======== Drafter===========")

    state={"messages":[]}
    for step in app.stream(state,stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])
    
    print("\n ========Finish============")

if __name__ == "__main__":
    run_document_agent()