
from typing import Annotated, Sequence, TypedDict

from langchain_core import tools
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import END
from langgraph.graph.message import StateGraph, add_messages
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode


model = ChatOpenAI(
    base_url="http://localhost:20128/v1",
    api_key="khongco",
    model="kiro",
    temperature=0.7
)

class AgentState(TypedDict):
    message:Annotated[Sequence[BaseMessage],add_messages]

@tool
def add(a:int,b:int)->int:
    """Đây là hàm cộng"""

    return a+b
@tool
def subtract(a:int,b:int)->int:
    """Đây là hàm trừ"""

    return a-b

tools = [add,subtract]
model =model.bind_tools(tools)

def model_call(state:AgentState)->AgentState:
    system_prompt = SystemMessage(content="Bạn là AI assistant, làm ơn trả lời yêu cầu của tôi")
    response = model.invoke([system_prompt] + state['message'])
    # print(f"Đây là reponse chạy trong hàm model_call \n{response}",)
    return {"message":[response]}

def should_continue(state: AgentState):
    message = state['message']
    last_message = message[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

graph = StateGraph(AgentState)
graph.add_node("out_agent",model_call)

tool_node = ToolNode(tools=tools,messages_key="message")
graph.add_node("tools",tool_node)
graph.set_entry_point("out_agent")
graph.add_conditional_edges("out_agent",
                            should_continue,
                            {
                                "continue":"tools",
                                "end":END
                            })
graph.add_edge("tools","out_agent")
app = graph.compile()
try:
    # Xuất đồ thị thành định dạng ảnh PNG
    png_bytes = app.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(png_bytes)
    print("Đã vẽ sơ đồ thành công! Bạn hãy mở file 'graph.png' trong thư mục để xem.")
except Exception as e:
    print(f"Không thể xuất ảnh PNG: {e}")
def print_stream(stream):
    for s in stream:
        message = s["message"][-1]
        if isinstance(message,tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"message":[("user"," tính 3+4, sau đó lấy kết quả trừ đi 2 , từ kết quả ta lại cộng cho 10")]}
print_stream(app.stream(inputs,stream_mode="values"))