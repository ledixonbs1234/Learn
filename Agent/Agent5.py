from typing import Annotated, Sequence, TypedDict
from langchain_core import tools
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_tavily import TavilySearch
from langgraph.graph import END, START
from langgraph.graph.message import StateGraph, add_messages
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
import os



class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage],add_messages]

os.environ['TAVILY_API_KEY'] ='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'
search_tool = TavilySearch(max_results =3)
tools = [search_tool]

model = ChatOpenAI(
    base_url="http://localhost:20128/v1",
    api_key="khongco",
    model="kiro",
    temperature=0.7
).bind_tools(tools)

def call_model(state:AgentState)->AgentState:
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages":[response]}

tool_node = ToolNode(tools)

def should_continue(state:AgentState):
    messages = state["messages"]
    last_message = messages[-1]

    if last_message.tool_calls:
        return "tools"
    return END

graph = StateGraph(AgentState)
graph.add_node('agent',call_model)
graph.add_node("tools",tool_node)
graph.add_edge(START,'agent')

graph.add_conditional_edges("agent",should_continue)

graph.add_edge('tools','agent')

app = graph.compile()
try:
    # Xuất đồ thị thành định dạng ảnh PNG
    png_bytes = app.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(png_bytes)
    print("Đã vẽ sơ đồ thành công! Bạn hãy mở file 'graph.png' trong thư mục để xem.")
except Exception as e:
    print(f"Không thể xuất ảnh PNG: {e}")


query = {"messages":"Sự kiện công nghệ mới nhất tuần này là gì"}

for event in app.stream(query):
    for key,value in event.items():
        print(f"\n[Node: {key}]")
        print(value["messages"][-1].content)
