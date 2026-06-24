import os
from typing import Annotated, Sequence, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver


class TeamState(TypedDict):
    messages:Annotated[Sequence[BaseMessage],add_messages]
    next:str
    
model = ChatOpenAI(
    base_url="http://localhost:20128/v1",
    api_key="khongco",
    model="kiro",
    temperature=0.3
)


# ==================================================
# 3. ĐỊNH NGHĨA CÁC WORKER NODE (MEMBER AGENTS)
# ==================================================

# Worker 1: Coder Python
def coder_worker(state:TeamState):
    print("\n--- 💻 [WORKER: CODER] ĐANG VIẾT CODE ---")
    messages = state["messages"]
    system_msg = SystemMessage(content="Bạn là một lập trình viên Python tài năng. Hãy tập trung viết code sạch, tối ưu "
                "để giải quyết phần việc được giao. Chỉ viết code trong khối ```python ... ```."
    )
    
    response = model.invoke([system_msg]+list(messages))
    response.name = 'coder'
    return {"messages":[response]}

def searcher_worker(state:TeamState):
    print("\n--- 🔍 [WORKER: SEARCHER] ĐANG TRA CỨU THÔNG TIN ---")
    messages = state["messages"]

    system_msg = SystemMessage(
        content="Bạn là một chuyên gia tìm kiếm thông tin. Hãy giải quyết phần việc được giao liên quan đến "
                "thu thập dữ liệu, phân tích ưu nhược điểm hoặc giải thích lý thuyết."
    )
    response = model.invoke([system_msg]+list(messages))
    response.name = 'searcher'
    return {"messages":[response]}

def supervisor_node(state:TeamState):
    print("\n--- 👑 [SUPERVISOR] ĐANG ĐÁNH GIÁ VÀ PHÂN CHIA VIỆC ---")
    messages = state["messages"]
    
    system_prompt = (
        "Bạn là Quản lý dự án (Supervisor). Nhiệm vụ của bạn là phân công công việc "
        "cho các thành viên trong đội dựa trên yêu cầu của người dùng.\n\n"
        "Các thành viên bao gồm:\n"
        "1. 'coder': Chuyên giải quyết các bài toán viết code, thuật toán, lập trình.\n"
        "2. 'searcher': Chuyên tra cứu thông tin thực tế, lý thuyết, giải thích khái niệm.\n\n"
        "Quy tắc điều hướng:\n"
        "- Nếu yêu cầu cần tìm hiểu lý thuyết hoặc số liệu trước, hãy phân công cho 'searcher'.\n"
        "- Nếu yêu cầu cần lập trình hoặc viết code, hãy phân công cho 'coder'.\n"
        "- Nếu các thành viên đã hoàn thành nhiệm vụ và bạn đã có câu trả lời cuối cùng để gửi người dùng, hãy chọn 'FINISH'.\n\n"
        "BẮT BUỘC TRẢ VỀ ĐỊNH DẠNG: NEXT: [coder hoặc searcher hoặc FINISH]"
    )
    
    response = model.invoke([SystemMessage(content=system_prompt)]+list(messages))
    content = response.content

    
    next_node = "FINISH"
    if 'coder' in content.lower():
        next_node = 'coder'
    elif 'searcher' in content.lower():
        next_node = 'searcher'
    print(f"-> Quyết định của Supervisor: Chuyển việc cho '{next_node}'")
    return {"next": next_node}

def supervisor_routing(state: TeamState):
    next_node = state.get('next',"FINISH")
    if next_node == "coder":
        return "coder"
    elif next_node == 'searcher':
        return 'searcher'
    return END

builder = StateGraph(TeamState)
builder.add_node('coder',coder_worker)
builder.add_node('searcher',searcher_worker)
builder.add_node('supervisor',supervisor_node)

builder.add_edge(START,'supervisor')

builder.add_conditional_edges(
    'supervisor',
    supervisor_routing,
    {
        "coder":"coder",
        "searcher":"searcher",
        END:END
    }
)

builder.add_edge('coder','supervisor')
builder.add_edge('searcher','supervisor')
memory = MemorySaver()
app = builder.compile(checkpointer=memory)