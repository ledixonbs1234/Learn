import re
import subprocess
import sys
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

class CoderState(TypedDict):
    messages: Annotated[Sequence[BaseMessage],add_messages]
    code:str
    error:str
    iteration:int


model = ChatOpenAI(
    base_url="http://localhost:20128/v1",
    api_key="khongco",
    model="kiro",
    temperature=0.3
)

def extract_code(text:str)->str:
    """Phân giải code tìm cấu trúc python"""
    match = re.search(r"```python\n(.*?)\n```",text,re.DOTALL)
    if match:
        return match.group(1)
    return text

def programmer_node(state:CoderState):
    """AI is creating summary for programmer_node

    Args:
        state (CoderState): [description]
    """
    print("\n--- 💻 NODE: PROGRAMMER ĐANG VIẾT CODE ---")
    messages = list(state["messages"])
    error = state.get("error","")
    iterations =state.get("iteration",0)


    if error:
        feedback_prompt = (
            f"Code bạn vừa viết đã xảy ra lỗi Runtime sau đây khi chạy thử:\n"
            f"----------------------------------------\n"
            f"{error}\n"
            f"----------------------------------------\n"
            f"Hãy phân tích nguyên nhân gây lỗi, sau đó viết lại mã nguồn hoàn chỉnh đã sửa lỗi.\n"
            f"Lưu ý: Chỉ trả về mã Python hoàn chỉnh nằm trong khối ```python ... ```."
         )
        messages.append(HumanMessage(content=feedback_prompt))
    else:
        system_intruction= (
             "Bạn là một lập trình viên Python chuyên nghiệp. Hãy viết mã nguồn Python hoàn chỉnh "
            "để giải quyết yêu cầu của người dùng. Viết toàn bộ code trong khối ```python ... ```. "
            "Không viết thêm lời giải thích rườm rà ngoài code."
        )
        messages.insert(0,SystemMessage(content=system_intruction))
    
    
    
    print(f"Toàn bộ code hiện tại ")
    for message in messages:
        print(f"\n {message.content}")
        
    print("-------------------------------------")
    response = model.invoke(messages)
    code = extract_code(response.content)

    return{
        "messages":[response],
        "code":code
    }

def executor_node (state :CoderState):
    print("\n--- ⚙️ NODE: EXECUTOR ĐANG CHẠY THỬ CODE ---")
    code = state.get('code')
    iterations = state.get('iteration',0)

    if not code:
        return {"error": "Không tìm thấy mã nguồn để thực thi.", "iterations": iterations + 1}
    print("Đang thực thi đoạn code sau:")
    print("----------------------------------------")
    print(code)
    print("----------------------------------------")

    try:
        
        # 1. Tạo bản sao môi trường hệ thống hiện tại
        env = os.environ.copy()
        
        # 2. Ép buộc tiến trình con Python chạy ở chế độ UTF-8 (tương thích Python 3.7+)
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(
            [sys.executable,'-c',code],
            capture_output=True,
            text=True,
            timeout=5,
            encoding='utf-8',
            env=env
        )

        if result.returncode == 0:
            print("✅ CHẠY CODE THÀNH CÔNG!")
            print("Output nhận được:\n", result.stdout)
            return CoderState(iteration=iterations+1,error="")   
        else:
            print("❌ PHÁT HIỆN LỖI RUNTIME!")
            print("Chi tiết lỗi:\n", result.stderr)
            return CoderState(iteration=iterations+1,error=result.stderr)   
    except subprocess.TimeoutExpired:
        print("❌ LỖI: CODE CHẠY QUÁ THỜI GIAN CHO PHÉP (TIMEOUT)!")
        return CoderState(
            error="Lỗi: Quá thời gian thực thi (Timeout - 5 giây). Có thể code bị lặp vô hạn.",
            iteration=iterations+1
        )
    except Exception as e:
        print("❌ LỖI HỆ THỐNG KHI THỰC THI!")
        return CoderState(
            error= str(e),
            iteration=iterations+1
        )
    
def decide_next_step(state:CoderState):
    error = state.get('error','')
    iterations = state.get('iteration',0)
    if error == '':
        print("-> Đã hoàn thành code chạy ổn định.")
        return END
    if iterations >=3:
        print("-> Đã đạt giới hạn tối đa 3 lần sửa lỗi. Dừng lại.")
        return END
    
    print(f"-> Code bị lỗi. Chuyển lại cho Programmer sửa đổi (Lần thử {iterations}/3)...")
    return "programmer" 
    

builder = StateGraph(CoderState)
builder.add_node('programmer',programmer_node)
builder.add_node('executor',executor_node)
builder.add_edge(START,'programmer')
builder.add_edge('programmer',"executor")

builder.add_conditional_edges('executor',
                            decide_next_step,
                            {
                                END:END,
                                "programmer":"programmer"
                            })
memory = MemorySaver()
app = builder.compile(checkpointer=memory)


def main():
    user_query = "Viết code Python tạo một danh sách chứa 5 số ngẫu nhiên từ 1 đến 100, sắp xếp chúng giảm dần, tính tổng và in ra màn hình cả danh sách và tổng."

    print(f'Yêu cầu từ người dùng: {user_query}')

    initstate = {"messages":[HumanMessage(content=user_query)],"iteration":0,"code":'',"error":''}
    config = {"configurable":{"thread_id":"dummy"}}

    for event in app.stream(initstate,config=config,stream_mode='values'):
        pass


    final_state = app.get_state({"configurable": {"thread_id": "dummy"}}) # Lấy trạng thái hiện tại
    print("\n================ KẾT QUẢ CUỐI CÙNG ================")
    print("Mã nguồn hoàn chỉnh cuối cùng:\n")
    print(final_state.values.get("code"))
    print("====================================================")

if __name__ == "__main__":
    main()   


