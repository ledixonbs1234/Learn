import os
import re
import subprocess
import sys
from typing import Annotated, Sequence, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

class MultiAgentState(TypedDict):
    messages:Annotated[Sequence[BaseMessage],add_messages]
    code:str
    feedback:str
    approved:bool
    review_count:int
    error:str
    iteration:int

    
    
model = ChatOpenAI(
    base_url="http://localhost:20128/v1",
    api_key="khongco",
    model="kiro",
    temperature=0.3
)

# Hàm trích xuất code Python từ markdown
def extract_code(text: str) -> str:
    match = re.search(r"```python\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1)
    return text

def programmer_code(state:MultiAgentState):
    print("\n--- 💻 [NODE: PROGRAMMER] ĐANG VIẾT/SỬA CODE ---")
    messages = list(state["messages"])
    feedback = state["feedback"]
    if feedback:
        revision_prompt = (
            f"Code của bạn đã bị từ chối bởi Reviewer với phản hồi sau:\n"
            f"----------------------------------------\n"
            f"{feedback}\n"
            f"----------------------------------------\n"
            f"Hãy sửa đổi, tối ưu hóa và viết lại mã nguồn hoàn chỉnh theo các ý kiến trên.\n"
            f"Chỉ trả về code Python nằm trong khối ```python ... ```."
        )
        messages.append(HumanMessage(content=revision_prompt))

    instruction = (
            "Bạn là một lập trình viên Python xuất sắc. Hãy viết code hoàn chỉnh "
            "đáp ứng yêu cầu của người dùng. Hãy đặt tên biến rõ ràng, viết code "
            "sạch sẽ và chỉ trả về mã nguồn trong khối ```python ... ```."
        )
    system = SystemMessage(content=instruction)
    response = model.invoke([system]+messages)
    code = extract_code(response.content)

    return {"messages":[response],"code":code}
    

def executor_node (state :MultiAgentState):
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
            return MultiAgentState(iteration=iterations+1,error="")   
        else:
            print("❌ PHÁT HIỆN LỖI RUNTIME!")
            print("Chi tiết lỗi:\n", result.stderr)
            return MultiAgentState(iteration=iterations+1,error=result.stderr)   
    except subprocess.TimeoutExpired:
        print("❌ LỖI: CODE CHẠY QUÁ THỜI GIAN CHO PHÉP (TIMEOUT)!")
        return MultiAgentState(
            error="Lỗi: Quá thời gian thực thi (Timeout - 5 giây). Có thể code bị lặp vô hạn.",
            iteration=iterations+1
        )
    except Exception as e:
        print("❌ LỖI HỆ THỐNG KHI THỰC THI!")
        return MultiAgentState(
            error= str(e),
            iteration=iterations+1
        )
    
def reviewer_node(state:MultiAgentState):
    print("\n--- 👁️ [NODE: REVIEWER] ĐANG KIỂM DUYỆT CHẤT LƯỢNG CODE ---")
    code = state.get("code", "")
    review_count = state.get("review_count", 0)

    review_prompt = (
        f"Bạn là một Senior Code Reviewer chuyên nghiệp. Hãy đánh giá kỹ đoạn code sau đây:\n\n"
        f"```python\n{code}\n```\n\n"
        f"Tiêu chí kiểm duyệt:\n"
        f"1. Đúng logic yêu cầu của người dùng chưa?\n"
        f"2. Có tối ưu hiệu năng không (ví dụ: tránh dùng vòng lặp thừa, tránh thuật toán quá chậm)?\n"
        f"3. Code có sạch sẽ, đặt tên biến rõ ràng không?\n\n"
        f"BẮT BUỘC TUÂN THỦ QUY TẮC PHẢN HỒI:\n"
        f"- Nếu code đạt yêu cầu: Ghi rõ từ khóa [APPROVED] ở dòng đầu tiên, sau đó nhận xét ngắn gọn.\n"
        f"- Nếu code CHƯA đạt yêu cầu: Ghi rõ từ khóa [REJECTED] ở dòng đầu tiên, sau đó đưa ra feedback chi tiết và hướng dẫn cụ thể để sửa lỗi."
    )
    
    response = model.invoke([HumanMessage(content=review_prompt)])
    review_text = response.content
    
    approved = "[APPROVED]" in review_text
    feedback = "" if approved else review_text
    print("-------------------- REVIEW REPORT --------------------")
    print(review_text)
    print("-------------------------------------------------------")

    return {
        "messages":[response],
        "feedback":feedback,
        "approved":approved,
        "review_count":review_count+1
    }
 
def decide_next_step(state:MultiAgentState):
    approved = state.get('approved',False)
    review_count =state.get('review_count',0)   

    if approved:
        print("-> Code đã được Reviewer duyệt. Hoàn thành luồng!")
        return 'execute'
    if review_count >=3:
        print("-> Đã đạt giới hạn 3 lần kiểm duyệt. Kết thúc luồng.")
        return END
    print(f"-> Code chưa đạt yêu cầu (Lần đánh giá: {review_count}/3). Gửi lại cho Programmer sửa đổi...")
    return "programmer"

def decide_error_next_step(state:MultiAgentState):
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

graph = StateGraph(MultiAgentState)
graph.add_node('programmer',programmer_code)
graph.add_node('reviewer',reviewer_node)
graph.add_node('execute',executor_node)

graph.add_edge(START,'programmer')
graph.add_edge('programmer','reviewer')

graph.add_conditional_edges(
    'reviewer',
    decide_next_step,
    {
        END:END,
        'programmer':'programmer',
        'execute':'execute'
    }
)
graph.add_conditional_edges(
    'execute',
    decide_error_next_step,
    {
        'programmer':'programmer',
        END:END
    }
)
memory = MemorySaver()
app = graph.compile(checkpointer=memory)
try:
    # Xuất đồ thị thành định dạng ảnh PNG
    png_bytes = app.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(png_bytes)
    print("Đã vẽ sơ đồ thành công! Bạn hãy mở file 'graph.png' trong thư mục để xem.")
except Exception as e:
    print(f"Không thể xuất ảnh PNG: {e}")
def main():
    
    # Chúng ta đưa ra một yêu cầu viết code nhưng có thể giải quyết bằng 2 cách:
    # 1. Cách thô sơ, chậm chạp (ví dụ: dùng nested loops O(N^2)).
    # 2. Cách tối ưu hơn (dùng Set hoặc Dictionary O(N)).
    # Hãy xem Reviewer có phát hiện ra cách viết chưa tối ưu của Programmer và yêu cầu sửa không.
    user_query = (
        "Viết một hàm Python tìm các phần tử trùng nhau giữa hai danh sách A và B. "
        "Hãy viết thuật toán sao cho chạy nhanh nhất có thể khi danh sách có hàng triệu phần tử."
    )
    print(f"Yêu cầu từ người dùng: '{user_query}'")
    initial_state = {
        "messages": [HumanMessage(content=user_query)],
        "review_count": 0,
        "approved": False,
        "code": "",
        "feedback": ""
    }
    config = {"configurable":{'thread_id':'multi_agent_session'}}
    for event in app.stream(initial_state,config=config, stream_mode="values"):
        pass
    final_state = app.get_state({"configurable": {"thread_id": "multi_agent_session"}})
    print("\n================ MÃ NGUỒN CUỐI CÙNG SAU KHI ĐƯỢC DUYỆT ================")
    print(final_state.values.get("code"))
    print("=========================================================================")

if __name__ == "__main__":
    main()
    


    