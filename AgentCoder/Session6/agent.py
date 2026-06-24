import os
import re
import sys
from typing import Annotated, Sequence, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
# Lấy đường dẫn tuyệt đối của thư mục chứa file agent.py hiện tại (Session6)
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
def extract_python_code(text: str) -> str:
    """Trích xuất mã nguồn Python nằm trong khối markdown ```python ... ```.

    Nếu không tìm thấy khối markdown, trả về toàn bộ text gốc làm phương án dự
    phòng.
    """
    # Regex tìm đoạn văn bản nằm giữa ```python và ``` (không phân biệt chữ hoa/thường)
    pattern = r"```(?:python)?\s*(.*?)\s*```"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)

    if matches:
        # Lấy khối code đầu tiên tìm thấy và loại bỏ khoảng trắng thừa
        return matches[0].strip()

    # Phương án dự phòng: Nếu LLM không dùng code block mà trả về thẳng code
    return text.strip()
class TDDState(TypedDict):
    requirements: str  # Yêu cầu bài toán từ người dùng
    test_code: str  # Mã nguồn file test_solution.py
    impl_code: str  # Mã nguồn file solution.py
    test_results: str  # Kết quả chạy test từ Sandbox
    test_passed: bool  # Đã vượt qua tất cả bài test chưa?
    iterations: int  # Số lần Agent tự sửa đổi code
    max_iterations: int  # Giới hạn số lần thử tối đa
    

model = ChatOpenAI(
    base_url="http://localhost:20128/v1",
    api_key="khongco",
    model="kiro",
    temperature=0.3
)



def write_tests_node(state:TDDState)->dict:
    print("\n[Node 1] Đang soạn thảo các ca kiểm thử (Unit Tests)...")
    
    prompt = f"""Bạn là một kỹ sư QA lão luyện. Hãy viết các ca kiểm thử bằng thư viện `pytest` dựa trên yêu cầu dưới đây.
    Đảm bảo bộ test bao gồm cả các trường hợp biên (edge cases).
    YÊU CẦU ĐỊNH DẠNG:
    - Bạn bắt buộc phải bọc toàn bộ mã nguồn của mình trong khối markdown:
    ```python
    # Code của bạn ở đây
    ```
    LƯU Ý QUAN TRỌNG:
    - Bạn phải import hàm hoặc class cần test từ module `solution`. Ví dụ: `from solution import my_function`
    - Chỉ viết code test, không viết code logic thực thi của bài toán.

    Yêu cầu bài toán:
    {state['requirements']}
    """
    response = model.invoke(prompt)
    return {
        'test_code':extract_python_code( response.content),
        'iterations':0
    }
    
def write_implementation_node(state:TDDState)->dict:
    iterations = state.get('iterations',0)+1
    print(
        f"\n[Node 2] Đang viết/sửa code thực thi (Lần lặp thứ: {iterations})..."
    )
    
    if not state.get('test_results'):
        prompt = f"""Bạn là một lập trình viên Python chuyên nghiệp. Hãy viết mã nguồn để đáp ứng yêu cầu bài toán và vượt qua bộ unit test dưới đây.
        YÊU CẦU ĐỊNH DẠNG:
        - Bạn bắt buộc phải bọc toàn bộ mã nguồn của mình trong khối markdown:
        ```python
        # Code của bạn ở đây
        ```
        Yêu cầu bài toán:
        {state['requirements']}

        Bộ Unit Test bạn cần vượt qua:
        {state['test_code']}
        """
    else :
        prompt = f"""Mã nguồn hiện tại của bạn đã bị lỗi khi chạy unit test. Hãy phân tích lỗi bên dưới và sửa lại mã nguồn trong file `solution.py` một cách chính xác nhất.

        Yêu cầu bài toán:
        {state['requirements']}

        Bộ Unit Test cần vượt qua:
        {state['test_code']}

        Lịch sử code đã viết trước đó:
        {state['impl_code']}

        Kết quả kiểm thử bị lỗi (Error Logs):
        {state['test_results']}
        """
    response = model.invoke(prompt)
    return {
        'impl_code':extract_python_code( response.content),
        'iterations':iterations
    }


from sandbox import run_test_in_sandbox

def run_test_node(state:TDDState)->dict:
    print("\n[Node 3] Đang tiến hành chạy thử nghiệm trong Sandbox an toàn...")
    passed,logs = run_test_in_sandbox(state['impl_code'],state['test_code'])

    if passed:
        print("🎉 Tuyệt vời! Tất cả các bài test đã vượt qua thành công.")
    else:
        print("❌ Phát hiện lỗi trong mã nguồn. Cần gửi lại phản hồi để sửa.")
    return {
        'test_passed':passed,
        'test_results':logs
    }
    
def decide_next_step(state:TDDState):
    if state['test_passed']:
        return END
    if state['iterations']>= state['max_iterations']:
        print(
            "\n⚠️ CẢNH BÁO: Đã đạt giới hạn sửa lỗi tối đa nhưng chưa thành công."
        )
        return END
    return 'write_impl'

    
builder = StateGraph(TDDState)
builder.add_node('write_tests',write_tests_node)
builder.add_node("write_impl",write_implementation_node)
builder.add_node('run_tests',run_test_node)

builder.set_entry_point('write_tests')
builder.add_edge('write_tests','write_impl')
builder.add_edge('write_impl','run_tests')

builder.add_conditional_edges(
    'run_tests',
    decide_next_step,
    {
        END:END,
        'write_impl':'write_impl'
    }
)

app = builder.compile()


