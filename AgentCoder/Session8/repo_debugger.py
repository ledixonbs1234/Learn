import os
import re
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import subprocess
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from patch_tool import apply_patch_to_file

    
    

def clean_code_block(code:str)->str:
    """Loại bỏ các dòng trống thừa do thẻ XML tạo ra nhưng giữ nguyên indentation của code."""
    if not code:
        return ""
    return code.strip("\r\n")

def parse_xml_patch(response_text:str)->dict:
    """Sử dụng Regex để bóc tách thông tin từ các thẻ XML do LLM trả về."""
    file_path_match = re.search(
        r"<file_path>(.*?)</file_path>", response_text, re.DOTALL
    )
    search_match = re.search(
        r"<search_block>(.*?)</search_block>", response_text, re.DOTALL
    )
    replace_match = re.search(
        r"<replace_block>(.*?)</replace_block>", response_text, re.DOTALL
    )

    return {
        "file_path": (
            file_path_match.group(1).strip() if file_path_match else ""
        ),
        "search_block": (
            clean_code_block(search_match.group(1)) if search_match else ""
        ),
        "replace_block": (
            clean_code_block(replace_match.group(1)) if replace_match else ""
        ),
    }
    
class DebuggerState(TypedDict):
    bug_description:str
    file_to_debug:str
    test_file:str
    code_content: str
    test_content: str
    test_result:str
    patch:dict
    test_passed:bool
    iterations:int
    
    
model = ChatOpenAI(
    base_url="http://localhost:20128/v1",
    api_key="khongco",
    model="kiro",
    temperature=0.1
)

def read_file_node(state:DebuggerState)->dict:
    print("\n[Node 1] Đang đọc nội dung mã nguồn và file kiểm thử...")
    
    file_to_debug = state["file_to_debug"]
    test_file = state["test_file"]

    # Nếu đường dẫn truyền vào là tương đối (ví dụ: "calculator.py"),
    # chúng ta tự động ghép nối nó với thư mục hiện hành chứa file `repo_debugger.py` (current_dir)
    if not os.path.isabs(file_to_debug):
        file_to_debug = os.path.join(current_dir, file_to_debug)
    if not os.path.isabs(test_file):
        test_file = os.path.join(current_dir, test_file)
    
    with open(file_to_debug,'r',encoding='utf-8') as f:
        code_content = f.read()
    with open(test_file,'r',encoding='utf-8') as f:
        test_content = f.read()
    return {
        "file_to_debug": file_to_debug,
        "test_file": test_file,
        "code_content":code_content,
        "test_content":test_content,
        "iterations":0
    }
    
def run_tests_node(state:DebuggerState):
    print("\n[Node 2] Đang thực thi Unit Test để kiểm tra...")
    result = subprocess.run(
        [sys.executable,'-m','pytest',state['test_file']],
        capture_output=True,
        text=True,
        timeout=5
    )    
    passed = result.returncode == 0
    logs = result.stdout +'\n' +result.stderr
    return {
        "test_passed":passed,
        "test_result":logs
    }
    
def propose_patch_node(state:DebuggerState):
    print("\n[Node 3] Agent đang phân tích lỗi và đề xuất bản vá...")
    prompt = f"""Bạn là một Chuyên gia gỡ lỗi (Senior Debugger). 
    Nhiệm vụ của bạn là sửa lỗi trong file `{state['file_to_debug']}` để vượt qua bài kiểm thử trong file `{state['test_file']}`.

    Mô tả lỗi:
    {state['bug_description']}

    Mã nguồn hiện tại của `{state['file_to_debug']}`:
    {state['code_content']}

    Nội dung file test `{state['test_file']}`:
    {state['test_content']}

    Log lỗi chi tiết từ lần chạy thử gần nhất:
    {state.get('test_result', 'Chưa chạy test lần nào.')}

    ---
    YÊU CẦU ĐẦU RA:
    Bạn phải trả về phản hồi theo định dạng XML chính xác như sau. KHÔNG viết thêm bất kỳ lời giải thích nào khác ngoài cấu trúc này:

    <file_path>{state['file_to_debug']}</file_path>
    <search_block>
    [Đoạn code cũ CHÍNH XÁC hiện tại trong file cần được thay thế, giữ nguyên khoảng trắng thụt dòng]
    </search_block>
    <replace_block>
    [Đoạn code mới sẽ thay thế cho đoạn code cũ, giữ nguyên khoảng trắng thụt dòng]
    </replace_block>
    """
    
    response = model.invoke(prompt)

    parsed_patch = parse_xml_patch(response.content)

    print(
        f"-> Agent đề xuất thay thế đoạn code trong: {parsed_patch['file_path']}"
    )
    print(f"--- CODE CŨ KHỚP ĐƯỢC ---\n{parsed_patch['search_block']}")
    print(f"--- CODE MỚI THAY THẾ ---\n{parsed_patch['replace_block']}")

    return {"patch":parsed_patch}

def apply_patch_node(state:DebuggerState):
    print("\n[Node 4] Đang áp dụng bản vá vào tệp tin thực tế...")
    patch=state['patch']

    if not patch['search_block'] or not patch['replace_block']:
        print(
            "❌ Lỗi: Agent không xuất ra đúng định dạng XML yêu cầu. Đang bỏ qua lượt vá này."
        )
        return {"iterations": state["iterations"] + 1}
    
    result = apply_patch_to_file(patch['file_path'],patch['search_block'],patch['replace_block'])
    print(f"-> Kết quả áp dụng: {result}")
    with open(state['file_to_debug'],'r',encoding='utf-8') as f:
        updated_code = f.read()

    return {
        'code_content':updated_code,
        'iterations':state['iterations']+1
    }

def check_test_status(state: DebuggerState):
    if state["test_passed"]:
        print(
            "\n🎉 Chúc mừng! Lỗi đã được sửa thành công và vượt qua toàn bộ Unit Tests."
        )
        return END

    if state["iterations"] >= 3:
        print(
            "\n⚠️ Đã đạt giới hạn sửa lỗi tối đa (3 lần) nhưng vẫn thất bại."
        )
        return END

    return "propose_patch"

builder = StateGraph(DebuggerState)
builder.add_node("read_files",read_file_node)
builder.add_node("run_tests",run_tests_node)
builder.add_node("propose_patch",propose_patch_node)
builder.add_node('apply_patch',apply_patch_node)

builder.add_edge(START,'read_files')
builder.add_edge('read_files','run_tests')

builder.add_conditional_edges(
    'run_tests',
    check_test_status,
    {
        END:END,
        'propose_patch':'propose_patch'
    }
)

builder.add_edge('propose_patch','apply_patch')
builder.add_edge('apply_patch','run_tests')
app = builder.compile()

# initial_state = {
#         "bug_description": "Hàm divide() bị crash khi b = 0. File test yêu cầu khi b = 0 thì hàm trả về 0.0",
#         "file_to_debug": "calculator.py",
#         "test_file": "test_calculator.py",
#         "patch": {},
#         "iterations": 0,
#     }
# print("--- KHỞI CHẠY REPO DEBUGGER AGENT (XML PARSER MODE) ---")
# final_state = app.invoke(initial_state)