import os
import ast
import re
import subprocess
import sys
from typing import Annotated, Sequence, TypedDict, Literal
from pathlib import Path

from langchain_core.messages import BaseMessage, ToolMessage, AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool

# Hạn mức số lần Agent được phép sửa đổi trước khi kích hoạt Autosubmit
MAX_PATCH_ATTEMPTS = 3

class SWEAgentState(TypedDict):
    # Lịch sử tin nhắn giữa Agent và hệ thống
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    # Danh sách các tệp tin mà Agent đã sửa đổi trong phiên làm việc
    modified_files: list[str]
    
    # Bộ lưu trữ nội dung nguyên bản của tệp tin trước khi sửa (để rollback nếu lỗi)
    backup_files: dict[str, str]
    
    # Số lần Agent đã thực hiện vá lỗi (để chặn vòng lặp vô tận)
    patch_attempts: int
    
    # Trạng thái kiểm tra cú pháp tĩnh (AST)
    is_syntax_valid: bool
    
    # Kết quả chạy Unit Test
    test_passed: bool
    test_feedback: str

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

@tool
def view_file_lines(filepath: str, start_line: int, end_line: int) -> str:
    """
    Đọc một phân đoạn dòng cụ thể của một tệp tin. Hãy dùng công cụ này thay vì đọc toàn bộ file.
    Dòng bắt đầu (start_line) và kết thúc (end_line) được đánh chỉ số từ 1.
    """
    path = resolve_file_path(filepath)

    if not path.exists():
        return f"Lỗi: Không tìm thấy tệp tin '{filepath}'."
    
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        total_lines = len(lines)
        
        # Đảm bảo chỉ số nằm trong phạm vi hợp lệ
        start = max(1, start_line) - 1
        end = min(total_lines, end_line)
        
        segment = lines[start:end]
        output = []
        for idx, line in enumerate(segment, start=start + 1):
            output.append(f"{idx:04d} | {line}")
            
        return f"--- Hiển thị {filepath} (Dòng {start+1} đến {end}) ---\n" + "\n".join(output)
    except Exception as e:
        return f"Lỗi khi đọc file: {str(e)}"


@tool
def search_symbol_ast(filepath: str, symbol_name: str) -> str:
    """
    Tìm kiếm vị trí định nghĩa của một Hàm (Function) hoặc Lớp (Class) trong một file Python 
    bằng cách phân tích cây cú pháp AST. Trả về dòng bắt đầu của symbol đó.
    """
    path = resolve_file_path(filepath)
    if not path.exists():
        return f"Lỗi: Không tìm thấy tệp tin '{filepath}'."
        
    try:
        code = path.read_text(encoding="utf-8")
        tree = ast.parse(code)
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                if node.name == symbol_name:
                    return f"Tìm thấy '{symbol_name}' trong '{filepath}' tại dòng {node.lineno}."
                    
        return f"Không tìm thấy định nghĩa của '{symbol_name}' trong '{filepath}'."
    except SyntaxError:
        return f"Lỗi: Tệp tin '{filepath}' hiện đang có lỗi cú pháp, không thể phân tích AST."
    except Exception as e:
        return f"Lỗi hệ thống: {str(e)}"

def resolve_file_path(filepath: str) -> Path:
    """
    Hàm bổ trợ giúp định vị tệp tin một cách bền bỉ (robust).
    Nó tìm kiếm tệp tại 3 vị trí khác nhau để tránh lỗi 'File Not Found'
    do sự khác biệt về môi trường chạy giữa máy cục bộ và Docker/LangGraph Studio.
    """
    path = Path(filepath)
    
    # Nếu đường dẫn truyền vào đã là tuyệt đối, trả về luôn
    if path.is_absolute():
        return path
        
    # Điểm neo 1: Tìm theo thư mục làm việc hiện hành (CWD)
    if path.exists():
        return path.resolve()
        
    # Điểm neo 2: Tìm tương đối so với thư mục chứa chính file python đang chạy (__file__)
    script_dir_path = Path(__file__).resolve().parent / filepath
    if script_dir_path.exists():
        return script_dir_path.resolve()
        
    # Điểm neo 3: Tìm tương đối so với thư mục cha của file python đang chạy (thường là project root)
    parent_dir_path = Path(__file__).resolve().parent.parent / filepath
    if parent_dir_path.exists():
        return parent_dir_path.resolve()
        
    # Nếu không tìm thấy ở bất kỳ đâu, trả về đường dẫn mặc định
    # nhưng chuyển thành tuyệt đối để dễ dàng in ra debug
    return path.absolute()
@tool
def apply_search_replace_patch(filepath: str, patch_block: str) -> str:
    """
    Áp dụng bản vá sửa đổi tối giản vào một tệp tin thông qua khối Tìm kiếm & Thay thế (Search-and-Replace).
    Định dạng bắt buộc của tham số patch_block phải tuân thủ cấu trúc sau:
    <<<<<<< SEARCH
    [Đoạn code cũ cần tìm chính xác trong file]
    =======
    [Đoạn code mới sẽ thay thế]
    >>>>>>> REPLACE
    """
    path = resolve_file_path(filepath)
    if not path.exists():
        return f"Lỗi: Không tìm thấy tệp tin '{filepath}'."
        
    # Phân tích cú pháp khối Search-and-Replace
    pattern = r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE"
    match = re.search(pattern, patch_block, re.DOTALL)
    
    if not match:
        return (
            "Lỗi định dạng bản vá! Bạn phải viết đúng cấu trúc:\n"
            "<<<<<<< SEARCH\n...\n=======\n...\n>>>>>>> REPLACE"
        )
        
    search_code = match.group(1)
    replace_code = match.group(2)
    
    try:
        original_content = path.read_text(encoding="utf-8")
        
        # Kiểm tra xem khối SEARCH có tồn tại chính xác trong file không
        if search_code not in original_content:
            return (
                f"Lỗi: Không tìm thấy đoạn mã cần sửa (SEARCH block) trong '{filepath}'. "
                "Hãy chắc chắn rằng bạn đã copy chính xác từng khoảng trắng và ký tự xuống dòng."
            )
            
        # Thực hiện thay thế
        new_content = original_content.replace(search_code, replace_code, 1)
        path.write_text(new_content, encoding="utf-8")
        
        return f"Thành công: Đã áp dụng bản vá cho '{filepath}'."
    except Exception as e:
        return f"Lỗi khi ghi đè bản vá: {str(e)}"

# Danh sách các công cụ mà LLM có thể gọi trực tiếp
aci_tools = [view_file_lines, search_symbol_ast, apply_search_replace_patch]


def run_tools_and_update_state(state: SWEAgentState) -> dict:
    """
    Node tùy chỉnh thực thi các công cụ được Agent gọi. 
    Node này có nhiệm vụ quan trọng: sao lưu tệp tin trước khi sửa và ghi nhận tệp bị sửa đổi.
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # Khởi tạo các giá trị trả về cho State
    new_messages = []
    modified_files = list(state.get("modified_files", []))
    backup_files = dict(state.get("backup_files", {}))
    patch_attempts = state.get("patch_attempts", 0)
    
    # Bản đồ ánh xạ tên công cụ sang hàm thực thi
    tool_map = {tool.name: tool for tool in aci_tools}
    
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls: # type: ignore
        return {}
        
    for tool_call in last_message.tool_calls: # type: ignore
        tool_name = tool_call["name"]
        args = tool_call["args"]
        tool_id = tool_call["id"]
        
        # Nếu Agent chuẩn bị vá lỗi, thực hiện sao lưu tệp tin trước
        if tool_name == "apply_search_replace_patch":
            filepath = args.get("filepath")
            if filepath and os.path.exists(filepath):
                # Chỉ lưu bản backup của trạng thái ban đầu tiên của tệp tin
                if filepath not in backup_files:
                    backup_files[filepath] = Path(filepath).read_text(encoding="utf-8")
                
                if filepath not in modified_files:
                    modified_files.append(filepath)
            
            # Tăng số lần Agent thực hiện vá lỗi
            patch_attempts += 1
            
        # Thực thi công cụ
        func = tool_map.get(tool_name)
        if func:
            result = func.invoke(args)
        else:
            result = f"Lỗi: Không tìm thấy công cụ '{tool_name}'."
            
        # Tạo ToolMessage trả về cho đồ thị
        new_messages.append(ToolMessage(content=str(result), tool_call_id=tool_id))
        
    return {
        "messages": new_messages,
        "modified_files": modified_files,
        "backup_files": backup_files,
        "patch_attempts": patch_attempts
    }
    
    
model = ChatOpenAI(
    base_url="http://localhost:20128/v1",
    api_key="khongco", # type: ignore
    model="kiro",
    temperature=0.1
).bind_tools(aci_tools)



def agent_node(state: SWEAgentState) -> dict:
    """
    Node đưa ra quyết định suy nghĩ tiếp theo của Agent.
    """
    system_prompt = (
        "Bạn là một Kỹ sư phần mềm AI (SWE-Agent) chuyên nghiệp.\n"
        "Nhiệm vụ của bạn là tìm và sửa lỗi trong mã nguồn theo yêu cầu của người dùng.\n\n"
        "Nguyên tắc làm việc của bạn:\n"
        "1. Hãy thám hiểm mã nguồn bằng cách dùng `view_file_lines` để đọc vùng mã nghi vấn "
        "hoặc `search_symbol_ast` để tìm vị trí định nghĩa hàm.\n"
        "2. Khi muốn sửa đổi, hãy DÙNG DUY NHẤT công cụ `apply_search_replace_patch`.\n"
        "3. Đừng viết lại toàn bộ file. Chỉ sinh khối SEARCH/REPLACE tối giản chứa thay đổi thực tế.\n"
        "4. Nếu bạn nhận được phản hồi lỗi cú pháp hoặc lỗi Unit Test, hãy phân tích kỹ và tiếp tục vá lỗi."
    )
    
    # Định dạng lại lịch sử tin nhắn kèm System Prompt
    messages = [HumanMessage(content=system_prompt)] + list(state["messages"])
    response = model.invoke(messages)
    
    return {"messages": [response]}


def validation_node(state: SWEAgentState) -> dict:
    """
    Kiểm tra cú pháp của tất cả các tệp tin đã bị thay đổi bằng AST parser.
    """
    modified_files = state.get("modified_files", [])
    is_syntax_valid = True
    error_feedback = ""
    
    for filepath in modified_files:
        path = Path(filepath)
        if path.exists():
            try:
                code = path.read_text(encoding="utf-8")
                # Phân tích cú pháp tệp tin để phát hiện lỗi Syntax
                ast.parse(code)
                print('Khong tim thay loi')
            except SyntaxError as e:
                is_syntax_valid = False
                error_feedback += (
                    f"Lỗi cú pháp tại tệp '{filepath}':\n"
                    f"Dòng {e.lineno}, cột {e.offset}: {e.msg}\n"
                    f"Đoạn mã gây lỗi: {e.text}\n"
                )
                break
                
    if not is_syntax_valid:
        # Tạo tin nhắn hệ thống phản hồi lỗi cú pháp cho Agent
        feedback_message = HumanMessage(
            content=f"⚠️ RÀO CHẮN CÚ PHÁP PHÁT HIỆN LỖI:\n{error_feedback}\n"
                    f"Vui lòng sử dụng lại công cụ 'apply_search_replace_patch' để sửa lại lỗi cú pháp này."
        )
        return {
            "is_syntax_valid": False,
            "messages": [feedback_message]
        }
        
    return {"is_syntax_valid": True}

def test_node(state: SWEAgentState) -> dict:
    """
    Thực thi bộ Unit Test bằng cách gọi tiến trình con (subprocess).
    """
    # Nếu cú pháp không hợp lệ từ bước trước, bỏ qua chạy test
    if not state.get("is_syntax_valid", True):
        return {"test_passed": False}
        
    try:
        # Chạy pytest giả lập hoặc pytest thực tế trên tệp test được chỉ định
        # Ở đây chúng ta chạy pytest cho thư mục hiện hành hoặc file test cụ thể
        result = subprocess.run(
            ["pytest", "-q"], 
            capture_output=True, 
            text=True, 
            timeout=15
        )
        
        if result.returncode == 0:
            feedback = "Tất cả các Unit Test đã vượt qua thành công! (Pass 100%)"
            test_passed = True
        else:
            feedback = f"Một số Unit Test đã thất bại:\n{result.stdout}\n{result.stderr}"
            test_passed = False
            
    except Exception as e:
        feedback = f"Lỗi hệ thống khi chạy test suite: {str(e)}"
        test_passed = False
        
    # Gửi phản hồi kết quả test về cho Agent dưới dạng tin nhắn hệ thống
    test_message = HumanMessage(
        content=f"🧪 KẾT QUẢ UNIT TEST:\n{feedback}\n"
                f"{'Hãy tiếp tục hoàn thiện mã nguồn nếu chưa pass hết.' if not test_passed else ''}"
    )
    
    return {
        "test_passed": test_passed,
        "test_feedback": feedback,
        "messages": [test_message]
    }
    
    
def autosubmit_node(state: SWEAgentState) -> dict:
    """
    Node xử lý tình huống khẩn cấp: Khôi phục lại mã nguồn an toàn từ bản sao lưu
    và xuất ra file git diff chứa những thay đổi đã cố gắng thực hiện.
    """
    backup_files = state.get("backup_files", {})
    modified_files = state.get("modified_files", [])
    
    print("🚨 Kích hoạt chế độ Autosubmit: Đang khôi phục hệ thống về trạng thái an toàn...")
    
    # 1. Khôi phục lại nội dung gốc của các file để đảm bảo hệ thống không bị hỏng
    for filepath, original_content in backup_files.items():
        try:
            Path(filepath).write_text(original_content, encoding="utf-8")
        except Exception as e:
            print(f"Lỗi khôi phục tệp {filepath}: {str(e)}")
            
    # 2. Tạo một báo cáo tóm tắt
    report_message = AIMessage(
        content=(
            "❌ Hệ thống tự động kích hoạt cơ chế rút lui an toàn (Autosubmit Pattern).\n"
            f"Lý do: Đã vượt quá số lần vá lỗi cho phép ({MAX_PATCH_ATTEMPTS} lần) nhưng vẫn chưa pass Unit Test.\n"
            "Chúng tôi đã khôi phục lại toàn bộ mã nguồn về trạng thái gốc để tránh gây hư hỏng hệ thống.\n"
            "Dưới đây là file gốc đã được khôi phục. Kỹ sư con người vui lòng kiểm duyệt thủ công."
        )
    )
    
    return {
        "messages": [report_message],
        "test_passed": False
    }
    
def route_after_agent(state: SWEAgentState) -> Literal["tools", "__end__"]:
    """
    Quyết định đi tiếp tới thực thi công cụ hay kết thúc dựa trên việc Agent có gọi công cụ hay không.
    """
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls: # type: ignore
        return "tools"
    return "__end__"


def route_after_tools(state: SWEAgentState) -> Literal["validation", "autosubmit", "agent"]:
    """
    Hàm rẽ nhánh thông minh sau khi chạy công cụ:
    1. Nếu vượt quá giới hạn vá lỗi -> "autosubmit".
    2. Nếu Agent vừa thực hiện sửa đổi file (apply_search_replace_patch) -> "validation".
    3. Nếu Agent chỉ đọc/tìm kiếm file (view_file_lines, search_symbol_ast) -> "agent" để suy nghĩ tiếp.
    """
    patch_attempts = state.get("patch_attempts", 0)
    if patch_attempts >= MAX_PATCH_ATTEMPTS:
        return "autosubmit"
        
    # Duyệt ngược lịch sử tin nhắn để tìm tin nhắn gọi công cụ gần nhất của Agent
    last_ai_message = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage):
            last_ai_message = msg
            break
            
    if last_ai_message and hasattr(last_ai_message, "tool_calls"):
        # Kiểm tra xem trong các công cụ được gọi, có công cụ sửa đổi nào không
        has_patch_call = any(
            tc["name"] == "apply_search_replace_patch" 
            for tc in last_ai_message.tool_calls
        )
        if has_patch_call:
            # Nếu có sửa đổi file, bắt buộc phải đi qua Validation -> Test
            return "validation"
            
    # Nếu Agent chỉ gọi công cụ đọc file hoặc tìm kiếm, quay lại Node Agent ngay lập tức
    return "agent"


def route_after_validation(state: SWEAgentState) -> Literal["test", "agent"]:
    """
    Nếu cú pháp không hợp lệ, lập tức quay lại Agent để yêu cầu sửa code mà không chạy test.
    """
    if not state.get("is_syntax_valid", True):
        return "agent"
    return "test"


def route_after_test(state: SWEAgentState) -> Literal["agent", "__end__"]:
    """
    Nếu vượt qua tất cả Unit Test thì kết thúc luồng chạy.
    Nếu thất bại, quay về Agent để phân tích và tiếp tục vòng lặp sửa lỗi.
    """
    if state.get("test_passed", False):
        return "__end__"
    return "agent"

from langgraph.checkpoint.memory import MemorySaver

# Khởi tạo đồ thị
builder = StateGraph(SWEAgentState)

# Thêm các Node vào đồ thị
builder.add_node("agent", agent_node)
builder.add_node("tools", run_tools_and_update_state)
builder.add_node("validation", validation_node)
builder.add_node("test", test_node)
builder.add_node("autosubmit", autosubmit_node)

# Thiết lập các kết nối tĩnh (Edges) và kết nối động (Conditional Edges)
builder.add_edge(START, "agent")

builder.add_conditional_edges(
    "agent",
    route_after_agent,
    {
        "tools": "tools",
        "__end__": END
    }
)

# Cập nhật ánh xạ conditional_edges của node tools
builder.add_conditional_edges(
    "tools",
    route_after_tools,
    {
        "validation": "validation",
        "autosubmit": "autosubmit",
        "agent": "agent"  # Thêm đường dẫn quay lại Agent cho các tác vụ Read-only
    }
)

builder.add_conditional_edges(
    "validation",
    route_after_validation,
    {
        "test": "test",
        "agent": "agent"
    }
)

builder.add_conditional_edges(
    "test",
    route_after_test,
    {
        "agent": "agent",
        "__end__": END
    }
)

builder.add_edge("autosubmit", END)

# Tích hợp bộ nhớ để giám sát và debug qua LangSmith
memory = MemorySaver()
app = builder.compile(checkpointer=memory)