import json
import operator
import re
from typing import Annotated, List, TypedDict
from pydantic  import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send 


def clean_and_parse_json(text: str) -> list | dict:
    """Trích xuất khối JSON từ phản hồi văn bản thô của LLM."""
    # Tìm khối markdown ```json ... ``` hoặc ``` ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        json_str = match.group(1).strip()
    else:
        # Nếu không tìm thấy markdown block, tìm ngoặc nhọn hoặc ngoặc vuông ngoài cùng
        match_raw = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
        json_str = match_raw.group(1).strip() if match_raw else text.strip()

    # Chuyển đổi chuỗi thành kiểu dữ liệu Python (list hoặc dict)
    return json.loads(json_str)


def clean_python_code(text: str) -> str:
    """Trích xuất mã nguồn Python sạch từ khối markdown ```python ... ```."""
    match = re.search(r"```(?:python)?\s*([\s\S]*?)\s*```", text)
    if match:
        return match.group(1).strip()
    return text.strip()


class EndpointTask(TypedDict):
    path:str = Field(description="Đường dẫn API, ví dụ /items, items/{id}")
    method:str = Field(description= "HTTP Method ví dụ: GET, POST, DELETE")
    description:str= Field(description= "Mô tả logic nghiệp vụ của endpoint này")
    
    
class PlanOutput(TypedDict):
    endpoints:List[EndpointTask]=Field(
        description="Danh sách các API Endpoints cần xây dựng"
    )
    
class EndpointCode(TypedDict):
    path:str
    method:str
    code:str = Field(
        description="Đoạn code Python của endpoint này  (chỉ định nghĩa hàm, không cần khởi tạo FastAPI app)"
    )

# 1. State cho từng luồng chạy song song (Sub-State)
# Mỗi Agent viết code chỉ cần biết thông tin nhiệm vụ cụ thể của nó
class CoderState(TypedDict):
    task:EndpointTask

# 2. State chính của toàn bộ Graph (Main-State)
class ProjectState(TypedDict):
    requirements:str
    tasks:List[EndpointTask]

    # Sử dụng operator.add để khi các luồng song song trả về kết quả,
    # chúng tự động được append (gộp thêm) vào list generated_codes thay vì ghi đè.
    generated_codes: Annotated[List[EndpointCode],operator.add]

    final_code:str
    
model = ChatOpenAI(
    base_url="http://localhost:20128/v1",
    api_key="khongco",
    model="kiro",
    temperature=0.1
)

def planner_node(state:ProjectState):
    print("\n[Node Planner] Đang phân tích yêu cầu và lập kế hoạch API...")
    prompt = f"""Bạn là một Kiến trúc sư Hệ thống Backend. Hãy phân tích yêu cầu dưới đây và thiết kế danh sách các API Endpoints (FastAPI) cần thiết.

    Yêu cầu hệ thống:
    {state['requirements']}

    YÊU CẦU ĐỊNH DẠNG ĐẦU RA:
    Bạn phải trả về duy nhất một chuỗi JSON dạng mảng các đối tượng (array of objects), KHÔNG giải thích thêm, KHÔNG viết lời dẫn.

    Định dạng mẫu bắt buộc:
    ```json
    [
    {{
        "path": "/tasks",
        "method": "GET",
        "description": "Lấy danh sách các công việc"
    }},
    {{
        "path": "/tasks",
        "method": "POST",
        "description": "Tạo một công việc mới"
    }}
    ]
    ```
    """
    response = model.invoke(prompt)

    tasks = clean_and_parse_json(response.content)
    print(f"-> Đã lập kế hoạch xong. Cần xây dựng {len(tasks)} endpoints.")
    for idx, ep in enumerate(tasks, 1):
        print(f"   {idx}. [{ep['method']}] {ep['path']} - {ep['description']}")
    
    return {"tasks":tasks}

def coder_node(state: CoderState):
    task = state['task']
    print(
        f"\n[Node Coder] Đang sinh code song song cho: [{task['method']}] {task['path']}..."
    )
    
    prompt = f"""Bạn là một lập trình viên FastAPI chuyên nghiệp.
    Hãy viết mã nguồn Python cho API Endpoint duy nhất dưới đây:

    Thông tin endpoint cần sinh:
    - Path: {task['path']}
    - Method: {task['method']}
    - Mô tả: {task['description']}

    YÊU CẦU:
    - Chỉ định nghĩa hàm xử lý (route function) kèm decorator thích hợp (ví dụ: @app.{task['method'].lower()}("{task['path']}")).
    - Hãy viết mã nguồn Python hoàn chỉnh và bọc nó trong khối markdown ```python ... ```. Không viết thêm bất kỳ lời giải thích nào khác.
    """

    response = model.invoke(prompt)
    # Làm sạch mã nguồn Python thu được
    cleaned_code = clean_python_code(response.content)
    
    result_code: EndpointCode = {
        "path": task["path"],
        "method": task["method"],
        "code": cleaned_code,
    }
    return {"generated_codes":[result_code]}

def aggreator_node(state: ProjectState):
    print(
        f"\n[Node Aggregator] Đã nhận đủ {len(state['generated_codes'])} kết quả song song. Đang gộp mã nguồn..."
    )
    raw_endpoints_code = ""
    for item in state["generated_codes"]:
        raw_endpoints_code += (
            f"\n# Endpoint: {item['method']} {item['path']}\n{item['code']}\n"
        )
    prompt = f"""Bạn là một Kỹ sư Tích hợp Code. Bạn nhận được danh sách các hàm endpoint FastAPI rời rạc dưới đây.
Nhiệm vụ của bạn là tích hợp chúng thành một tệp tin `main.py` hoàn chỉnh, mạch lạc và chạy được.

Yêu cầu:
1. Thêm đầy đủ các thư viện import cần thiết (ví dụ: fastapi, BaseModel từ pydantic, Optional, v.v.).
2. Khởi tạo ứng dụng FastAPI: `app = FastAPI()`.
3. Định nghĩa các Pydantic model (DTOs) cần thiết để các hàm hoạt động không bị lỗi thiếu Class.
4. Đảm bảo toàn bộ logic của các endpoint dưới đây được tích hợp đầy đủ và không bị sửa đổi nghiệp vụ.

Các endpoint rời rạc:
{raw_endpoints_code}
"""

    response = model.invoke(prompt)
    final_code_clean = clean_python_code(response.content)

    return {"final_code":final_code_clean}


def map_to_coders(state:ProjectState):
    # Với mỗi task trong danh sách, chúng ta tạo ra một lệnh Send
    # Send("tên_node_đích", "state_con_truyền_vào")
    sends = [Send('coder',{"task":task}) for task in state['tasks']]
    return sends

builder = StateGraph(ProjectState)

builder.add_node('planner',planner_node)
builder.add_node('coder',coder_node)
builder.add_node('aggregator',aggreator_node)

builder.add_edge(START,'planner')

builder.add_conditional_edges('planner',map_to_coders,['coder'])

builder.add_edge('coder','aggregator')
builder.add_edge('aggregator',END)


app = builder.compile()