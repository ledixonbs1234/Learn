import json
import re
import subprocess
from typing import TypedDict, List, Dict, Any
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import tempfile

from langgraph.graph import END, START, StateGraph

class Task(TypedDict):
    id: int
    task: str
    status: str  # Các trạng thái: "pending", "completed", "failed"
    result: str  # Lưu kết quả/lỗi sau khi thực hiện

class PlanState(TypedDict):
    input_request: str     # Yêu cầu ban đầu từ người dùng
    plan: List[Task]       # Danh sách các bước trong kế hoạch
    current_task_id: int   # ID của tác vụ đang xử lý
    latest_output: str     # Kết quả đầu ra gần nhất của Executor
    error_count: int       # Số lần lỗi tích lũy để tránh lặp vô hạn

def robust_json_parser(text: str) -> Any:
    """
    Trích xuất và parse dữ liệu JSON từ phản hồi dạng văn bản của LLM.
    Hỗ trợ bóc tách khối ```json ... ```, các khối ``` thông thường,
    hoặc tìm kiếm mảng/đối tượng JSON trực tiếp bằng vị trí dấu ngoặc.
    """
    text_clean = text.strip()
    
    # Trường hợp 1: Trích xuất từ cặp thẻ ```json ... ```
    json_block_pattern = r"```json\s*(.*?)\s*```"
    match = re.search(json_block_pattern, text_clean, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Trường hợp 2: Trích xuất từ cặp thẻ ``` ... ``` thông thường
    generic_block_pattern = r"```\s*(.*?)\s*```"
    match_generic = re.search(generic_block_pattern, text_clean, re.DOTALL)
    if match_generic:
        try:
            return json.loads(match_generic.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Trường hợp 3: Tìm kiếm vị trí xuất hiện của [ hoặc { đầu tiên và cuối cùng
    # (Hữu ích khi LLM viết lời dẫn rồi mới viết JSON tự do)
    start_bracket = text_clean.find('[')
    start_brace = text_clean.find('{')
    
    # Xác định điểm bắt đầu của JSON
    start_idx = -1
    if start_bracket != -1 and start_brace != -1:
        start_idx = min(start_bracket, start_brace)
    elif start_bracket != -1:
        start_idx = start_bracket
    elif start_brace != -1:
        start_idx = start_brace
        
    # Xác định điểm kết thúc của JSON
    end_bracket = text_clean.rfind(']')
    end_brace = text_clean.rfind('}')
    end_idx = max(end_bracket, end_brace)
    
    if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
        candidate = text_clean[start_idx:end_idx + 1].strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Trường hợp cuối: Thử parse toàn bộ văn bản
    try:
        return json.loads(text_clean)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Không thể phân tách cấu trúc JSON từ phản hồi của LLM.\n"
            f"Phản hồi thực tế: {text}\nLỗi chi tiết: {str(e)}"
        )
        
model = ChatOpenAI(
    base_url="http://localhost:20128/v1",
    api_key="khongco",
    model="kiro",
    temperature=0.1
)
PLANNER_SYSTEM_PROMPT = """Bạn là một Kiến trúc sư Phần mềm AI cấp cao.
Nhiệm vụ của bạn là nhận yêu cầu lập trình từ người dùng và phân rã nó thành một danh sách các bước thực hiện chi tiết (Checklist).

BẠN BẮT BUỘC PHẢI TRẢ VỀ kết quả dưới dạng một danh sách JSON nằm trong khối mã ```json ... ```. 
KHÔNG giải thích gì thêm ngoài khối mã JSON này.

Định dạng JSON yêu cầu:
```json
[
  {"id": 1, "task": "Mô tả chi tiết tác vụ 1", "status": "pending", "result": ""},
  {"id": 2, "task": "Mô tả chi tiết tác vụ 2", "status": "pending", "result": ""}
]
```

Lưu ý quan trọng:
- Chia nhỏ nhiệm vụ hợp lý (thường từ 3-5 bước).
- Các bước phải có tính tuần tự, bước sau kế thừa kết quả bước trước.
- Giữ trường 'status' mặc định là 'pending' và 'result' là chuỗi rỗng "".
"""

# 2. Prompt dành cho REPLANNER
REPLANNER_SYSTEM_PROMPT = """Bạn là Giám đốc Dự án AI (Project Manager).
Nhiệm vụ của bạn là xem xét kế hoạch hiện tại (Plan) và kết quả thực thi gần nhất của Executor đối với tác vụ hiện tại, sau đó cập nhật lại trạng thái của kế hoạch.

BẠN BẮT BUỘC PHẢI TRẢ VỀ kết quả dưới dạng danh sách JSON đã được cập nhật nằm trong khối mã ```json ... ```.
KHÔNG giải thích gì thêm ngoài khối mã JSON này.

Quy tắc cập nhật:
1. Đánh giá kết quả của tác vụ vừa thực thi (ID: {current_task_id}):
   - Nếu thành công: Cập nhật 'status' thành 'completed' và ghi tóm tắt kết quả vào 'result'.
   - Nếu thất bại: Cập nhật 'status' thành 'failed' và ghi chi tiết lỗi vào 'result'.
2. Bạn có quyền sửa đổi các tác vụ tiếp theo (thêm tác vụ gỡ lỗi, sửa đổi tác vụ, hoặc giữ nguyên cấu trúc nếu mọi thứ đang đi đúng hướng).
3. Đảm bảo toàn bộ mảng JSON trả về có cấu trúc đầy đủ của tất cả các bước (cả cũ và mới).

Định dạng trả về bắt buộc:
```json
[
  {{"id": 1, "task": "...", "status": "completed/failed/pending", "result": "..."}},
  ...
]
```
"""


def planner_node(state:PlanState)->Dict[str,Any]:
    print("\n=== [NODE] Planner: Đang phác thảo kế hoạch... ===")
    messages = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=f'Yêu cầu lập trình : {state["input_request"]}')
    ]
    
    response = model.invoke(messages)
    
    try:
        parsed_plan = robust_json_parser(response.content)
        return {
            'plan':parsed_plan,
            'error_count':0
        }
    except Exception as e:
        print(f"⚠️ Lỗi phân tích cú pháp Planner: {str(e)}")
        fallback_plan = [
            {"id": 1, "task": f"Tạo giải pháp cho: {state['input_request']}", "status": "pending", "result": ""}
        ]
        return {"plan": fallback_plan, "error_count": 1}
    
def executor_node(state:PlanState):
    plan = state['plan']
    active_task = next((t for t in plan if t['status'] == 'pending'),None)
    
    if not active_task:
        return {"latest_output":"Không còn tác vụ nào cần xử lý"}
    current_id = active_task["id"]
    task_desc = active_task["task"]
    print(f"\n=== [NODE] Executor: Đang xử lý Tác vụ {current_id} ===")
    print(f"-> Nhiệm vụ: {task_desc}")

    executor_prompt = f"""Bạn là một lập trình viên thực thi. Hãy hoàn thành nhiệm vụ sau đây:
    Nhiệm vụ: {task_desc}

    Yêu cầu:
    Nếu nhiệm vụ yêu cầu viết mã nguồn, hãy chỉ viết mã nguồn Python hoàn chỉnh và bọc trong khối mã ```python ... ```.
    Đảm bảo mã nguồn chạy được, không lỗi cú pháp.
    """
    
    response = model.invoke(executor_prompt)
    model_output = response.content

    code_match = re.search(r"```python\s*(.*?)\s*```", model_output, re.DOTALL)
    if code_match:
        code_to_run = code_match.group(1).strip() 
        print("🔧 Phát hiện mã nguồn Python. Tiến hành kiểm tra an toàn...")
        
        with tempfile.NamedTemporaryFile(suffix='.py',delete=False) as temp_file:
            temp_file.write(code_to_run.encode('utf-8'))
            temp_filepath = temp_file.name
        
        try:
            result = subprocess.run(
                ['python',temp_filepath],
                capture_output=True,
                timeout=5,
                text=True
            )
            os.remove(temp_filepath)

            if result.returncode == 0:
                execution_result = f"Thực thi thành công. Output:\n{result.stdout}"
                print("✅ Chạy thử thành công!")
            else:
                execution_result = f"Lỗi Runtime (Mã lỗi {result.returncode}):\n{result.stderr}"
                print("❌ Chạy thử thất bại!")
        except Exception as ex:
            execution_result = f"Không thể thực thi mã nguồn: {str(ex)}"
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
    else:
        # Nếu nhiệm vụ chỉ là phân tích hoặc cấu trúc không chứa mã nguồn
        execution_result = model_output
        print("💬 Tác vụ không chứa mã nguồn Python trực tiếp. Trả về văn bản phân tích.")
    return {
        "current_task_id": current_id,
        "latest_output": execution_result
    } 
    
def replanner_node(state:PlanState):
    print("\n=== [NODE] Replanner: Đang cập nhật trạng thái tiến trình... ===")
    plan_str = json.dumps(state['plan'], indent=2,ensure_ascii=False)
    
    prompt = REPLANNER_SYSTEM_PROMPT.format(current_task_id = state["current_task_id"])

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=f"Kế hoạch hiện tại:\n{plan_str}\n\nKết quả thực thi tác vụ {state['current_task_id']} mới nhất:\n{state['latest_output']}")
    ]
    
    response = model.invoke(messages)
    try:
        updated_plan = robust_json_parser(response.content)
        return {'plan':updated_plan}
    except Exception as e:
        print(f"⚠️ Lỗi phân tích cú pháp Replanner: {str(e)}")
        # Xử lý fallback thủ công nếu Replanner bị lỗi parser: tự đánh dấu hoàn thành tác vụ hiện tại để tránh lặp vô hạn
        fallback_plan = []
        for task in state["plan"]:
            if task["id"] == state["current_task_id"]:
                task["status"] = "failed"
                task["result"] = f"Lỗi hệ thống parser: {str(e)}"
            fallback_plan.append(task)
        
        return {
            "plan": fallback_plan,
            "error_count": state.get("error_count", 0) + 1
        }

def should_continue(state: PlanState):
    plan = state['plan']

    if state.get("error_count",0)>=3:
        print("🛑 Đạt giới hạn lỗi hệ thống (3 lỗi). Dừng đồ thị khẩn cấp để đảm bảo an toàn.")
        return END
    pending_tasks = [t for t in plan if t['status']=='pending']
    if not pending_tasks:
        print("\n🎉 Tất cả các tác vụ trong kế hoạch đã được xử lý hoàn tất!")
        return END
    print(f"🔄 Vẫn còn {len(pending_tasks)} tác vụ đang chờ. Chuyển tiếp tới Executor.")
    return "continue"

builder = StateGraph(PlanState)
builder.add_node('planner',planner_node)
builder.add_node('executor',executor_node)
builder.add_node('replanner',replanner_node)

builder.add_edge(START, "planner")
builder.add_edge("planner", "executor")
builder.add_edge("executor", "replanner")

# Thiết lập cạnh rẽ nhánh (Conditional Edges) sau bước Replanner
builder.add_conditional_edges(
    "replanner",
    should_continue,
    {
        "continue": "executor",
        END: END
    }
)
app = builder.compile()
# inputs = {
#     "input_request": """Viết một hàm Python tính giai thừa của một số nguyên dương đầu vào. 
#                      Nếu người dùng truyền vào số âm hoặc kiểu dữ liệu không phải là số nguyên, 
#                      hàm phải ném ra ngoại lệ ValueError kèm thông báo phù hợp. Viết kèm 2 dòng kiểm thử.
#                      """
# }

# # Chạy đồ thị ở chế độ stream_mode='values' để xem toàn bộ sự thay đổi State theo thời gian thực
# for event in app.stream(inputs, {"configurable": {"thread_id": "demo_tdd_1"}}, stream_mode="values"):
#     if "plan" in event:
#         print("\n--- [TRẠNG THÁI KẾ HOẠCH HIỆN TẠI] ---")
#         for task in event["plan"]:
#             status_symbol = "⏳" if task["status"] == "pending" else "✅" if task["status"] == "completed" else "❌"
#             print(f"{status_symbol} Bước {task['id']}: {task['task']}")
#             if task["result"]:
#                 print(f"   ↳ Kết quả: {task['result'][:150]}...")