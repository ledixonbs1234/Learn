# state.py
from typing import List, Dict, Any, Literal, Optional, Sequence, TypedDict, Annotated, Union
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# ==========================================
# CẤU TRÚC ĐẦU RA MONG MUỐN (STRUCTURED OUTPUT)
# ==========================================
class WorkspaceDetection(BaseModel):
    workspace_path: str = Field(description="Đường dẫn tuyệt đối đã xác minh của workspace chứa dự án.")

class Task(BaseModel):
    id: str = Field(description="Mã định danh duy nhất cho nhiệm vụ, ví dụ: 'T1', 'T2'")
    description: str = Field(description="Mô tả chi tiết hành động cần thực hiện")
    dependencies: List[str] = Field(
        default_factory=list,
        description="Mảng chứa các ID nhiệm vụ cần hoàn thành trước. Bắt buộc phải có trường này, nếu không phụ thuộc ai hãy trả về mảng rỗng []"
    )
    status: Literal["pending", "completed"] = Field(
        default="pending", 
        description="Trạng thái thực thi nhiệm vụ. Luôn luôn khởi tạo là 'pending'"
    )

class TaskPlan(BaseModel):
    tasks: List[Task] = Field(
        default_factory=list,
        description="Danh sách có thứ tự của các nhiệm vụ cần thực hiện (thiết lập quan hệ DAG chặt chẽ)."
    )
    explanation: str = Field(
        default="",
        description="Phân tích chiến lược triển khai và giải thích cách xử lý các tác vụ."
    )
    task_type: Literal["analysis", "development"] = Field(
        default="development",
        description="Phân loại hướng xử lý của toàn bộ yêu cầu: 'analysis' hoặc 'development'."
    )

class PlanUpdate(BaseModel):
    should_modify_plan: bool = Field(
        default=False,
        description="True nếu dựa trên kết quả thực thi vừa qua, bạn thấy cần sửa đổi hoặc bổ sung thêm nhiệm vụ mới vào kế hoạch. False nếu kế hoạch hiện tại vẫn đúng đắn và có thể tiếp tục trực tiếp."
    )
    explanation: str = Field(
        default="",
        description="Giải thích chi tiết lý do tại sao quyết định điều chỉnh hoặc giữ nguyên kế hoạch hành động."
    )
    updated_tasks: List[Task] = Field(
        default_factory=list,
        description="Danh sách toàn bộ các nhiệm vụ (gồm cả nhiệm vụ cũ đã hoàn thành và nhiệm vụ mới/sửa đổi)."
    )
    task_type: Literal["analysis", "development"] = Field(
        default="development",
        description="Phân loại hướng xử lý tiếp theo của kế hoạch."
    )

class TaskTriage(BaseModel):
    is_simple: bool = Field(
        default=False,
        description="True nếu yêu cầu cực kỳ đơn giản. False nếu yêu cầu phức tạp cần khảo sát sâu hoặc thiết kế nhiều bước."
    )
    task_type: Literal["analysis", "development"] = Field(
        default="development",
        description="Phân loại hướng xử lý của yêu cầu."
    )

# ==========================================
# CUSTOM REDUCERS VÀ STATE GRAPH
# ==========================================
def reduce_findings(left: Union[List[str], None], right: Union[List[str], None]) -> List[str]:
    left_list = left or []
    right_list = right or []
    if not right_list:
        return left_list
    if right_list[0] == "__RESET__":
        return right_list[1:]
    return left_list + right_list


def reduce_file_registry(left: Dict[str, str], right: Dict[str, str]) -> Dict[str, str]:
    merged = dict(left or {})
    if right:
        merged.update(right)
    return merged

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    workspace_path: str
    workspace_context: str  
    plan: List[Task]                      
    task_type: Literal["analysis", "development"]
    git_branch: str
    error_logs: str
    modified_files: List[str]
    file_registry: Annotated[Dict[str, str], reduce_file_registry] 
    attempts: int
    step_findings: Annotated[List[str], reduce_findings]
    last_executed_task_ids: List[str]     
    replanning_count: int
    is_simple: bool
    

class WebInteractionState(TypedDict):
    workspace_path: str 
    url: str
    action_type: Literal["explore", "test_js"]
    target_description: str
    js_code_to_test: Optional[str]           # Code JS mà Agent chính muốn chạy thử
    
    # Kết quả trả về
    detected_selectors: Optional[Dict[str, Any]]
    execution_success: Optional[bool]
    dom_state_after: Optional[Dict[str, Any]]
    screenshot_path: Optional[str]
    error: Optional[str]
    attempts: int