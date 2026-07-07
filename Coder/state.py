# Coder/state.py
from typing import List, Dict, Any, Literal, Optional, Sequence, TypedDict, Annotated, Union
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# =====================================================================
# CẤU TRÚC DỮ LIỆU PYDANTIC SỬ DỤNG TRONG LẬP KẾ HOẠCH & PHÂN LOẠI
# =====================================================================

class Task(BaseModel):
    id: str = Field(description="Mã định danh duy nhất cho nhiệm vụ, ví dụ: 'T1', 'T2'")
    description: str = Field(description="Mô tả chi tiết hành động cần thực hiện")
    dependencies: List[str] = Field(
        default_factory=list,
        description="Mảng chứa các ID nhiệm vụ cần hoàn thành trước. Nếu không phụ thuộc trả về mảng rỗng []"
    )
    status: Literal["pending", "completed"] = Field(
        default="pending", 
        description="Trạng thái thực thi nhiệm vụ."
    )

class TaskPlan(BaseModel):
    """Bản kế hoạch DAG khởi tạo ban đầu được đề xuất bởi Kiến trúc sư."""
    tasks: List[Task] = Field(
        default_factory=list,
        description="Danh sách có thứ tự của các nhiệm vụ cần thực hiện (DAG)."
    )
    explanation: str = Field(
        default="",
        description="Phân tích chiến lược triển khai và giải thích cách xử lý các tác vụ bằng tiếng Việt."
    )
    task_type: Literal["analysis", "development"] = Field(
        default="development",
        description="Phân loại hướng xử lý chính: 'analysis' hoặc 'development'."
    )

class PlanUpdate(BaseModel):
    """Bản cập nhật kế hoạch được sử dụng khi tái lập lộ trình (Replanning)."""
    should_modify_plan: bool = Field(
        default=False,
        description="True nếu cần sửa đổi hoặc bổ sung thêm nhiệm vụ mới vào kế hoạch. False nếu giữ nguyên."
    )
    explanation: str = Field(
        default="",
        description="Giải thích chi tiết lý do điều chỉnh hoặc giữ nguyên kế hoạch."
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
    """Phân loại tác vụ đầu vào tại điểm xuất phát."""
    is_simple: bool = Field(
        default=False,
        description="True nếu yêu cầu cực kỳ đơn giản. False nếu phức tạp cần lên kế hoạch nhiều bước."
    )
    task_type: Literal["analysis", "development"] = Field(
        default="development",
        description="Phân loại hướng xử lý của yêu cầu."
    )
    detailed_analysis: str = Field(
        default="",
        description="Bản phân tích chi tiết yêu cầu người dùng (Mục đích, file bị tác động, ràng buộc)."
    )
    recommended_skills: List[str] = Field(
        default_factory=list,
        description="Danh sách các tên định danh kỹ năng phù hợp nhất từ thư viện .skills/."
    )

class RuntimeVerificationResult(BaseModel):
    has_critical_error: bool = Field(description="True nếu phát hiện lỗi crash, exception nghiêm trọng.")
    error_summary: str = Field(description="Tóm tắt lỗi runtime phát hiện được.")

# =====================================================================
# KHÔI PHỤC HOÀN TOÀN SCHEMA ĐỊNH NGHĨA KỸ NĂNG ĐỘNG (DYNAMIC SKILLS)
# =====================================================================

class SkillParameter(BaseModel):
    type: str = Field(description="Kiểu dữ liệu của tham số (string, integer, boolean, object, array).")
    description: str = Field(description="Mô tả chi tiết bằng tiếng Việt về tham số này.")
    required: bool = Field(default=True, description="Tham số này có bắt buộc không.")

class SkillDefinition(BaseModel):
    name: str = Field(description="Tên định danh của kỹ năng, ví dụ: 'excel_edit_cell'.")
    description: str = Field(description="Mô tả chi tiết nhiệm vụ và trường hợp sử dụng của kỹ năng này.")
    parameters: Dict[str, SkillParameter] = Field(description="Từ điển chứa các tham số đầu vào cần thiết.")
    usage_example: str = Field(description="Ví dụ cụ thể về cách chuẩn bị tham số và kết quả mong đợi.")

# =====================================================================
# HÀM BỔ TRỢ ÉP KIỂU PHÒNG THỦ TRÁNH LỖI CHECKPOINT SERIALIZATION
# =====================================================================

def ensure_task_objects(plan: List[Union[Task, dict]]) -> List[Task]:
    """Chuyển đổi đồng bộ các dict thô thu được từ checkpoint trở lại thành đối tượng Task Pydantic [1]."""
    if not plan:
        return []
    return [t if isinstance(t, Task) else Task(**t) for t in plan]

# =====================================================================
# CUSTOM REDUCERS VÀ STATE GRAPH
# =====================================================================

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
    task_type: Literal["analysis", "development", "clarify"]
    git_branch: str
    error_logs: str
    modified_files: List[str]
    file_registry: Annotated[Dict[str, str], reduce_file_registry] 
    attempts: int
    step_findings: Annotated[List[str], reduce_findings]
    last_executed_task_ids: List[str]     
    replanning_count: int
    is_simple: bool
    detailed_analysis: str
    extension_path: str
    browser_console_logs: str
    active_skills: Dict[str, str]
    doubt_findings: str         # Lưu kết quả rà soát đối kháng
    doubt_attempts: int
    recommended_skills: List[str]

class WebInteractionState(TypedDict):
    workspace_path: str 
    url: str
    action_type: Literal["explore", "test_js"]
    target_description: str
    js_code_to_test: Optional[str]
    extension_path: Optional[str]
    browser_console_logs: Optional[str]
    detected_selectors: Optional[Dict[str, Any]]
    execution_success: Optional[bool]
    dom_state_after: Optional[Dict[str, Any]]
    screenshot_path: Optional[str]
    error: Optional[str]
    attempts: int