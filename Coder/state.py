# state.py
from typing import List, Dict, Any, Literal, Sequence, TypedDict, Annotated, Union
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
    dependencies: List[str] = Field(default=[], description="Danh sách ID các nhiệm vụ cần hoàn thành trước khi bắt đầu nhiệm vụ này, ví dụ: ['T1']")
    status: Literal["pending", "completed"] = Field(default="pending", description="Trạng thái thực thi nhiệm vụ")

class TaskPlan(BaseModel):
    tasks: List[Task] = Field(description="Danh sách các nhiệm vụ có thiết lập quan hệ phụ thuộc lẫn nhau (DAG).")
    explanation: str = Field(description="Mô tả chiến lược thực hiện nhiệm vụ và cách xử lý tuần tự/song song.")
    task_type: Literal["analysis", "development"] = Field(
        description="Phân loại yêu cầu: 'analysis' nếu chỉ đọc/khảo sát/báo cáo thông tin, 'development' nếu có viết/sửa/nâng cấp mã nguồn."
    )

class TaskTriage(BaseModel):
    is_simple: bool = Field(
        description="True nếu yêu cầu cực kỳ đơn giản (chỉ cần sửa đổi trực tiếp 1-2 file, thêm tính năng nhỏ, sửa lỗi cú pháp). False nếu yêu cầu phức tạp cần khảo sát sâu hoặc thiết kế nhiều bước."
    )
    task_type: Literal["analysis", "development"] = Field(
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


# Trạng thái của Đồ thị chính
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    workspace_path: str
    workspace_context: str  
    plan: List[Task]                      
    task_type: Literal["analysis", "development"]
    git_branch: str
    error_logs: str
    modified_files: List[str]
    attempts: int
    step_findings: Annotated[List[str], reduce_findings]
    last_executed_task_ids: List[str]     


# Trạng thái độc lập của Đồ thị con dò tìm Workspace [1.2.2]
class WorkspaceDiscoveryState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    workspace_path: str
    finished: bool