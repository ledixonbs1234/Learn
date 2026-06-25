# state.py
from typing import List, Dict, Any, Literal, Sequence, TypedDict, Annotated, Union
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# ==========================================
# CẤU TRÚC ĐẦU RA MONG MUỐN (STRUCTURED OUTPUT)
# ==========================================
class WorkspaceDetection(BaseModel):
    workspace_path: str = Field(description="Đường dẫn của workspace chứa dự án.")

class TaskPlan(BaseModel):
    steps: List[str] = Field(description="Các bước thực hiện tuần tự để giải quyết yêu cầu.")
    explanation: str = Field(description="Mô tả chiến lược thực hiện nhiệm vụ.")
    task_type: Literal["analysis", "development"] = Field(
        description="Phân loại yêu cầu: 'analysis' nếu chỉ đọc/khảo sát/báo cáo thông tin, 'development' nếu có viết/sửa/nâng cấp mã nguồn."
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


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    workspace_path: str
    plan: List[str]
    task_type: Literal["analysis", "development"]
    current_step_idx: int
    git_branch: str
    error_logs: str
    modified_files: List[str]
    attempts: int
    step_findings: Annotated[List[str], reduce_findings]