import os
import re
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Literal, Sequence, TypedDict, Annotated
import operator

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END, add_messages

# ==========================================
# 1. CẤU HÌNH TRACING LANGSMITH
# ==========================================
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "LangGraph-Production-Coder-V3"


# ==========================================
# 2. KIỂM SOÁT ĐƯỜNG DẪN AN TOÀN (CROSS-PLATFORM)
# ==========================================

def sanitize_and_resolve_path(workspace: str, raw_target_path: str) -> Path:
    """
    Chuẩn hóa đường dẫn phù hợp với hệ điều hành hiện tại (Windows/Linux)
    và ngăn chặn lỗ hổng bảo mật Path Traversal.
    """
    cleaned = raw_target_path.replace('"', '').replace("'", "").replace("\\", "/").strip()
    workspace_path = Path(workspace).resolve()
    target_path = Path(cleaned)
    
    if target_path.is_absolute():
        try:
            target_path = target_path.relative_to(workspace_path)
        except ValueError:
            parts = target_path.parts
            if parts[0].endswith(":") or parts[0] == "/":
                target_path = Path(*parts[1:])
                
    final_path = (workspace_path / target_path).resolve()
    
    if not str(final_path).startswith(str(workspace_path)):
        raise ValueError(f"Cảnh báo bảo mật: Đường dẫn nằm ngoài vùng an toàn.")
        
    final_path.parent.mkdir(parents=True, exist_ok=True)
    return final_path


# ==========================================
# 3. QUẢN LÝ GIT BIỆT LẬP (GIT MANAGER)
# ==========================================

class GitManager:
    """
    Quản lý quy trình kiểm soát phiên bản tự động của AI.
    """
    def __init__(self, workspace_path: str):
        self.workspace = Path(workspace_path).resolve()
        
    def _run_cmd(self, args: list) -> str:
        try:
            res = subprocess.run(args, cwd=str(self.workspace), capture_output=True, text=True, check=True)
            return res.stdout.strip()
        except subprocess.CalledProcessError as e:
            return f"ERROR: {e.stderr.strip()}"

    def init_and_prepare_branch(self) -> str:
        """
        Khởi tạo Git nếu chưa có, hoặc chuyển sang nhánh tạm ai-development cố định.
        """
        git_dir = self.workspace / ".git"
        if not git_dir.exists():
            self._run_cmd(["git", "init"])
            self._run_cmd(["git", "config", "user.name", "AI-Agent"])
            self._run_cmd(["git", "config", "user.email", "ai-agent@production.local"])
            self._run_cmd(["git", "add", "."])
            self._run_cmd(["git", "commit", "-m", "Initial commit from Agent Workspace Setup"])
            
        branch_name = "ai-development"
        branches_str = self._run_cmd(["git", "branch"])
        
        if branch_name in branches_str:
            self._run_cmd(["git", "checkout", branch_name])
        else:
            self._run_cmd(["git", "checkout", "-b", branch_name])
            
        return branch_name

    def commit_changes(self, message: str):
        self._run_cmd(["git", "add", "."])
        status = self._run_cmd(["git", "status", "--porcelain"])
        if status and not status.startswith("ERROR"):
            self._run_cmd(["git", "commit", "-m", message])


# ==========================================
# 4. HÀM BÓC TÁCH NỘI DUNG TỆP (ROBUST EXTRACTOR)
# ==========================================

def extract_file_content(raw_content: str) -> str:
    """
    Bóc tách nội dung tệp tin từ phản hồi của LLM một cách mạnh mẽ.
    Hỗ trợ hoàn hảo việc viết các tệp lồng nhau (như Markdown chứa khối code con).
    """
    # Tìm vị trí xuất hiện của dấu mở khối mã đầu tiên
    first_triple_backtick = raw_content.find("```")
    if first_triple_backtick == -1:
        return ""
    
    # Bỏ qua dòng khai báo ngôn ngữ đầu tiên (ví dụ: ```python hoặc ```markdown)
    start_idx = raw_content.find("\n", first_triple_backtick)
    if start_idx == -1:
        start_idx = first_triple_backtick + 3
    else:
        start_idx += 1
        
    # Tìm dấu đóng khối mã cuối cùng từ dưới lên để bao quát toàn bộ nội dung
    last_triple_backtick = raw_content.rfind("```")
    if last_triple_backtick == -1 or last_triple_backtick <= first_triple_backtick:
        return raw_content[start_idx:]
        
    return raw_content[start_idx:last_triple_backtick].strip()


# ==========================================
# 5. ĐỊNH NGHĨA CÁC ĐỐI TƯỢNG CẤU TRÚC (PYDANTIC)
# ==========================================

class WorkspaceDetection(BaseModel):
    workspace_path: str = Field(description="Đường dẫn của workspace chứa dự án.")

class TaskPlan(BaseModel):
    steps: List[str] = Field(description="Các bước thực hiện tuần tự để giải quyết yêu cầu.")
    explanation: str = Field(description="Mô tả chiến lược thực hiện.")


# ==========================================
# 6. TRẠNG THÁI HỆ THỐNG (STATE GRAPH)
# ==========================================

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage],add_messages]
    workspace_path: str
    plan: List[str]
    current_step_idx: int
    git_branch: str
    error_logs: str
    modified_files: List[str]
    attempts: int


# ==========================================
# 7. KHỞI TẠO MODEL VÀ CÁC NODES XỬ LÝ
# ==========================================

model = ChatOpenAI(
    base_url="http://localhost:20128/v1",
    api_key="khongco", # type: ignore
    model="kiro",
    temperature=0.1
)

def detect_workspace_node(state: AgentState) -> Dict[str, Any]:
    last_message = state["messages"][-1].content
    structured_llm = model.with_structured_output(WorkspaceDetection, method="function_calling")
    
    detected = structured_llm.invoke([
        {"role": "system", "content": "Trích xuất đường dẫn thư mục dự án cần xử lý từ yêu cầu người dùng."},
        {"role": "user", "content": last_message}
    ])
    
    raw_ws = getattr(detected, "workspace_path", ".")
    resolved_ws = str(Path(raw_ws).resolve())
    return {
        "workspace_path": resolved_ws,
        "messages": [AIMessage(content=f"Đã phát hiện workspace tại: `{resolved_ws}`")]
    }


def git_setup_node(state: AgentState) -> Dict[str, Any]:
    ws = state["workspace_path"]
    git_manager = GitManager(ws)
    branch = git_manager.init_and_prepare_branch()
    return {
        "git_branch": branch,
        "messages": [AIMessage(content=f"Đã chuẩn bị nhánh Git làm việc: `{branch}`")]
    }


def planner_node(state: AgentState) -> Dict[str, Any]:
    ws = state["workspace_path"]
    user_request = state["messages"][0].content
    
    structured_llm = model.with_structured_output(TaskPlan, method="function_calling")
    system_prompt = (
        "Bạn là một Kiến trúc sư phần mềm cấp cao. Hãy lập một kế hoạch thực hiện "
        f"cho dự án nằm trong thư mục '{ws}'. Nếu yêu cầu là đọc/khảo sát dự án, "
        "hãy lên kế hoạch tạo một tệp tin tài liệu (ví dụ: `project_structure.md`) để lưu trữ kết quả phân tích."
    )
    
    plan_output = structured_llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_request}
    ])
    
    plan_steps = getattr(plan_output, "steps", [])
    return {
        "plan": plan_steps,
        "current_step_idx": 0,
        "messages": [AIMessage(content="Đã lập kế hoạch thực hiện:\n" + "\n".join([f"- {s}" for s in plan_steps]))]
    }


def executor_node(state: AgentState) -> Dict[str, Any]:
    ws = state["workspace_path"]
    steps = state["plan"]
    current_idx = state["current_step_idx"]
    error_logs = state.get("error_logs", "")
    
    if current_idx >= len(steps):
         return {"messages": [AIMessage(content="Đã hoàn thành toàn bộ kế hoạch.")]}
         
    current_step = steps[current_idx]
    
    prompt = f"""
    Bạn đang thực hiện bước {current_idx + 1}/{len(steps)}: "{current_step}"
    Workspace hiện hành: {ws}
    """
    if error_logs:
        prompt += f"\nVUI LÒNG SỬA LỖI TỪ LƯỢT CHẠY TRƯỚC:\n{error_logs}"
        
    prompt += """
    QUY TẮC BẮT BUỘC:
    1. Chỉ định đường dẫn tệp tại dòng đầu tiên: [FILE_PATH]: <đường_dẫn_tương_đối_trong_workspace>
    2. Viết toàn bộ nội dung tệp tin bên trong khối markdown chuẩn:
    ```
    Nội dung tệp
    ```
    Hãy đảm bảo đường dẫn tệp là chính xác để hệ thống ghi dữ liệu.
    """
    
    response = model.invoke([
        {"role": "system", "content": "Bạn là kỹ sư lập trình thực thi. Viết mã nguồn hoặc tài liệu phân tích chi tiết."},
        {"role": "user", "content": prompt}
    ])
    
    raw_content = response.content
    if not isinstance(raw_content, str):
        raw_content = str(raw_content)
        
    path_match = re.search(r"\[FILE_PATH\]:\s*([^\n]+)", raw_content)
    code_content = extract_file_content(raw_content)
    
    modified_files = list(state.get("modified_files", []))
    
    if path_match and code_content:
        raw_path = path_match.group(1).strip()
        safe_path = sanitize_and_resolve_path(ws, raw_path)
        
        with open(safe_path, "w", encoding="utf-8") as f:
            f.write(code_content)
            
        modified_files.append(str(safe_path))
        status_msg = f"Đã cập nhật tệp tin: `{safe_path.relative_to(ws)}` thành công."
    else:
        status_msg = "Bước này không phát sinh thay đổi tệp tin vật lý hoặc định dạng phản hồi không hợp lệ."

    # LƯU Ý QUAN TRỌNG: Tăng chỉ số bước (current_step_idx) tại đây để đảm bảo lưu trữ vào State Checkpoint
    return {
        "messages": [AIMessage(content=f"Thực thi bước '{current_step}':\n{status_msg}")],
        "modified_files": modified_files,
        "error_logs": "",
        "current_step_idx": current_idx + 1 
    }


def tester_node(state: AgentState) -> Dict[str, Any]:
    modified_files = state.get("modified_files", [])
    attempts = state.get("attempts", 0)
    errors = []
    
    for file_path in modified_files:
        if file_path.endswith(".py"):
            res = subprocess.run(["python", "-m", "py_compile", file_path], capture_output=True, text=True)
            if res.returncode != 0:
                errors.append(f"Lỗi cú pháp tại {Path(file_path).name}:\n{res.stderr.strip()}")
                
    if errors:
        combined_error = "\n".join(errors)
        return {
            "error_logs": combined_error,
            "attempts": attempts + 1,
            "messages": [AIMessage(content=f"Phát hiện lỗi kiểm tra:\n{combined_error}")]
        }
        
    return {
        "error_logs": "",
        "attempts": attempts,
        "messages": [AIMessage(content="Tất cả các tệp tin đã vượt qua vòng kiểm tra thành công.")]
    }


def commit_node(state: AgentState) -> Dict[str, Any]:
    ws = state["workspace_path"]
    git_manager = GitManager(ws)
    commit_msg = "feat(ai): automatic documentation and code implementation \n\nSteps:\n" + "\n".join(state["plan"])
    git_manager.commit_changes(commit_msg)
    return {
        "messages": [AIMessage(content=f"Đã hoàn thành xuất sắc yêu cầu và commit lên nhánh `{state['git_branch']}`.")]
    }


# ==========================================
# 8. CẤU HÌNH ĐỊNH TUYẾN ĐỒ THỊ (PURI R/O ROUTERS)
# ==========================================

def workflow_step_router(state: AgentState) -> Literal["executor", "tester"]:
    """
    Router an toàn: Chỉ đọc trạng thái để đưa ra quyết định rẽ nhánh.
    """
    current_idx = state["current_step_idx"]
    plan = state["plan"]
    
    # So sánh chỉ số hiện tại đã hoàn thành hết các phần của Plan chưa
    if current_idx < len(plan):
        return "executor"
    return "tester"


def error_recovery_router(state: AgentState) -> Literal["executor", "commit"]:
    error = state.get("error_logs", "")
    attempts = state.get("attempts", 0)
    
    if error and attempts < 3:
        return "executor"
    return "commit"


# ==========================================
# 9. KẾT NỐI ĐỒ THỊ LANGGRAPH
# ==========================================

builder = StateGraph(AgentState)

# Thêm các Nodes
builder.add_node("detect_workspace", detect_workspace_node)
builder.add_node("git_setup", git_setup_node)
builder.add_node("planner", planner_node)
builder.add_node("executor", executor_node)
builder.add_node("tester", tester_node)
builder.add_node("commit", commit_node)

# Thiết lập Luồng đi cố định
builder.add_edge(START, "detect_workspace")
builder.add_edge("detect_workspace", "git_setup")
builder.add_edge("git_setup", "planner")
builder.add_edge("planner", "executor")

# Vòng lặp các bước trong kế hoạch (Sử dụng Router an toàn)
builder.add_conditional_edges(
    "executor",
    workflow_step_router,
    {
        "executor": "executor",
        "tester": "tester"
    }
)

# Vòng lặp sửa lỗi tự phục hồi (Self-Healing)
builder.add_conditional_edges(
    "tester",
    error_recovery_router,
    {
        "executor": "executor",
        "commit": "commit"
    }
)

builder.add_edge("commit", END)

app = builder.compile()