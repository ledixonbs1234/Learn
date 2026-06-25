import os
import re
import subprocess
import fnmatch
from pathlib import Path
from typing import List, Dict, Any, Literal, Sequence, TypedDict, Annotated, Union

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.checkpoint.memory import MemorySaver  # Kích hoạt lưu trữ phiên làm việc

# ==========================================
# 1. CẤU HÌNH TRACING LANGSMITH
# ==========================================
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "LangGraph-Production-Coder-V3"


class GitIgnoreMatcher:
    """
    Phân tích cú pháp .gitignore để xác định xem một đường dẫn có bị bỏ qua hay không.
    """
    def __init__(self, workspace_path: Path):
        self.workspace = workspace_path
        self.patterns = []
        
        # Đọc tệp .gitignore nếu có
        gitignore_file = workspace_path / ".gitignore"
        if gitignore_file.exists():
            try:
                for line in gitignore_file.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    # Bỏ qua dòng trống hoặc dòng chú thích
                    if line and not line.startswith("#"):
                        self.patterns.append(line)
            except Exception:
                pass
                
        # Mặc định luôn bổ sung các thư mục hệ thống/thư mục build rác phổ biến để bảo vệ bộ nhớ Context
        self.patterns.extend([
            ".git", "__pycache__", "*.pyc", ".DS_Store", "node_modules", 
            ".dart_tool", "build", "ios/Pods", "android/.gradle"
        ])

    def is_ignored(self, path: Path) -> bool:
        try:
            rel_path = path.relative_to(self.workspace)
        except ValueError:
            return False
            
        rel_str = rel_path.as_posix()
        
        for pattern in self.patterns:
            pat = pattern.strip()
            if pat.endswith('/'):
                pat = pat[:-1]
            
            # 1. Khớp toàn bộ đường dẫn tương đối (bao gồm cả wildcard và tiền tố)
            if (fnmatch.fnmatch(rel_str, pat) or 
                fnmatch.fnmatch(rel_str, pat + "/*") or 
                fnmatch.fnmatch(rel_str, "*/" + pat) or 
                fnmatch.fnmatch(rel_str, "*/" + pat + "/*")):
                return True
                
            # 2. Khớp trực tiếp với tên tệp tin cụ thể (ví dụ: *.log)
            if fnmatch.fnmatch(rel_path.name, pat):
                return True
                
            # 3. Khớp nếu bất kỳ thư mục cha nào trùng khớp hoàn toàn với tên của pattern
            for part in rel_path.parts:
                if fnmatch.fnmatch(part, pat):
                    return True
                    
        return False


# ==========================================
# 2. KIỂM SOÁT ĐƯỜNG DẪN AN TOÀN (CROSS-PLATFORM)
# ==========================================

def sanitize_and_resolve_path(workspace: str, raw_target_path: str, create_parent: bool = False) -> Path:
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
        
    if create_parent:
        final_path.parent.mkdir(parents=True, exist_ok=True)
    return final_path


# ==========================================
# 3. QUẢN LÝ GIT BIỆT LẬP (GIT MANAGER)
# ==========================================

class GitManager:
    def __init__(self, workspace_path: str):
        self.workspace = Path(workspace_path).resolve()
        
    def _run_cmd(self, args: list) -> str:
        try:
            res = subprocess.run(args, cwd=str(self.workspace), capture_output=True, text=True, check=True)
            return res.stdout.strip()
        except subprocess.CalledProcessError as e:
            return f"ERROR: {e.stderr.strip()}"

    def init_and_prepare_branch(self) -> str:
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
# 4. HỆ THỐNG CÔNG CỤ WORKSPACE THỰC TẾ (REAL WORKSPACE TOOLS)
# ==========================================

class WorkspaceTools:
    """
    Các hàm hỗ trợ đọc, viết tệp tin và khám phá thư mục trong Workspace.
    """
    def __init__(self, workspace_path: str):
        self.workspace = Path(workspace_path).resolve()

    def read_files(self, file_paths: Union[str, List[str]]) -> str:
        """
        Đọc nội dung của một hoặc nhiều tệp tin trong workspace cùng lúc.
        """
        # Chuẩn hóa đầu vào thành danh sách nếu người dùng truyền một chuỗi đơn lẻ
        paths = [file_paths] if isinstance(file_paths, str) else file_paths
        
        if not paths:
            return "Lỗi: Danh sách đường dẫn tệp tin trống."
            
        results = []
        for path in paths:
            try:
                safe_path = sanitize_and_resolve_path(str(self.workspace), path, create_parent=False)
                if not safe_path.exists():
                    results.append(f"--- THẤT BẠI: '{path}' (Tệp không tồn tại) ---")
                    continue
                if safe_path.is_dir():
                    results.append(f"--- THẤT BẠI: '{path}' (Đường dẫn được chỉ định là thư mục) ---")
                    continue
                
                content = safe_path.read_text(encoding="utf-8")
                results.append(f"=== BẮT ĐẦU NỘI DUNG TỆP: {path} ===\n{content}\n=== KẾT THÚC NỘI DUNG TỆP: {path} ===")
            except Exception as e:
                results.append(f"--- THẤT BẠI: '{path}' (Lỗi đọc tệp: {str(e)}) ---")
                
        return "\n\n".join(results)

    def write_file(self, file_path: str, content: str) -> str:
        try:
            safe_path = sanitize_and_resolve_path(str(self.workspace), file_path, create_parent=True)
            safe_path.write_text(content, encoding="utf-8")
            return f"Đã ghi và lưu tệp thành công tại đường dẫn: '{file_path}'"
        except Exception as e:
            return f"Lỗi ghi tệp: {str(e)}"

    def list_directory(self, sub_dir: str = ".") -> str:
        try:
            safe_path = sanitize_and_resolve_path(str(self.workspace), sub_dir, create_parent=False)
            if not safe_path.exists():
                return f"Lỗi: Thư mục '{sub_dir}' không tồn tại."
            
            matcher = GitIgnoreMatcher(self.workspace)
            
            # Hàm đệ quy duyệt cây thư mục cục bộ
            def traverse(current_path: Path, depth: int = 0, max_depth: int = 4) -> List[str]:
                if depth > max_depth:
                    return []
                
                results = []
                try:
                    # Sắp xếp: Thư mục lên trước, tệp tin theo sau, sắp xếp theo tên không phân biệt hoa thường
                    items = sorted(list(current_path.iterdir()), key=lambda x: (not x.is_dir(), x.name.lower()))
                except Exception as e:
                    return [f"{'  ' * depth}[Lỗi truy cập: {str(e)}]"]
                
                for item in items:
                    if matcher.is_ignored(item):
                        continue
                        
                    indent = "  " * depth
                    
                    if item.is_dir():
                        results.append(f"{indent}📁 {item.name}/")
                        # Đệ quy xuống thư mục con
                        results.extend(traverse(item, depth + 1, max_depth))
                    else:
                        results.append(f"{indent}📄 {item.name}")
                        
                return results

            # Khởi động quét đệ quy từ thư mục chỉ định
            tree_lines = traverse(safe_path, depth=0, max_depth=4)
            
            if not tree_lines:
                return f"Thư mục '{sub_dir}' trống hoặc toàn bộ các tệp tin bên trong đã bị loại bỏ theo cấu hình .gitignore."
                
            header = f"Cấu trúc thư mục tương đối của '{sub_dir}' (Đã lọc bỏ tệp tin .gitignore):\n"
            return header + "\n".join(tree_lines)
            
        except Exception as e:
            return f"Lỗi liệt kê thư mục: {str(e)}"


# Định nghĩa schemas phục vụ Tool Calling cho LLM
class ReadFilesSchema(BaseModel):
    file_paths: Union[str, List[str]] = Field(
        description="Đường dẫn tương đối hoặc danh sách các đường dẫn tương đối của các tệp tin trong workspace cần đọc."
    )

class WriteFileSchema(BaseModel):
    file_path: str = Field(description="Đường dẫn tương đối của tệp tin trong workspace cần ghi hoặc cập nhật.")
    content: str = Field(description="Toàn bộ nội dung tệp tin chi tiết cần lưu xuống đĩa.")

class ListDirSchema(BaseModel):
    sub_dir: str = Field(
        default=".", 
        description="Đường dẫn tương đối của thư mục cần xem. Công cụ sẽ tự động quét đệ quy (quét sâu bên trong) và loại bỏ các thư mục, tệp tin có trong danh sách .gitignore."
    )

# ==========================================
# 5. ĐỊNH NGHĨA CÁC ĐỐI TƯỢNG CẤU TRÚC (PYDANTIC)
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
# 6. TRẠNG THÁI HỆ THỐNG (STATE GRAPH)
# ==========================================

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
    # Lấy tin nhắn cuối cùng để tìm ra đường dẫn thư mục
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
        "messages": [AIMessage(content=f"Đã phát hiện và thiết lập workspace tại: `{resolved_ws}`")]
    }


def git_setup_node(state: AgentState) -> Dict[str, Any]:
    ws = state["workspace_path"]
    git_manager = GitManager(ws)
    branch = git_manager.init_and_prepare_branch()
    return {
        "git_branch": branch,
        "messages": [AIMessage(content=f"Đã cấu hình nhánh Git hoạt động: `{branch}`")]
    }


def planner_node(state: AgentState) -> Dict[str, Any]:
    """
    Node Lập kế hoạch: Tự động phân tích toàn bộ cuộc hội thoại hiện tại
    để lập kế hoạch mới hoặc bổ sung các bước giải quyết yêu cầu mới ở Turn 2.
    """
    ws = state["workspace_path"]
    conversation_history = state["messages"]
    existing_plan = state.get("plan", [])
    existing_idx = state.get("current_step_idx", 0)
    
    structured_llm = model.with_structured_output(TaskPlan, method="function_calling")
    
    system_prompt = (
        "Bạn là một Kiến trúc sư phần mềm cấp cao. Hãy lập hoặc cập nhật kế hoạch thực hiện "
        f"cho dự án nằm trong thư mục '{ws}'.\n"
        f"Kế hoạch hiện hành của hệ thống trước đó: {existing_plan} (Đang ở bước: {existing_idx}).\n"
        "Hãy phân tích kỹ lưỡng toàn bộ lịch sử trò chuyện của người dùng để sinh ra kế hoạch hành động tiếp theo thích hợp nhất.\n"
        "Nếu người dùng yêu cầu thay đổi, sửa lỗi hoặc thêm tính năng ở lượt chat mới, hãy tích hợp các bước đọc/sửa đổi/kiểm thử cần thiết vào kế hoạch mới.\n"
        "Đặc biệt lưu ý phân loại trường `task_type` chính xác:\n"
        "- Chọn 'analysis' nếu yêu cầu chỉ là đọc, khảo sát cấu trúc dự án hoặc viết báo cáo.\n"
        "- Chọn 'development' nếu yêu cầu có can thiệp sửa đổi, cập nhật hoặc viết mới mã nguồn."
    )
    
    # Gửi kèm toàn bộ lịch sử trò chuyện để Planner không bị mất dấu ngữ cảnh
    plan_output = structured_llm.invoke([
        {"role": "system", "content": system_prompt},
        *conversation_history
    ])
    
    plan_steps = getattr(plan_output, "steps", [])
    task_type = getattr(plan_output, "task_type", "development")
    
    # Khi bắt đầu một chu kỳ kế hoạch mới (đặc biệt là Turn 2), 
    # ta reset các biến tạm thời để tránh việc mang rác của chu kỳ cũ sang
    return {
        "plan": plan_steps,
        "task_type": task_type,
        "current_step_idx": 0,
        "modified_files": [],  # Reset danh sách file sửa đổi cho chu kỳ kế hoạch mới
        "attempts": 0,         # Reset số lượt sửa lỗi cho kế hoạch mới
        "messages": [AIMessage(content=f"Đã lập kế hoạch hành động (Loại: {task_type.upper()}):\n" + "\n".join([f"- {s}" for s in plan_steps]))]
    }


def executor_node(state: AgentState) -> Dict[str, Any]:
    """
    Node thực thi thông minh: Sử dụng cơ chế Tool Calling thực tế để thao tác với Workspace.
    """
    ws = state["workspace_path"]
    steps = state["plan"]
    current_idx = state["current_step_idx"]
    error_logs = state.get("error_logs", "")
    
    if current_idx >= len(steps):
         return {"messages": [AIMessage(content="Đã hoàn thành toàn bộ các bước trong kế hoạch.")]}
         
    current_step = steps[current_idx]
    tools_mgr = WorkspaceTools(ws)
    
    # Khai báo các công cụ cục bộ tương tác động dựa trên thư mục hiện tại của Thread
    @tool(args_schema=ReadFilesSchema)
    def read_files(file_paths: Union[str, List[str]]) -> str:
        """Đọc nội dung của một hoặc nhiều tệp tin bất kỳ trong workspace cùng lúc."""
        return tools_mgr.read_files(file_paths)

    @tool(args_schema=WriteFileSchema)
    def write_file(file_path: str, content: str) -> str:
        """Ghi mới hoặc cập nhật nội dung chi tiết của một tệp tin trong workspace."""
        return tools_mgr.write_file(file_path, content)

    @tool(args_schema=ListDirSchema)
    def list_directory(sub_dir: str = ".") -> str:
        """Liệt kê các tệp và thư mục con trong thư mục chỉ định đệ quy, tự động lọc theo cấu hình tệp .gitignore."""
        return tools_mgr.list_directory(sub_dir)
    
    # Liên kết các công cụ với mô hình ngôn ngữ lớn
    tools = [read_files, write_file, list_directory]
    model_with_tools = model.bind_tools(tools)
    
    # Thiết lập chuỗi hội thoại ReAct nhỏ bên trong node để thực thi bước hiện tại
    system_prompt = (
        "Bạn là một kỹ sư phần mềm thực thi chuyên nghiệp.\n"
        f"Nhiệm vụ: Bạn đang thực hiện bước {current_idx + 1}/{len(steps)}: '{current_step}'.\n"
        f"Thư mục làm việc (Workspace): {ws}\n"
        "Hãy sử dụng các công cụ được cung cấp (`read_files`, `write_file`, `list_directory`) để đọc tệp, tìm hiểu cấu trúc và chỉnh sửa mã nguồn cho phù hợp.\n"
        "Khuyên dùng: Khi cần đọc thông tin của nhiều tệp, hãy truyền một danh sách các đường dẫn vào công cụ 'read_files' "
        "để đọc đồng thời, tối ưu hóa lượt gọi và giữ cấu trúc hệ thống gọn gàng. "
        "Khi đã hoàn thành bước này, hãy tổng hợp phản hồi rõ ràng về các hành động bạn đã thực hiện."
    )
    
    react_messages = [SystemMessage(content=system_prompt)]
    if error_logs:
        react_messages.append(HumanMessage(content=f"LƯU Ý SỬA LỖI TỪ LƯỢT CHẠY TRƯỚC:\n{error_logs}")) # type: ignore
    react_messages.append(HumanMessage(content=f"Yêu cầu: Hãy hoàn thành bước này: '{current_step}'")) # type: ignore
    
    modified_files = list(state.get("modified_files", []))
    
    # Vòng lặp ReAct tối đa 5 bước gọi công cụ để đảm bảo hoàn thành nhiệm vụ
    for _ in range(5):
        response = model_with_tools.invoke(react_messages)
        react_messages.append(response) # type: ignore
        
        if not response.tool_calls:
            break
            
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            # Thực thi công cụ thực tế
            if tool_name == "read_files":
                result = read_files.invoke(tool_args)
            elif tool_name == "write_file":
                result = write_file.invoke(tool_args)
                try:
                    # Ghi nhận tệp tin đã bị sửa đổi để tiến hành kiểm thử cú pháp sau này
                    safe_path = sanitize_and_resolve_path(ws, tool_args["file_path"], create_parent=True)
                    if str(safe_path) not in modified_files:
                        modified_files.append(str(safe_path))
                except Exception:
                    pass
            elif tool_name == "list_directory":
                result = list_directory.invoke(tool_args)
            else:
                result = f"Lỗi: Không tìm thấy công cụ '{tool_name}'."
                
            # Đưa kết quả trả về của công cụ vào lịch sử ReAct để LLM đọc
            react_messages.append(ToolMessage(
                content=str(result),
                name=tool_name,
                tool_call_id=tool_call["id"]
            )) # type: ignore
            
    final_output = react_messages[-1].content if isinstance(react_messages[-1], AIMessage) else "Đã thực thi công cụ hoàn tất."
    
    return {
        "messages": [AIMessage(content=f"**Thực thi bước '{current_step}':**\n\n{final_output}")],
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
    task_type = state.get("task_type", "development")
    git_manager = GitManager(ws)
    
    status = git_manager._run_cmd(["git", "status", "--porcelain"])
    
    if status and not status.startswith("ERROR"):
        commit_msg = f"feat(ai): automatic execution ({task_type}) \n\nSteps:\n" + "\n".join(state["plan"])
        git_manager.commit_changes(commit_msg)
        msg = f"Đã hoàn thành yêu cầu và commit các thay đổi lên nhánh `{state['git_branch']}`."
    else:
        msg = "Đã hoàn thành quy trình công việc. Không có thay đổi tệp tin vật lý nào cần commit lên Git."
        
    return {
        "messages": [AIMessage(content=msg)]
    }


# ==========================================
# 8. CẤU HÌNH ĐỊNH TUYẾN ĐỒ THỊ (ROUTERS)
# ==========================================

def start_router(state: AgentState) -> Literal["detect_workspace", "planner"]:
    """
    Router quyết định điểm vào đồ thị:
    - Nếu là Turn 1 (chưa có workspace_path): Đi qua quy trình dò tìm thư mục và dựng git.
    - Nếu là Turn 2+ (đã tồn tại workspace_path): Đi thẳng tới Planner để cập nhật kế hoạch từ hội thoại mới.
    """
    if not state.get("workspace_path"):
        return "detect_workspace"
    return "planner"


def workflow_step_router(state: AgentState) -> Literal["executor", "tester", "commit"]:
    current_idx = state["current_step_idx"]
    plan = state["plan"]
    task_type = state.get("task_type", "development")
    
    if current_idx < len(plan):
        return "executor"
        
    if task_type == "analysis":
        return "commit"
        
    return "tester"


def error_recovery_router(state: AgentState) -> Literal["executor", "commit"]:
    error = state.get("error_logs", "")
    attempts = state.get("attempts", 0)
    
    if error and attempts < 3:
        return "executor"
    return "commit"


# ==========================================
# 9. KẾT NỐI ĐỒ THỊ LANGGRAPH VỚI CHECKPOINTER
# ==========================================

builder = StateGraph(AgentState)

# Đăng ký các Nodes
builder.add_node("detect_workspace", detect_workspace_node)
builder.add_node("git_setup", git_setup_node)
builder.add_node("planner", planner_node)
builder.add_node("executor", executor_node)
builder.add_node("tester", tester_node)
builder.add_node("commit", commit_node)

# Định tuyến ĐỘNG khi bắt đầu khởi động đồ thị (START Router)
builder.add_conditional_edges(
    START,
    start_router,
    {
        "detect_workspace": "detect_workspace",
        "planner": "planner"
    }
)

# Sơ đồ kết nối tuyến tính ban đầu (chỉ kích hoạt ở Turn 1)
builder.add_edge("detect_workspace", "git_setup")
builder.add_edge("git_setup", "planner")
builder.add_edge("planner", "executor")

# Vòng lặp thực thi các bước
builder.add_conditional_edges(
    "executor",
    workflow_step_router,
    {
        "executor": "executor",
        "tester": "tester",
        "commit": "commit"
    }
)

# Vòng lặp sửa lỗi tự động
builder.add_conditional_edges(
    "tester",
    error_recovery_router,
    {
        "executor": "executor",
        "commit": "commit"
    }
)

builder.add_edge("commit", END)

# Sử dụng MemorySaver để kích hoạt tính năng lưu giữ trạng thái đồ thị qua nhiều lượt chat
memory = MemorySaver()
app = builder.compile(checkpointer=memory)