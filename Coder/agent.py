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
    file_paths: Union[str, List[str]] = Field(description="Đường dẫn tương đối hoặc danh sách đường dẫn tương đối của các tệp tin.")

class WriteFileSchema(BaseModel):
    file_path: str = Field(description="Đường dẫn tương đối của tệp tin trong workspace cần ghi hoặc cập nhật.")
    content: str = Field(description="Toàn bộ nội dung tệp tin chi tiết cần lưu xuống đĩa.")

class ListDirSchema(BaseModel):
    sub_dir: str = Field(default=".", description="Đường dẫn tương đối của thư mục cần xem.")


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
    step_findings: List[str]


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
        "Hãy phân tích kỹ lường toàn bộ lịch sử trò chuyện của người dùng để sinh ra kế hoạch hành động tiếp theo thích hợp nhất.\n"
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
        "modified_files": [],
        "attempts": 0,
        "step_findings": [],
        "messages": [AIMessage(content=f"Đã lập kế hoạch hành động (Loại: {task_type.upper()}):\n" + "\n".join([f"- {s}" for s in plan_steps]))]
    }


# ==========================================
# 7.1 NODE THỰC THI KHẢO SÁT (ANALYSIS EXECUTOR) - READ ONLY
# ==========================================
def analysis_executor_node(state: AgentState) -> Dict[str, Any]:
    """
    Node Khảo sát độc lập (Chỉ có quyền đọc, không cấp công cụ write_file để bảo vệ tệp tin).
    """
    ws = state["workspace_path"]
    steps = state["plan"]
    current_idx = state["current_step_idx"]
    
    if current_idx >= len(steps):
         return {"messages": [AIMessage(content="Đã hoàn thành khảo sát toàn bộ các bước.")]}
         
    current_step = steps[current_idx]
    tools_mgr = WorkspaceTools(ws)
    
    @tool(args_schema=ReadFilesSchema)
    def read_files(file_paths: Union[str, List[str]]) -> str:
        """Đọc nội dung của một hoặc nhiều tệp tin trong workspace cùng lúc."""
        return tools_mgr.read_files(file_paths)

    @tool(args_schema=ListDirSchema)
    def list_directory(sub_dir: str = ".") -> str:
        """Liệt kê các tệp và thư mục con trong thư mục chỉ định đệ quy."""
        return tools_mgr.list_directory(sub_dir)
    
    tools = [read_files, list_directory]
    model_with_tools = model.bind_tools(tools)
    
    system_prompt = (
        "Bạn là một kiến trúc sư chuyên khảo sát, đọc hiểu và phân tích cấu trúc mã nguồn (Read-Only Mode).\n"
        f"Nhiệm vụ: Bạn đang thực hiện bước khảo sát {current_idx + 1}/{len(steps)}: '{current_step}'.\n"
        f"Thư mục làm việc: {ws}\n"
        "Hãy sử dụng các công cụ `read_files` và `list_directory` để đọc hiểu cấu trúc hệ thống.\n"
        "YÊU CẦU QUAN TRỌNG:\n"
        "- Bạn chỉ có quyền ĐỌC dữ liệu, tuyệt đối không chỉnh sửa mã nguồn hoặc tự ý tạo tệp tin trong bước này.\n"
        "- Trình bày chi tiết, chuyên nghiệp về kết quả phát hiện được của bạn ở tin nhắn phản hồi cuối cùng."
    )
    
    react_messages = [SystemMessage(content=system_prompt), HumanMessage(content=f"Yêu cầu: Hãy khảo sát bước này: '{current_step}'")]
    
    for _ in range(5):
        response = model_with_tools.invoke(react_messages)
        react_messages.append(response)
        if not response.tool_calls:
            break
            
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            if tool_name == "read_files":
                result = read_files.invoke(tool_args)
            elif tool_name == "list_directory":
                result = list_directory.invoke(tool_args)
            else:
                result = f"Lỗi: Không tìm thấy công cụ '{tool_name}'."
                
            react_messages.append(ToolMessage(content=str(result), name=tool_name, tool_call_id=tool_call["id"]))
            
    final_output = ""
    for msg in reversed(react_messages):
        if isinstance(msg, AIMessage) and msg.content and len(msg.content.strip()) > 15:
            final_output = msg.content.strip()
            break
            
    if not final_output:
        summary_prompt = "Hãy viết một báo cáo tóm tắt chi tiết các kết quả khảo sát bạn đã thu được ở bước này."
        react_messages.append(HumanMessage(content=summary_prompt))
        forced_response = model.invoke(react_messages)
        final_output = forced_response.content
        
    findings = list(state.get("step_findings", []))
    findings.append(f"### Khảo sát bước '{current_step}':\n{final_output}")
    
    return {
        "messages": [AIMessage(content=f"**[Phân tích] Thực thi bước '{current_step}':**\n\n{final_output}")],
        "current_step_idx": current_idx + 1,
        "step_findings": findings
    }


# ==========================================
# 7.2 NODE THỰC THI PHÁT TRIỂN (DEVELOPMENT EXECUTOR) - FULL ACCESS
# ==========================================
def development_executor_node(state: AgentState) -> Dict[str, Any]:
    """
    Node Lập trình phát triển (Toàn quyền đọc/ghi mã nguồn, được cấu hình độc lập hoàn toàn).
    """
    ws = state["workspace_path"]
    steps = state["plan"]
    current_idx = state["current_step_idx"]
    error_logs = state.get("error_logs", "")
    
    if current_idx >= len(steps):
         return {"messages": [AIMessage(content="Đã hoàn thành toàn bộ các bước phát triển.")]}
         
    current_step = steps[current_idx]
    tools_mgr = WorkspaceTools(ws)
    
    # Khai báo các công cụ cục bộ tương tác động dựa trên thư mục hiện tại của Thread
    @tool(args_schema=ReadFilesSchema)
    def read_files(file_paths: Union[str, List[str]]) -> str:
        """Đọc nội dung của một hoặc nhiều tệp tin trong workspace cùng lúc."""
        return tools_mgr.read_files(file_paths)

    @tool(args_schema=WriteFileSchema)
    def write_file(file_path: str, content: str) -> str:
        """Ghi mới hoặc cập nhật nội dung chi tiết của một tệp tin trong workspace."""
        return tools_mgr.write_file(file_path, content)

    @tool(args_schema=ListDirSchema)
    def list_directory(sub_dir: str = ".") -> str:
        """Liệt kê các tệp và thư mục con trong thư mục chỉ định đệ quy."""
        return tools_mgr.list_directory(sub_dir)
    
    # Liên kết các công cụ với mô hình ngôn ngữ lớn
    tools = [read_files, write_file, list_directory]
    model_with_tools = model.bind_tools(tools)
    
    # Thiết lập chuỗi hội thoại ReAct nhỏ bên trong node để thực thi bước hiện tại
    system_prompt = (
        "Bạn là một kỹ sư phần mềm thực thi chuyên nghiệp chuyên sửa lỗi và viết mới mã nguồn (Write-Access Mode).\n"
        f"Nhiệm vụ: Bạn đang thực hiện bước phát triển {current_idx + 1}/{len(steps)}: '{current_step}'.\n"
        f"Thư mục làm việc: {ws}\n"
        "Hãy sử dụng các công cụ `read_files`, `write_file`, và `list_directory` để sửa đổi hoặc tạo mới mã nguồn cho phù hợp.\n"
        "YÊU CẦU QUAN TRỌNG:\n"
        "- Khi chỉnh sửa tệp đã có, luôn luôn dùng `read_files` đọc nội dung trước để hiểu ngữ cảnh và tránh làm mất mã nguồn cũ.\n"
        "- Mô tả thật chi tiết các hành động, giải pháp và vị trí các dòng code bạn đã cập nhật ở tin nhắn phản hồi cuối cùng."
    )
    
    react_messages = [SystemMessage(content=system_prompt)]
    if error_logs:
        react_messages.append(HumanMessage(content=f"LƯU Ý SỬA LỖI TỪ LƯỢT CHẠY TRƯỚC (TESTER BÁO LỖI):\n{error_logs}")) # type: ignore
    react_messages.append(HumanMessage(content=f"Yêu cầu: Hãy thực hiện phát triển bước này: '{current_step}'")) # type: ignore
    
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
                
            react_messages.append(ToolMessage(content=str(result), name=tool_name, tool_call_id=tool_call["id"]))
            
    final_output = ""
    for msg in reversed(react_messages):
        if isinstance(msg, AIMessage) and msg.content and len(msg.content.strip()) > 15:
            final_output = msg.content.strip()
            break
            
    if not final_output:
        summary_prompt = "Hãy viết một tóm tắt chi tiết về mã nguồn bạn đã chỉnh sửa hoặc tạo mới trong bước này."
        react_messages.append(HumanMessage(content=summary_prompt))
        forced_response = model.invoke(react_messages)
        final_output = forced_response.content
        
    return {
        "messages": [AIMessage(content=f"**[Phát triển] Thực thi bước '{current_step}':**\n\n{final_output}")],
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


# ==========================================
# 7.3 NODE TỔNG HỢP TÀI LIỆU CHUYÊN BIỆT (SYNTHESIS NODE)
# ==========================================
def synthesis_node(state: AgentState) -> Dict[str, Any]:
    ws = state["workspace_path"]
    findings = state.get("step_findings", [])
    
    if not findings:
        return {"messages": [AIMessage(content="Không thu thập được thông tin khảo sát để tổng hợp.")]}
        
    compiled_data = "\n\n---\n\n".join(findings)
    
    synthesis_prompt = (
        "Bạn là một Kiến trúc sư Hệ thống cao cấp chuyên biên soạn tài liệu kỹ thuật.\n"
        "Nhiệm vụ của bạn là đọc toàn bộ thông tin khảo sát thô bên dưới và tổng hợp thành một tài liệu 'THONGTIN.md' duy nhất.\n\n"
        "⚠️ YÊU CẦU CỰC KỲ QUAN TRỌNG ĐỂ TRÁNH LỖI TRÀN TOKEN ĐẦU RA (OUTPUT TRUNCATION):\n"
        "- Hãy viết tài liệu thật SÚC TÍCH, CÔ ĐỌNG, tập trung vào cấu trúc, kiến trúc hệ thống và dependencies.\n"
        "- TUYỆT ĐỐI KHÔNG sao chép hay chèn các đoạn mã nguồn dài dòng (ví dụ: các định nghĩa models, controllers, scrapers).\n"
        "- KHÔNG chia nhỏ tài liệu thành các phần rác như 'Phần 1/3', không chèn các chú thích hứa hẹn 'Xem tiếp ở phần sau'. Tài liệu phải hoàn chỉnh trong lượt sinh này.\n"
        "- Sử dụng bảng biểu súc tích thay vì viết các đoạn mô tả dài dòng.\n\n"
        "Tài liệu phải bao gồm các mục tiêu chuẩn:\n"
        "1. Tổng quan & Môi trường phát triển.\n"
        "2. Cấu trúc thư mục chính (tóm tắt súc tích cấu trúc `lib/`).\n"
        "3. Danh sách Dependencies chính và vai trò.\n"
        "4. Mô tả các Module chính, Scrapers & Services.\n"
        "5. Đánh giá nhanh (Điểm mạnh, điểm yếu, đề xuất nâng cấp)."
    )
    
    try:
        response = model.invoke([
            SystemMessage(content=synthesis_prompt),
            HumanMessage(content=f"Dưới đây là toàn bộ thông tin thô đã khảo sát được:\n\n{compiled_data}\n\nHãy tạo tệp 'THONGTIN.md' hoàn chỉnh.")
        ])
        
        md_content = response.content
        if isinstance(md_content, str) and md_content.strip():
            cleaned_md = md_content.strip()
            if cleaned_md.startswith("```markdown"):
                cleaned_md = cleaned_md[11:]
            elif cleaned_md.startswith("```"):
                cleaned_md = cleaned_md[3:]
            if cleaned_md.endswith("```"):
                cleaned_md = cleaned_md[:-3]
            cleaned_md = cleaned_md.strip()
            
            tools_mgr = WorkspaceTools(ws)
            tools_mgr.write_file("THONGTIN.md", cleaned_md)
            
            git_manager = GitManager(ws)
            git_manager._run_cmd(["git", "add", "THONGTIN.md"])
            
            return {
                "messages": [AIMessage(content="**Tổng hợp tài liệu hoàn tất:** Hệ thống đã thu thập tri thức của toàn bộ các bước khảo sát và biên dịch thành tệp `THONGTIN.md` thành công tại thư mục gốc của dự án.")]
            }
    except Exception as e:
        return {
            "messages": [AIMessage(content=f"Cảnh báo: Có lỗi xảy ra trong quá trình tổng hợp tệp THONGTIN.md: {str(e)}")]
        }
    return {}


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


# Router định tuyến sau Node Planner để chuyển hướng trực tiếp vào 2 phân nhánh độc lập
def planner_router(state: AgentState) -> Literal["analysis_executor", "development_executor"]:
    task_type = state.get("task_type", "development")
    if task_type == "analysis":
        return "analysis_executor"
    return "development_executor"


# Router cho phân nhánh Khảo sát (Analysis)
def analysis_router(state: AgentState) -> Literal["analysis_executor", "synthesis"]:
    current_idx = state["current_step_idx"]
    plan = state["plan"]
    if current_idx < len(plan):
        return "analysis_executor"
    return "synthesis"


# Router cho phân nhánh Phát triển (Development)
def development_router(state: AgentState) -> Literal["development_executor", "tester"]:
    current_idx = state["current_step_idx"]
    plan = state["plan"]
    if current_idx < len(plan):
        return "development_executor"
    return "tester"


# Router xử lý phục hồi lỗi tự động sau pha kiểm tra
def tester_router(state: AgentState) -> Literal["development_executor", "commit"]:
    error = state.get("error_logs", "")
    attempts = state.get("attempts", 0)
    if error and attempts < 3:
        return "development_executor"
    return "commit"


# ==========================================
# 9. KẾT NỐI ĐỒ THỊ LANGGRAPH VỚI CHECKPOINTER
# ==========================================

builder = StateGraph(AgentState)

# Đăng ký các Nodes
builder.add_node("detect_workspace", detect_workspace_node)
builder.add_node("git_setup", git_setup_node)
builder.add_node("planner", planner_node)
builder.add_node("analysis_executor", analysis_executor_node)       # Node Khảo sát mới
builder.add_node("development_executor", development_executor_node) # Node Phát triển mới
builder.add_node("tester", tester_node)
builder.add_node("synthesis", synthesis_node)
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

# Rẽ nhánh ĐỘNG ngay sau node Planner dựa trên task_type
builder.add_conditional_edges(
    "planner",
    planner_router,
    {
        "analysis_executor": "analysis_executor",
        "development_executor": "development_executor"
    }
)

# [Vòng lặp nhánh Khảo sát] 
builder.add_conditional_edges(
    "analysis_executor",
    analysis_router,
    {
        "analysis_executor": "analysis_executor",
        "synthesis": "synthesis"
    }
)

# [Vòng lặp nhánh Phát triển]
builder.add_conditional_edges(
    "development_executor",
    development_router,
    {
        "development_executor": "development_executor",
        "tester": "tester"
    }
)

# [Vòng lặp tự động sửa lỗi của nhánh Phát triển]
builder.add_conditional_edges(
    "tester",
    tester_router,
    {
        "development_executor": "development_executor",
        "commit": "commit"
    }
)

# Kết nối các đầu ra của các nhánh về node commit và END
builder.add_edge("synthesis", "commit")
builder.add_edge("commit", END)

# Sử dụng MemorySaver để kích hoạt tính năng lưu giữ trạng thái đồ thị qua nhiều lượt chat
memory = MemorySaver()
app = builder.compile(checkpointer=memory)