# oder/nodes.py
import asyncio
import json
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional,  Tuple, List
from concurrent.futures import ThreadPoolExecutor

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.types import interrupt
from config import find_project_root_heuristic, model, sanitize_and_resolve_path, fast_model
from mcp_helper import run_agent_with_devtools_mcp
from state import AgentState, PlanUpdate, TaskPlan, TaskTriage, Task
from tools import (
    ChromeDevToolsMcpTool, GitManager, ReadFileLinesTool, UniversalSymbolSearchTool, WebInteractAndTestTool, WorkspaceTools, 
    ReadFilesTool, WriteFileTool, ApplyPatchTool, 
    ListDirectoryTool, RunTerminalTool, get_markdown_language
)

def get_text_content_safely(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
        
    elif isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif "text" in item and "type" not in item:
                    text_parts.append(item["text"])
            elif isinstance(item, str):
                text_parts.append(item)
        return " ".join(text_parts).strip()
        
    return ""

def sanitize_llm_response_content(response: AIMessage) -> AIMessage:
    if not response or not isinstance(response, AIMessage):
        return response
        
    content_str = response.content
    if isinstance(content_str, str) and content_str.strip():
        cleaned_content = re.sub(r"<thinking>.*?</thinking>", "", content_str, flags=re.DOTALL)
        cleaned_content = re.sub(r"<thought>.*?</thought>", "", cleaned_content, flags=re.DOTALL)
        response.content = cleaned_content.strip()
        
    return response

def compact_reading_tool_messages(messages: List[BaseMessage]) -> List[BaseMessage]:
    compacted_messages = []
    for msg in messages:
        if msg.type == "tool" and msg.name in ["read_files", "read_file_lines", "search_symbols_universal"]:
            content_str = str(msg.content)
            found_files = []
            
            matches_read = re.findall(r"=== TỆP TIN:\s*[`']?([^`'\n]+)[`']?\s*===", content_str)
            if matches_read:
                found_files.extend(matches_read)
                
            matches_lines = re.findall(r"=== NỘI DUNG TỆP KHOANH VÙNG:\s*([^ \n]+)", content_str)
            if matches_lines:
                found_files.extend(matches_lines)
                
            matches_symbols = re.findall(r"📁\s*`([^`\n]+)`", content_str)
            if matches_symbols:
                found_files.extend(matches_symbols)
                
            file_info = f" của tệp {', '.join([f'`{f}`' for f in found_files])}" if found_files else ""
            
            compacted_msg = ToolMessage(
                content=f"[Đã nạp thành công dữ liệu vật lý{file_info} vào File Registry. Hãy sử dụng cấu trúc mã nguồn cập nhật mới nhất trong System Prompt để làm việc]",
                name=msg.name,
                tool_call_id=msg.tool_call_id,
                id=msg.id
            )
            compacted_messages.append(compacted_msg)
        else:
            compacted_messages.append(msg)
            
    return compacted_messages

def clear_compiler_cache(workspace_path: Path, ext: str):
    try:
        if ext == ".py":
            for pycache in workspace_path.rglob("__pycache__"):
                if pycache.is_dir():
                    shutil.rmtree(pycache, ignore_errors=True)
            for pyc in workspace_path.rglob("*.pyc"):
                if pyc.is_file():
                    pyc.unlink(missing_ok=True)
        elif ext in [".ts", ".tsx", ".js", ".jsx"]:
            ts_cache = workspace_path / "node_modules" / ".cache"
            if ts_cache.exists():
                shutil.rmtree(ts_cache, ignore_errors=True)
    except Exception as e:
        print(f"[Cảnh báo] Không thể dọn dẹp cache biên dịch: {str(e)}")

def find_nearest_config(start_path: Path, config_name: str, max_depth: int = 5) -> Optional[Path]:
    current = start_path.resolve()
    if current.is_file():
        current = current.parent
        
    for _ in range(max_depth):
        target = current / config_name
        if target.exists() and target.is_file():
            return current
        if current.parent == current:
            break
        current = current.parent
    return None

def execute_validation_cmd(cmd: List[str], cwd: Path, timeout: int = 30) -> Tuple[int, str]:
    executable = cmd[0]
    is_windows = platform.system() == "Windows"
    
    resolved_executable = shutil.which(executable)
    if not resolved_executable and is_windows:
        for ext in [".cmd", ".bat", ".exe"]:
            if shutil.which(executable + ext):
                cmd[0] = executable + ext
                resolved_executable = shutil.which(cmd[0])
                break
                
    if not resolved_executable:
        return (-99, f"Cảnh báo: Trình biên dịch/phân tích '{executable}' chưa được cài đặt trên hệ thống.")
        
    try:
        env_copy = os.environ.copy()
        env_copy["PYTHONIOENCODING"] = "utf-8"
        env_copy["PYTHONUTF8"] = "1"
        res = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env_copy, 
            timeout=timeout
        )
        combined_output = (res.stdout or "") + "\n" + (res.stderr or "")
        return (res.returncode, combined_output.strip())
        
    except subprocess.TimeoutExpired:
        return (-2, f"Lỗi: Lệnh kiểm thử '{' '.join(cmd)}' bị treo và vượt quá thời gian chờ.")
    except Exception as e:
        return (-3, f"Lỗi hệ thống khi chạy lệnh kiểm thử: {str(e)}")

def clean_compiler_logs(raw_logs: str) -> str:
    lines = raw_logs.splitlines()
    filtered_lines = []
    error_keywords = ["error", "fail", "exception", "cause", "unhandled", "invalid", "undefined"]
    
    for line in lines:
        clean_line = line.strip()
        if not clean_line:
            continue
        has_error_kw = any(kw in clean_line.lower() for kw in error_keywords)
        has_line_indicator = ":" in clean_line or ".dart" in clean_line or ".py" in clean_line or ".ts" in clean_line
        
        if has_error_kw or has_line_indicator:
            filtered_lines.append(line)
            
    if not filtered_lines:
        if len(lines) > 40:
            return "\n".join(lines[:20] + ["... [Đã lược bớt các dòng ở giữa] ..."] + lines[-20:])
        return raw_logs
        
    return "\n".join(filtered_lines)

def extract_path_from_text(text: str) -> Optional[str]:
    if not text:
        return None
        
    quoted_paths = re.findall(r'["\']([^"\']+)["\']', text)
    for qp in quoted_paths:
        qp_clean = qp.strip()
        if qp_clean:
            try:
                resolved = Path(qp_clean).expanduser().resolve()
                if resolved.exists():
                    return str(resolved)
            except Exception:
                pass

    code_paths = re.findall(r'`([^`]+)`', text)
    for cp in code_paths:
        cp_clean = cp.strip()
        if cp_clean:
            try:
                resolved = Path(cp_clean).expanduser().resolve()
                if resolved.exists():
                    return str(resolved)
            except Exception:
                pass

    words = text.split()
    for word in words:
        cleaned = word.strip('`\'".,;()[]{}*')
        if not cleaned:
            continue
            
        is_path_like = (
            cleaned.startswith('/') or 
            cleaned.startswith('~/') or 
            cleaned.startswith('./') or 
            cleaned.startswith('.\\') or
            (len(cleaned) > 1 and cleaned[1] == ':' and (cleaned[2] == '/' or cleaned[2] == '\\'))
        )
        
        if is_path_like:
            try:
                resolved = Path(cleaned).expanduser().resolve()
                if resolved.exists():
                    return str(resolved)
            except Exception:
                pass
                
    return None


def find_extension_dir_heuristic(workspace_path: Path) -> Optional[str]:
    try:
        for path in workspace_path.rglob("manifest.json"):
            parts_lower = [p.lower() for p in path.parts]
            if not any(black in parts_lower for black in ["node_modules", ".venv", "venv", "env", "build", "dist", ".git"]):
                return str(path.parent.resolve())
    except Exception:
        pass
    return None


def detect_and_triage_node(state: AgentState) -> Dict[str, Any]:
    messages = state["messages"]
    user_msg = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) or getattr(msg, "type", None) == "human":
            user_msg = msg
            break
            
    user_query_text = get_text_content_safely(user_msg.content) if user_msg else ""
    detected_path_str = extract_path_from_text(user_query_text)
    
    workspace_path = None
    is_fallback_workspace = False
    
    if detected_path_str:
        start_path = Path(detected_path_str)
        resolved_root = find_project_root_heuristic(start_path)
        workspace_path = str(resolved_root)
    else:
        is_fallback_workspace = True
        workspace_path = state.get("workspace_path", ".")
        
    triage_res = triage_node(state)
    
    if triage_res is None:
        triage_res = {
            "plan": [],
            "task_type": "development",
            "last_executed_task_ids": [],
            "messages": [AIMessage(content="⚠️ Không thể phân loại tác vụ. Đang chạy ở chế độ mặc định.")],
            "replanning_count": 0,
            "modified_files": [],     
            "error_logs": "",         
            "step_findings": [],      
            "is_simple": True,
            "detailed_analysis": "",
            "extension_path": "",
            "browser_console_logs": ""
        }
        
    is_simple = triage_res.get("is_simple", False)
    
    if is_fallback_workspace:
        base_path = Path(workspace_path).resolve()
        
        if is_simple:
            temp_dir = base_path / "temp"
            if temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception:
                    pass
            temp_dir.mkdir(parents=True, exist_ok=True)
            workspace_path = str(temp_dir)
            
            try:
                root_gitignore = base_path / ".gitignore"
                if root_gitignore.exists():
                    gitignore_content = root_gitignore.read_text(encoding="utf-8")
                    if "temp/" not in gitignore_content:
                        with open(root_gitignore, "a", encoding="utf-8") as f:
                            f.write("\n# AI Generated Sandbox\ntemp/\n")
                else:
                    root_gitignore.write_text("# AI Generated Sandbox\ntemp/\n", encoding="utf-8")
            except Exception:
                pass
        else:
            resolved_root = find_project_root_heuristic(base_path)
            workspace_path = str(resolved_root)
            
    detect_messages = []
    if is_fallback_workspace:
        if is_simple:
            detect_messages.append(
                AIMessage(content=f"ℹ️ Phát hiện tác vụ Sandbox độc lập. Kích hoạt môi trường cách ly tại: `{workspace_path}`.")
            )
        else:
            detect_messages.append(
                AIMessage(content=f"🔍 Không chỉ định đường dẫn. Tự động thiết lập gốc dự án làm việc tại: `{workspace_path}`.")
            )
    else:
        detect_messages.append(
            AIMessage(content=f"🔍 Hệ thống đã phát hiện đường dẫn chỉ định và thiết lập workspace tại: `{workspace_path}`.")
        )
        
    merged_messages = detect_messages + triage_res.get("messages", [])
    
    return {
        "workspace_path": workspace_path,
        "plan": triage_res.get("plan", []),
        "task_type": triage_res.get("task_type", "development"),
        "last_executed_task_ids": triage_res.get("last_executed_task_ids", []),
        "replanning_count": triage_res.get("replanning_count", 0),
        "modified_files": triage_res.get("modified_files", []),
        "error_logs": triage_res.get("error_logs", ""),
        "step_findings": triage_res.get("step_findings", []),
        "is_simple": is_simple,
        "detailed_analysis": triage_res.get("detailed_analysis", ""), 
        "extension_path": triage_res.get("extension_path", ""),
        "browser_console_logs": triage_res.get("browser_console_logs", ""),
        "messages": merged_messages
    }


def get_eligible_tasks(plan: List[Any]) -> List[Any]:
    completed_ids = set()
    for t in plan:
        t_id = t.get("id") if isinstance(t, dict) else getattr(t, "id", None)
        t_status = t.get("status") if isinstance(t, dict) else getattr(t, "status", None)
        if t_status == "completed":
            completed_ids.add(t_id)
            
    eligible = []
    for t in plan:
        t_status = t.get("status") if isinstance(t, dict) else getattr(t, "status", None)
        t_deps = t.get("dependencies", []) if isinstance(t, dict) else (getattr(t, "dependencies", None) or [])
        
        if t_status == "pending":
            if all(dep in completed_ids for dep in t_deps):
                eligible.append(t)
    return eligible


def context_loader_node(state: AgentState) -> Dict[str, Any]:
    ws = state["workspace_path"]
    
    git_dir = Path(ws) / ".git"
    git_branch = "no_git"
    git_msg = "ℹ️ Không phát hiện Git repository. Kích hoạt chế độ Sửa đổi trực tiếp (Bypass Git)."
    
    if git_dir.exists():
        try:
            git_manager = GitManager(ws)
            git_branch = git_manager.init_and_prepare_branch()
            git_msg = f"Đã cấu hình nhánh Git hoạt động: `{git_branch}`"
        except Exception as e:
            git_branch = "no_git"
            git_msg = f"⚠️ Có lỗi xảy ra khi nạp Git ({str(e)}). Tự động chuyển sang chế độ Sửa đổi trực tiếp."
            
    workspace_context = ""
    thongtin_path = Path(ws) / "THONGTIN.md"
    context_msg = "📋 Không tìm thấy tệp cấu hình `THONGTIN.md`."
    
    if thongtin_path.exists():
        try:
            workspace_context = thongtin_path.read_text(encoding="utf-8")
            context_msg = "📋 Đã tải xong ngữ cảnh thông tin dự án từ tệp `THONGTIN.md`."
        except Exception as e:
            workspace_context = f"Lỗi khi đọc file THONGTIN.md: {str(e)}"
            context_msg = f"⚠️ Gặp sự cố khi đọc tệp `THONGTIN.md`: {str(e)}"
            
    ext_dir = find_extension_dir_heuristic(Path(ws))
    ext_msg = ""
    if ext_dir:
        ext_msg = f"\n📦 **[Tự động nhận diện Extension]**: Đã định vị thư mục Chrome Extension tại: `{ext_dir}`"
    else:
        ext_msg = f"\n📦 **[Tự động nhận diện Extension]**: Không tìm thấy manifest.json trực tiếp trong thư mục workspace."
        
    return {
        "workspace_context": workspace_context,
        "git_branch": git_branch,
        "extension_path": ext_dir or "",
        "browser_console_logs": "",
        "messages": [AIMessage(content=f"{git_msg}\n{context_msg}{ext_msg}")]
    }


def triage_node(state: AgentState) -> Dict[str, Any]:
    messages = state["messages"]
    user_msg = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) or getattr(msg, "type", None) == "human":
            user_msg = msg
            break
            
    if not user_msg:
        user_msg = messages[0]
        
    user_query_text = get_text_content_safely(user_msg.content)
    
    structured_llm = fast_model.with_structured_output(TaskTriage, method="function_calling")
    
    system_prompt = (
        "Bạn là một điều phối viên Agent thông minh cấp cao (Triage Supervisor).\n"
        "Nhiệm vụ của bạn là phân tích yêu cầu của người dùng để xác định xem yêu cầu đó nên được xử lý qua luồng Fast-Track (Đơn giản) hay luồng Lập kế hoạch (Phức tạp).\n\n"
        "Đồng thời, hãy viết một bản phân tích chi tiết vào thuộc tính 'detailed_analysis' để định hướng cho các Agent ở các bước sau. Bản phân tích này cần làm rõ:\n"
        "1. Mục tiêu cốt lõi cuối cùng người dùng muốn đạt được.\n"
        "2. Các tệp tin, thư mục hoặc phân hệ mã nguồn cụ thể có thể sẽ bị tác động hoặc cần đọc/sửa đổi.\n"
        "3. Các ràng buộc về logic kỹ thuật, ngôn ngữ trình bày, hoặc các trường hợp biên cần lưu ý.\n"
        "4. Gợi ý sơ bộ về phương pháp thực hiện tối ưu.\n\n"
        "Hãy dựa trên các tiêu chí phân loại nghiêm ngặt sau để phân loại:\n\n"
        "1. KIỂM TRA TÁC VỤ ĐƠN GIẢN (is_simple = True):\n"
        "   - Tương tác Web trực tiếp: Các yêu cầu truy cập website, nhấp nút, điền form, chụp ảnh màn hình hoặc chạy thử JS trên một trang cụ thể (ví dụ: 'vào trang web abc.com và nhấn...').\n"
        "   - Khảo sát hệ thống đơn giản: Đọc nội dung 1-2 tệp tin cụ thể, liệt kê thư mục, hoặc tìm kiếm symbol.\n"
        "   - Thực thi Terminal trực tiếp: Chạy một lệnh terminal đơn lẻ.\n"
        "   - Chỉnh sửa nhỏ: Sửa đổi nhanh chỉ một vài dòng mã hoặc ghi đè một tệp tin ngắn dưới 100 dòng.\n\n"
        "2. KIỂM TRA TÁC VỤ PHỨC TẠP (is_simple = False):\n"
        "   - Phát triển tính năng mới (Feature Development): Yêu cầu viết mới hoặc can thiệp chỉnh sửa logic phức tạp trên nhiều tệp tin nguồn khác nhau.\n"
        "   - Sửa lỗi hệ thống diện rộng (Complex Bug Fixing): Đòi hỏi phải phân tích kiến trúc, tìm kiếm ký hiệu xuyên suốt mã nguồn trước khi sửa đổi.\n"
        "   - Phân tích & Viết tài liệu tổng thể dự án: Khảo sát sâu toàn bộ workspace lớn.\n\n"
        "⚠️ QUY TẮC CHỌN TASK_TYPE (RẤT QUAN TRỌNG):\n"
        "   - Chọn 'analysis' CHỈ KHI yêu cầu thuần túy là đọc hiểu, giải thích cấu trúc mã nguồn, dịch thuật hoặc khảo sát tĩnh dự án (KHÔNG sửa đổi code, KHÔNG chạy lệnh terminal, và KHÔNG tương tác/chạy thử nghiệm trình duyệt web).\n"
        "   - BẮT BUỘC CHỌN 'development' cho mọi trường hợp còn lại, bao gồm: Có viết/sửa code, chạy lệnh terminal, HOẶC cần khởi chạy trình duyệt web thật (Dynamic Web Testing) để nạp extension, tương tác web, kiểm tra hành vi runtime của ứng dụng."
    )
    
    try:
        triage_output = structured_llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query_text}
        ])
        is_simple = getattr(triage_output, "is_simple", False)
        task_type = getattr(triage_output, "task_type", "development")
        detailed_analysis = getattr(triage_output, "detailed_analysis", "")
    except Exception:
        is_simple = False
        task_type = "development"
        detailed_analysis = "Không thể phân tích tự động mục tiêu yêu cầu của người dùng."

    plan = []
    if is_simple:
        plan = [
            Task(
                id="T1",
                description=f"Thực hiện trực tiếp yêu cầu: {user_query_text}",
                dependencies=[],
                status="pending"
            )
        ]
    else:
        plan = [
            Task(
                id="T_SURVEY",
                description="Khảo sát cấu trúc thư mục, tệp cấu hình manifest.json và mã nguồn chính của dự án bằng các công cụ khảo sát để nắm vững kiến trúc trước khi lập kế hoạch.",
                dependencies=[],
                status="pending"
            )
        ]
        task_type = "analysis" 
        
    return {
        "plan": plan,
        "task_type": task_type,
        "is_simple": is_simple,
        "detailed_analysis": detailed_analysis,
        "messages": [
            AIMessage(
                content=f"📊 **[Phân loại tác vụ]**: Hệ thống xác định yêu cầu thuộc diện "
                        f"{'ĐƠN GIẢN (Fast-Track)' if is_simple else 'PHỨC TẠP (Multi-Step Discovery)'} | "
                        f"Pha hoạt động khởi động: `{task_type.upper()}`.\n\n"
                        f"🎯 **[Phân tích mục tiêu]**:\n{detailed_analysis}"
            )
        ]
    }
def chrome_extension_debugger_node(state: AgentState) -> Dict[str, Any]:
    """
    Nút xử lý gỡ lỗi chuyên sâu sử dụng Chrome DevTools MCP.
    """
    user_query = "Hãy kiểm tra xem trang web hiện tại có phát sinh lỗi console hoặc lỗi mạng nào liên quan đến Extension của tôi không."
    
    # Chạy tác vụ bất đồng bộ từ luồng đồng bộ của LangGraph
    loop = asyncio.get_event_loop()
    if loop.is_running():
        # Tránh xung đột nếu Event Loop đang chạy
        import nest_asyncio
        nest_asyncio.apply()
        
    debug_result = asyncio.run(
        run_agent_with_devtools_mcp(
            model=model,
            prompt_message=user_query,
            chat_history=state.get("messages", [])
        )
    )
    
    return {
        "messages": [AIMessage(content=f"📋 **[Kết quả kiểm tra DevTools]**:\n\n{debug_result}")]
    }

def executor_node(state: AgentState) -> Dict[str, Any]:
    ws = state["workspace_path"]
    plan = state["plan"]
    error_logs = state.get("error_logs", "")
    workspace_context = state.get("workspace_context", "")
    file_registry = state.get("file_registry", {})
    messages = list(state["messages"])
    task_type = state.get("task_type", "development")
    extension_path = state.get("extension_path", "")
    
    eligible_tasks = get_eligible_tasks(plan)
    if not eligible_tasks:
        pending_tasks = [t for t in plan if (t.get("status") if isinstance(t, dict) else getattr(t, "status", None)) == "pending"]
        if pending_tasks:
            eligible_tasks = [pending_tasks[0]]
        else:
            return {"messages": [AIMessage(content=f"Đã hoàn thành khảo sát toàn bộ các bước {task_type}.")]}
         
    tasks_str = "\n".join([
        f"- [{getattr(t, 'id', None) or t.get('id')}] {getattr(t, 'description', None) or t.get('description')}"
        for t in eligible_tasks
    ])

    # =====================================================================
    # GIẢI PHÁP: TRÍCH XUẤT FILE REGISTRY LÀM SINGLE SOURCE OF TRUTH CHUNG
    # =====================================================================
    registry_context_str = ""
    if file_registry:
        registry_context_str = "\n=== 📦 NỘI DUNG MÃ NGUỒN CẬP NHẬT MỚI NHẤT (SINGLE SOURCE OF TRUTH) ===\n"
        for file_path, content in file_registry.items():
            lang = get_markdown_language(file_path)
            lines = content.splitlines()
            formatted_lines = [f"{idx+1:04d} | {line}" for idx, line in enumerate(lines)]
            
            registry_context_str += (
                f"\n--- TỆP TIN: `{file_path}` ---\n"
                f"```{lang}\n" + "\n".join(formatted_lines) + "\n```\n"
            )

    if task_type == "analysis":
        read_files = ReadFilesTool(workspace_path=ws)
        list_directory = ListDirectoryTool(workspace_path=ws)
        search_symbols = UniversalSymbolSearchTool(workspace_path=ws)
        read_file_lines = ReadFileLinesTool(workspace_path=ws)
        tools = [read_files, list_directory, search_symbols, read_file_lines]

        previous_findings_str = ""
        existing_findings = state.get("step_findings", [])
        if existing_findings:
            previous_findings_str = "\n\n--- CÁC KẾT QUẢ KHẢO SÁT BẠN ĐÃ THU THẬP ĐƯỢC Ở CÁC BƯỚC TRƯỚC ---\n" + "\n\n".join(existing_findings)
        
        system_prompt = (
            "Bạn là một kiến trúc sư chuyên khảo sát, đọc hiểu và phân tích cấu trúc mã nguồn (Read-Only Mode).\n"
            f"Nhiệm vụ: Bạn đang thực hiện đồng thời các nhiệm vụ sau:\n{tasks_str}\n"
            f"Thư mục làm việc: {ws}\n"
        )
        if workspace_context:
            system_prompt += f"\n--- TỔNG QUAN VỀ DỰ ÁN (THONGTIN.md) ---\n{workspace_context}\n"
            
        # 🔔 CẬP NHẬT: Nạp registry mã nguồn vào System Prompt của chế độ Khảo sát (Analysis Mode)
        if registry_context_str:
            system_prompt += registry_context_str
            
        system_prompt += (
            "\nHãy sử dụng các công cụ khảo sát cấu trúc hệ thống.\n"
            "\n⚠️ RÀNG BUỘC PHẠM VI NGHIÊM NGẶT (BẮT BUỘC):\n"
            "1. Bạn chỉ có quyền ĐỌC dữ liệu, tuyệt đối không chỉnh sửa mã nguồn hoặc tự ý tạo tệp tin trong bước này.\n"
            "2. KHÔNG ĐƯỢC PHÉP tự ý định dạng tài liệu báo cáo hoàn chỉnh, tổng hợp tri thức hay viết tệp THONGTIN.md. "
            "Nhiệm vụ của bạn chỉ là thu thập thông tin thô, liệt kê các phát hiện kỹ thuật (raw findings) thực tế. "
            "Việc tổng hợp chúng thành một tài liệu THONGTIN.md hoàn chỉnh là nhiệm vụ ĐỘC QUYỀN của node 'synthesis' tiếp theo."
        )
        if previous_findings_str:
            system_prompt += previous_findings_str

    else:  # development
        web_interact_tool = WebInteractAndTestTool(workspace_path=ws)
        # Khởi tạo công cụ Chrome DevTools MCP mới
        chrome_devtools_tool = ChromeDevToolsMcpTool(workspace_path=ws) # <--- THÊM DÒNG NÀY
        
        read_files = ReadFilesTool(workspace_path=ws)
        write_file = WriteFileTool(workspace_path=ws)
        apply_patch = ApplyPatchTool(workspace_path=ws)
        search_symbols = UniversalSymbolSearchTool(workspace_path=ws)
        list_directory = ListDirectoryTool(workspace_path=ws)
        run_terminal_command = RunTerminalTool(workspace_path=ws)
        read_file_lines = ReadFileLinesTool(workspace_path=ws)
        
        # Bổ sung chrome_devtools_tool vào danh sách tools dưới đây:
        tools = [
            read_files, write_file, apply_patch, list_directory, 
            run_terminal_command, search_symbols, read_file_lines, 
            web_interact_tool, 
            chrome_devtools_tool  # <--- THÊM DÒNG NÀY
        ]
        system_prompt = (
            "Bạn là một kỹ sư phần mềm thực thi chuyên nghiệp chuyên sửa lỗi và viết mới mã nguồn (Write-Access Mode).\n"
            f"Nhiệm vụ: Bạn đang thực hiện đồng thời các nhiệm vụ sau:\n{tasks_str}\n"
            f"Thư mục làm việc: {ws}\n"
        )
        if workspace_context:
            system_prompt += f"\n--- TỔNG QUAN VỀ DỰ ÁN (THONGTIN.md) ---\n{workspace_context}\n"
        
        if extension_path:
            system_prompt += f"\nℹ️ **[Phát hiện Chrome Extension]**: Thư mục Extension đã được định vị tại: `{extension_path}`. " \
                             f"Khi kiểm thử động trang web, hãy luôn truyền tham số `extension_path` này vào công cụ `web_interact_and_test` " \
                             f"để trình duyệt tự động nạp Extension của bạn.\n"

        # Nạp registry mã nguồn vào System Prompt của chế độ Phát triển (Development Mode)
        if registry_context_str:
            system_prompt += registry_context_str
            
        system_prompt += (
            "\n⚠️ QUY TẮC PHẠM VI VÀ PHÒNG TRÁNH LỆCH DÒNG (BẮT BUỘC):\n"
            "1. Bạn đã được cung cấp nguồn mã nguồn mới nhất (đã đánh số dòng chi tiết) trong mục 'SINGLE SOURCE OF TRUTH' ở trên.\n"
            "2. ĐÂY LÀ NỘI DUNG MỚI NHẤT VÀ CHÍNH XÁC NHẤT. Hãy luôn sử dụng mốc dòng và nội dung từ mục này để thiết lập khối SEARCH-AND-REPLACE cho công cụ `apply_search_replace_patch`.\n"
            "3. Nếu bạn vừa sửa đổi một file ở bước trước, nội dung file đó trong mục 'SINGLE SOURCE OF TRUTH' đã được cập nhật tự động. Bạn không cần phải gọi lại công cụ đọc trừ khi muốn kiểm tra sâu hơn.\n"
            "\n⚠️ QUY TẮC SỬ DỤNG CÔNG CỤ:\n"
            "1. Đối với file trên 300 dòng: BẮT BUỘC dùng `apply_search_replace_patch` để áp dụng bản vá, cấm ghi đè bừa bãi.\n"
            "2. Công cụ `write_file` chỉ dùng khi tạo mới hoặc sửa các tệp ngắn dưới 300 dòng.\n"
            "3. NGHIÊM CẤM thực hiện chạy các bộ kiểm thử tự động (như pytest, cargo test, dart test, npm test, vitest) bằng công cụ `run_terminal_command`."
        )

    model_with_tools = model.bind_tools(tools)
    optimized_history = compact_reading_tool_messages(messages)
    
    input_messages = [SystemMessage(content=system_prompt)]
    if task_type == "development" and error_logs:
        input_messages.append(HumanMessage(content=f"LƯU Ý SỬA LỖI TỪ LƯỢT CHẠY TRƯỚC:\n{error_logs}\nHãy sửa triệt để."))
        
    response = model_with_tools.invoke(input_messages + optimized_history)
    response = sanitize_llm_response_content(response)
    
    if not response.tool_calls:
        findings = []
        if task_type == "analysis" and response.content:
            tasks_ids = ", ".join([str(getattr(t, "id", None) or t.get('id')) for t in eligible_tasks])
            findings = [f"### Kết quả khảo sát các nhiệm vụ ({tasks_ids}):\n{response.content}"]
            
        updated_plan = []
        eligible_ids = {t.get("id") if isinstance(t, dict) else getattr(t, "id", None) for t in eligible_tasks}
        for t in plan:
            if isinstance(t, dict):
                t_copy = dict(t)
                if t_copy["id"] in eligible_ids:
                    t_copy["status"] = "completed"
            else:
                t_copy = t.model_copy()
                if t_copy.id in eligible_ids:
                    t_copy.status = "completed"
            updated_plan.append(t_copy)

        ret_dict = {
            "messages": [response],
            "plan": updated_plan,
            "last_executed_task_ids": list(eligible_ids)
        }
        if task_type == "analysis" and findings:
            ret_dict["step_findings"] = findings
            
        return ret_dict
    else:
        return {"messages": [response]}


def replanner_node(state: AgentState) -> Dict[str, Any]:
    replanning_count = state.get("replanning_count", 0)
    ws = state["workspace_path"]
    plan = state["plan"]
    messages = state["messages"]
    task_type = state.get("task_type", "development")
    workspace_context = state.get("workspace_context", "")
    error_logs = state.get("error_logs", "")
    
    is_survey_transition = (
        len(plan) == 1 and 
        (plan[0].id if isinstance(plan[0], Task) else plan[0].get("id")) == "T_SURVEY" and
        (plan[0].status if isinstance(plan[0], Task) else plan[0].get("status")) == "completed"
    )
    
    if replanning_count >= 5 or (not error_logs and not is_survey_transition):
        action_msg = "bypass_limit" if replanning_count >= 5 else "bypass_no_error"
        proposal_message = AIMessage(
            content=json.dumps({"action": action_msg}, ensure_ascii=False),
            name="replanner_proposal"
        )
        return {
            "messages": [proposal_message]
        }
    
    plan_str = "\n".join([
        f"- [{t.id if isinstance(t, Task) else t.get('id')}] {t.description if isinstance(t, Task) else t.get('description')} "
        f"(Trạng thái: {t.status if isinstance(t, Task) else t.get('status')}, "
        f"Phụ thuộc: {t.dependencies if isinstance(t, Task) else t.get('dependencies')})"
        for t in plan
    ])
    
    if is_survey_transition:
        system_prompt = (
            "Bạn là một Kiến trúc sư kiêm Điều phối viên dự án phần mềm cấp cao.\n"
            f"Nhiệm vụ: Dựa trên dữ liệu khảo sát và thám thính dự án vừa qua tại '{ws}' (ở các tin nhắn trước), "
            "hãy thiết kế một lộ trình hành động (DAG updated_tasks) hoàn chỉnh để giải quyết trọn vẹn yêu cầu của người dùng.\n\n"
            "⚠️ QUY TẮC THIẾT KẾ KẾ HOẠCH PHÁT TRIỂN & KIỂM THỬ (BẮT BUỘC):\n"
            "1. Đặt `should_modify_plan` là True để áp dụng kế hoạch mới.\n"
            "2. Giữ nguyên nhiệm vụ 'T_SURVEY' với trạng thái là 'completed'.\n"
            "3. Bổ sung các nhiệm vụ mới (ví dụ: T1, T2...) mô tả chính xác các file cần xem, các file cần sửa dựa trên dữ liệu thật thu được từ pha khảo sát.\n"
            "4. TUYỆT ĐỐI BẮT BUỘC phải lập kế hoạch cho nhiệm vụ 'Kiểm thử tích hợp động trên trình duyệt thật' sử dụng công cụ `web_interact_and_test` "
            "để trực tiếp nạp Extension, truy cập trang web đích (ví dụ: abc.com) và xác thực hành vi của Extension làm bước cuối cùng trong kế hoạch!\n"
            "5. Đặt `task_type` là 'development' (vì chúng ta sẽ sửa code và chạy trình duyệt kiểm thử)."
        )
    else:
        system_prompt = (
            "Bạn là một Kiến trúc sư kiêm Điều phối viên dự án phần mềm cấp cao.\n"
            f"Nhiệm vụ: Đánh giá tiến trình thực thi kế hoạch tại thư mục làm việc '{ws}'.\n\n"
            "Hệ thống vừa phát hiện lỗi nghiêm trọng không thể tự gỡ lỗi ở cấp độ cục bộ.\n"
            "Hãy đề xuất một kế hoạch điều chỉnh (được cập nhật trong updated_tasks) để giải quyết triệt để lỗi này.\n\n"
            "⚠️ QUY TẮC CẬP NHẬT KẾ HOẠCH CHO PRODUCTION (BẮT BUỘC):\n"
            "1. Đặt `should_modify_plan` là True và cập nhật danh sách nhiệm vụ trong `updated_tasks` để giải quyết vấn đề.\n"
            "2. ĐỐI VỚI CÁC NHIỆM VỤ ĐÃ HOÀN THÀNH (status: 'completed'): Bắt buộc giữ nguyên ID, mô tả và trạng thái là 'completed'.\n"
            "3. Kế hoạch cập nhật của bạn chỉ tập trung hoàn toàn vào các bước thực thi khảo sát vật lý (analysis) hoặc sửa đổi mã nguồn (development)."
        )
    
    if workspace_context:
        system_prompt += f"\n\n--- NGỮ CẢNH HỆ THỐNG (THONGTIN.md) ---\n{workspace_context}"
        
    user_prompt = f"Kế hoạch hiện tại:\n{plan_str}\n\n"
    if error_logs:
        user_prompt += f"🚨 LỖI BIÊN DỊCH CẦN SỬA ĐỔI KẾ HOẠCH:\n{error_logs}\n\n"
    user_prompt += "Hãy đưa ra phân tích và đề xuất cập nhật kế hoạch phù hợp thông qua cuộc gọi hàm."
    
    structured_llm = model.with_structured_output(PlanUpdate, method="function_calling")
    
    try:
        decision = structured_llm.invoke([
            {"role": "system", "content": system_prompt},
            *messages,
            {"role": "user", "content": user_prompt}
        ])
        
        should_modify = getattr(decision, "should_modify_plan", False)
        explanation = getattr(decision, "explanation", "")
        updated_tasks = getattr(decision, "updated_tasks", plan)
        updated_task_type = getattr(decision, "task_type", task_type)
        
        old_completed_tasks = {
            (t.id if isinstance(t, Task) else t.get("id")): t 
            for t in plan 
            if (t.status if isinstance(t, Task) else t.get("status")) == "completed"
        }
        
        refined_tasks = []
        seen_ids = set()
        for task_data in updated_tasks:
            task_obj = task_data if isinstance(task_data, Task) else Task(**task_data)
            t_id = task_obj.id
            if t_id in seen_ids:
                t_id = f"{t_id}_alt_{len(seen_ids)}"
                task_obj.id = t_id
            seen_ids.add(t_id)
            
            if t_id in old_completed_tasks:
                task_obj.status = "completed"
                old_task = old_completed_tasks[t_id]
                task_obj.description = old_task.description if isinstance(old_task, Task) else old_task.get("description")
            refined_tasks.append(task_obj)
            
        proposal_data = {
            "action": "propose",
            "explanation": explanation,
            "task_type": updated_task_type,
            "tasks": [t.model_dump() if hasattr(t, "model_dump") else t for t in refined_tasks]
        }
        
        proposal_message = AIMessage(
            content=json.dumps(proposal_data, ensure_ascii=False),
            name="replanner_proposal"
        )
        
        explanation_message = AIMessage(
            content=f"🔄 **[Đề xuất lộ trình hành động dựa trên kết quả khảo sát]**\n\n{explanation}\n\nHệ thống đang chờ bạn phê duyệt hoặc tinh chỉnh..."
        )
        
        return {
            "replanning_count": replanning_count + 1,
            "messages": [proposal_message, explanation_message]
        }
        
    except Exception as e:
        proposal_message = AIMessage(
            content=json.dumps({"action": "bypass_error", "error": str(e)}, ensure_ascii=False),
            name="replanner_proposal"
        )
        return {
            "messages": [proposal_message]
        }


def replanner_interrupt_node(state: AgentState) -> Dict[str, Any]:
    messages = state["messages"]
    plan = state["plan"]
    task_type = state.get("task_type", "development")
    
    proposal_msg = None
    for msg in reversed(messages):
        if getattr(msg, "name", None) == "replanner_proposal":
            proposal_msg = msg
            break
            
    if not proposal_msg:
        return {}
        
    try:
        proposal_data = json.loads(proposal_msg.content)
    except Exception:
        return {}
        
    if proposal_data.get("action") in ["bypass_limit", "bypass_no_error", "bypass_error"]:
        return {
            "error_logs": "",
            "attempts": 0,
            "modified_files": [],
            "messages": [AIMessage(content="🔄 [Bypass Replanner] Tiếp tục lộ trình thực thi hiện hành.")]
        }
        
    payload = {
        "title": "📋 ĐÁNH GIÁ & PHÊ DUYỆT KẾ HOẠCH HÀNH ĐỘNG",
        "explanation": proposal_data["explanation"],
        "proposed_tasks": proposal_data["tasks"],
        "prompt": (
            "Hệ thống đề xuất điều chỉnh lộ trình như trên.\n"
            "- Gửi phản hồi 'yes' hoặc rỗng để ĐỒNG Ý áp dụng kế hoạch mới.\n"
            "- Gửi phản hồi 'skip' hoặc 'no' để BỎ QUA việc lập kế hoạch lại (Agent sẽ tiếp tục chạy kế hoạch cũ của các bước chưa hoàn thành).\n"
            "- Gửi danh sách nhiệm vụ JSON tự chỉnh sửa để áp dụng kế hoạch thủ công."
        )
    }
    
    user_input = interrupt(payload)
    
    if isinstance(user_input, str):
        user_input_clean = user_input.strip().lower()
        
        if user_input_clean in ["skip", "no", "cancel"]:
            return {
                "error_logs": "",           
                "attempts": 0,
                "modified_files": [],
                "messages": [AIMessage(content="⏭️ **[Người dùng chọn đi tiếp]** Đã bỏ qua đợt lập kế hoạch lại theo ý muốn. Chuyển sang nhiệm vụ tiếp theo.")]
            }
            
        elif user_input_clean in ["yes", "approve", "ok", ""]:
            refined_tasks = [Task(**t) for t in proposal_data["tasks"]]
            return {
                "plan": refined_tasks,
                "task_type": proposal_data["task_type"],
                "error_logs": "",
                "attempts": 0,
                "modified_files": [],
                "messages": [AIMessage(content="✅ **[Kế hoạch được duyệt]** Người dùng đã phê duyệt kế hoạch chỉnh sửa.")]
            }
            
        else:
            try:
                parsed_tasks = json.loads(user_input)
                if isinstance(parsed_tasks, list):
                    custom_tasks = [Task(**t) for t in parsed_tasks]
                    return {
                        "plan": custom_tasks,
                        "error_logs": "",
                        "attempts": 0,
                        "modified_files": [],
                        "messages": [AIMessage(content="✏️ **[Điều chỉnh thủ công]** Đã áp dụng kế hoạch tùy chỉnh do người dùng thiết lập.")]
                    }
            except Exception:
                pass
            
            return {
                "error_logs": "",
                "attempts": 0,
                "modified_files": [],
                "messages": [AIMessage(content=f"⏭️ **[Đi tiếp với lưu ý]** Đã ghi nhận phản hồi của bạn: '{user_input}'. Giữ lộ trình cũ.")]
            }
            
    elif isinstance(user_input, dict):
        action = user_input.get("action", "approve")
        if action in ["skip", "no"]:
            return {
                "error_logs": "",
                "attempts": 0,
                "modified_files": [],
                "messages": [AIMessage(content="⏭️ **[Người dùng chọn đi tiếp]** Bỏ qua đợt lập kế hoạch lại.")]
            }
            
        if "tasks" in user_input:
            try:
                custom_tasks = [Task(**t) for t in user_input["tasks"]]
                return {
                    "plan": custom_tasks,
                    "task_type": user_input.get("task_type", task_type),
                    "error_logs": "",
                    "attempts": 0,
                    "modified_files": [],
                    "messages": [AIMessage(content="✏️ **[Đã cập nhật kế hoạch]** Áp dụng thiết kế kế hoạch thủ công từ UI.")]
                }
            except Exception as e:
                refined_tasks = [Task(**t) for t in proposal_data["tasks"]]
                return {
                    "plan": refined_tasks,
                    "task_type": proposal_data["task_type"],
                    "error_logs": "",
                    "attempts": 0,
                    "modified_files": [],
                    "messages": [AIMessage(content=f"⚠️ Định dạng kế hoạch tùy chỉnh không hợp lệ ({str(e)}). Tự động sử dụng bản đề xuất của AI.")]
                }

    refined_tasks = [Task(**t) for t in proposal_data["tasks"]]
    return {
        "plan": refined_tasks,
        "task_type": proposal_data["task_type"],
        "error_logs": "",
        "attempts": 0,
        "modified_files": [],
        "messages": [AIMessage(content="✅ **[Tự động duyệt]** Đồng ý kế hoạch điều chỉnh.")]
    }


def tool_node(state: AgentState) -> Dict[str, Any]:
    ws = state["workspace_path"]
    read_files = ReadFilesTool(workspace_path=ws)
    write_file = WriteFileTool(workspace_path=ws)
    apply_patch = ApplyPatchTool(workspace_path=ws)
    list_directory = ListDirectoryTool(workspace_path=ws)
    run_terminal_command = RunTerminalTool(workspace_path=ws)
    search_symbols = UniversalSymbolSearchTool(workspace_path=ws)
    read_file_lines = ReadFileLinesTool(workspace_path=ws)
    web_interact_tool = WebInteractAndTestTool(workspace_path=ws)
    chrome_devtools_tool = ChromeDevToolsMcpTool(workspace_path=ws)
    tools_map = {
        "read_files": read_files,
        "write_file": write_file,
        "apply_search_replace_patch": apply_patch,
        "list_directory": list_directory,
        "run_terminal_command": run_terminal_command,
        "search_symbols_universal": search_symbols,
        "read_file_lines": read_file_lines,
        "web_interact_and_test": web_interact_tool,
         "chrome_devtools_mcp_tool": chrome_devtools_tool 
    }
    
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {}
        
    tool_messages = []
    modified_files = list(state.get("modified_files", []))
    file_registry = dict(state.get("file_registry", {}))
    
    impacted_files = set()
    
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"] or {}
        tool_id = tool_call["id"]
        
        raw_path = tool_args.get("file_path") or tool_args.get("file_paths")
        if raw_path:
            if isinstance(raw_path, list):
                impacted_files.update(raw_path)
            else:
                impacted_files.add(str(raw_path))
        
        tool_instance = tools_map.get(tool_name)
        if not tool_instance:
            result = f"Lỗi: Không tìm thấy công cụ '{tool_name}'."
        else:
            try:
                result = tool_instance.invoke(tool_args)
                if tool_name in ["write_file", "apply_search_replace_patch"] and "Lỗi" not in str(result):
                    if raw_path and not isinstance(raw_path, list):
                        try:
                            safe_path = sanitize_and_resolve_path(ws, raw_path, create_parent=True)
                            if str(safe_path) not in modified_files:
                                modified_files.append(str(safe_path))
                        except Exception:
                            pass
            except Exception as e:
                result = f"Lỗi thực thi công cụ '{tool_name}': {str(e)}"
                
        tool_messages.append(ToolMessage(content=str(result), name=tool_name, tool_call_id=tool_id))
        
    for file_path in impacted_files:
        try:
            safe_path = sanitize_and_resolve_path(ws, file_path, create_parent=False)
            if safe_path.exists() and safe_path.is_file():
                current_content = safe_path.read_text(encoding="utf-8")
                file_registry[file_path] = current_content
        except Exception:
            pass
            
    return {
        "messages": tool_messages,
        "modified_files": modified_files,
        "file_registry": file_registry
    }


def tester_node(state: AgentState) -> Dict[str, Any]:
    modified_files = state.get("modified_files", [])
    attempts = state.get("attempts", 0)
    plan = state["plan"]
    ws = state["workspace_path"]
    last_executed_ids = state.get("last_executed_task_ids", [])
    
    errors = []
    warnings = []
    workspace_root = Path(ws).expanduser().resolve()
    
    files_by_ext: Dict[str, List[Path]] = {}
    for f_path_str in modified_files:
        try:
            p = Path(f_path_str).resolve()
            if p.exists() and p.is_file():
                ext = p.suffix.lower()
                files_by_ext.setdefault(ext, []).append(p)
                clear_compiler_cache(workspace_root, ext)
        except Exception:
            pass

    for ext, files in files_by_ext.items():
        if ext == ".py":
            import sys
            for f in files:
                if f.name.startswith("test_") or f.name.endswith("_test.py"):
                    code, output = execute_validation_cmd([sys.executable, str(f)], workspace_root)
                    if code == -99:
                        warnings.append(output)
                    elif code != 0:
                        errors.append(f"❌ [Lỗi Thực Thi Unit Test Python] tại tệp `{f.name}`:\n{clean_compiler_logs(output)}")
                else:
                    code, output = execute_validation_cmd([sys.executable, "-m", "py_compile", str(f)], workspace_root)
                    if code == -99:
                        warnings.append(output)
                    elif code != 0:
                        errors.append(f"❌ [Lỗi Cú Pháp Python] tại tệp `{f.name}`:\n{clean_compiler_logs(output)}")

        elif ext == ".dart":
            for f in files:
                target_dir = find_nearest_config(f, "pubspec.yaml") or workspace_root
                code, output = execute_validation_cmd(["dart", "analyze"], target_dir)
                if code == -99:
                    warnings.append(f"{output} (Bỏ qua kiểm tra tĩnh cho `{f.name}`)")
                elif code != 0:
                    errors.append(f"❌ [Lỗi Dart Analysis] tại sub-project `{target_dir.name}`:\n{clean_compiler_logs(output)}")
                    break

        elif ext in [".ts", ".tsx", ".js", ".jsx"]:
            for f in files:
                target_dir = find_nearest_config(f, "package.json") or workspace_root
                if ext in [".ts", ".tsx"]:
                    cmd = ["npx", "tsc", "--noEmit", "--skipLibCheck"]
                    code, output = execute_validation_cmd(cmd, target_dir)
                    if code == -99:
                        warnings.append(f"{output} (Bỏ qua phân tích kiểu dữ liệu cho `{f.name}`)")
                    elif code != 0:
                        errors.append(f"❌ [Lỗi TypeScript Compile] tại `{target_dir.name}`:\n{clean_compiler_logs(output)}")
                        break

        elif ext == ".rs":
            for f in files:
                target_dir = find_nearest_config(f, "Cargo.toml") or workspace_root
                code, output = execute_validation_cmd(["cargo", "check"], target_dir)
                if code == -99:
                    warnings.append(f"{output} (Bỏ qua biên dịch Rust cho `{f.name}`)")
                elif code != 0:
                    errors.append(f"❌ [Lỗi Biên Dịch Rust] tại `{target_dir.name}`:\n{clean_compiler_logs(output)}")
                    break

        elif ext == ".go":
            for f in files:
                target_dir = find_nearest_config(f, "go.mod") or workspace_root
                code, output = execute_validation_cmd(["go", "vet", "./..."], target_dir)
                if code == -99:
                    warnings.append(f"{output} (Bỏ qua kiểm tra tĩnh Go cho `{f.name}`)")
                elif code != 0:
                    errors.append(f"❌ [Lỗi Tĩnh Go Vet] tại `{target_dir.name}`:\n{clean_compiler_logs(output)}")
                    break

    warning_msg = ""
    if warnings:
        warning_msg = "⚠️ **Cảnh báo môi trường:**\n" + "\n".join([f"- {w}" for w in warnings]) + "\n\n"

    if errors:
        combined_error = "\n\n---\n\n".join(errors)
        
        if attempts < 3:
            updated_plan = []
            for t in plan:
                if isinstance(t, dict):
                    t_copy = dict(t)
                    if t_copy["id"] in last_executed_ids:
                        t_copy["status"] = "pending"
                else:
                    t_copy = t.model_copy()
                    if t_copy.id in last_executed_ids:
                        t_copy.status = "pending"
                updated_plan.append(t_copy)
                
            return {
                "error_logs": combined_error,
                "attempts": attempts + 1,
                "plan": updated_plan,
                "messages": [AIMessage(content=f"{warning_msg}⚠️ [Vòng kiểm thử thất bại] Phát hiện lỗi ở mã nguồn sửa đổi:\n\n{combined_error}\n\n⚙️ Đang gửi trả trạng thái nhiệm vụ về 'pending' để tự động sửa chữa.")]
            }
        else:
            return {
                "error_logs": "",
                "attempts": 0,
                "modified_files": [],
                "messages": [AIMessage(content=f"{warning_msg}❌ Đã vượt quá giới hạn số lần sửa lỗi tự động. Hệ thống sẽ bỏ qua lỗi để tiếp tục tiến trình.")]
            }
                
    success_content = f"{warning_msg}✅ [Vòng kiểm thử thành công] Toàn bộ mã nguồn đã vượt qua kiểm tra tĩnh và biên dịch."
    return {
        "error_logs": "",
        "attempts": 0,
        "modified_files": [],
        "messages": [AIMessage(content=success_content)]
    }


def synthesis_node(state: AgentState) -> Dict[str, Any]:
    ws = state["workspace_path"]
    findings = state.get("step_findings", [])
    git_branch = state.get("git_branch", "no_git")
    
    if not findings:
        return {"messages": [AIMessage(content="Không thu thập được thông tin khảo sát để tổng hợp.")]}
        
    compiled_data = "\n\n---\n\n".join(findings)
    
    synthesis_prompt = (
        "Bạn là một Kiến trúc sư Hệ thống chuyên nghiệp chuyên biên soạn tài liệu.\n"
        "Hãy tổng hợp toàn bộ thông tin khảo sát thô được ghi nhận ở các bước trước thành một tài liệu 'THONGTIN.md' duy nhất.\n"
        "Yêu cầu: Viết thật cô đọng, súc tích và có cấu trúc rõ ràng. TUYỆT ĐỐI KHÔNG chèn mã nguồn dài dòng."
    )
    
    try:
        response = fast_model.invoke([
            SystemMessage(content=synthesis_prompt),
            HumanMessage(content=f"Thông tin thu thập:\n\n{compiled_data}")
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
            
            if git_branch != "no_git":
                tools_mgr = WorkspaceTools(ws)
                tools_mgr.write_file("THONGTIN.md", cleaned_md)
                
                git_manager = GitManager(ws)
                git_manager._run_cmd(["git", "add", "THONGTIN.md"], ignore_error=True)
                
                message_content = (
                    "**Tổng hợp tài liệu hoàn tất:** Đã biên dịch tri thức khảo sát, "
                    "lưu vật lý thành tệp `THONGTIN.md` và đưa vào Git staging thành công."
                )
            else:
                message_content = (
                    "**Tổng hợp tài liệu hoàn tất (Chế độ In-Memory):** Tri thức khảo sát "
                    "đã được tổng hợp và nạp trực tiếp vào ngữ cảnh trạng thái đồ thị (`workspace_context`). "
                    "Tệp tin `THONGTIN.md` vật lý **không** được tạo trên ổ đĩa do hệ thống phát hiện không sử dụng Git."
                )
            
            return {
                "workspace_context": cleaned_md,
                "messages": [AIMessage(content=message_content)]
            }
    except Exception as e:
        return {"messages": [AIMessage(content=f"Cảnh báo: Có lỗi xảy ra khi tổng hợp tệp THONGTIN.md: {str(e)}")]}
    return {}


def commit_node(state: AgentState) -> Dict[str, Any]:
    ws = state["workspace_path"]
    task_type = state.get("task_type", "development")
    git_branch = state.get("git_branch", "no_git")
    
    if git_branch == "no_git":
        return {
            "messages": [AIMessage(content="Đã hoàn thành toàn bộ yêu cầu của bạn. Chế độ Không-Git được kích hoạt, bỏ qua commit.")]
        }
        
    git_manager = GitManager(ws)
    status = git_manager._run_cmd(["git", "status", "--porcelain"], ignore_error=True)
    
    plan = state["plan"]
    plan_steps_str = "\n".join([
        f"- [{getattr(t, 'id', None) or t.get('id')}] {getattr(t, 'description', None) or t.get('description')} ({getattr(t, 'status', None) or t.get('status')})"
        for t in plan
    ])
    
    if status and not status.startswith("ERROR"):
        commit_msg = f"feat(ai): automatic execution ({task_type}) \n\nSteps:\n{plan_steps_str}"
        git_manager.commit_changes(commit_msg)
        msg = f"Đã hoàn thành yêu cầu và commit các thay đổi lên nhánh `{git_branch}`."
    else:
        msg = "Đã hoàn thành quy trình công việc. Không có thay đổi tệp tin vật lý nào cần commit lên Git."
        
    return {"messages": [AIMessage(content=msg)]}