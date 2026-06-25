# nodes.py
import subprocess
from pathlib import Path
from typing import Dict, Any, Union, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from config import model, sanitize_and_resolve_path
from state import AgentState, WorkspaceDetection, TaskPlan, TaskTriage
from tools import (
    GitManager, WorkspaceTools, 
    ReadFilesTool, WriteFileTool, ApplyPatchTool, 
    ListDirectoryTool, RunTerminalTool
)

def detect_workspace_node(state: AgentState) -> Dict[str, Any]:
    if state.get("workspace_path"):
        resolved_ws = str(Path(state["workspace_path"]).resolve())
        return {
            "workspace_path": resolved_ws,
            "messages": [AIMessage(content=f"Sử dụng workspace sẵn có: `{resolved_ws}`")]
        }

    messages = state["messages"]
    user_msg = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) or getattr(msg, "type", None) == "human":
            user_msg = msg
            break
            
    if not user_msg:
        user_msg = messages[-1]

    last_message = user_msg.content
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


def context_loader_node(state: AgentState) -> Dict[str, Any]:
    ws = state["workspace_path"]
    
    messages = state["messages"]
    user_msg = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) or getattr(msg, "type", None) == "human":
            user_msg = msg
            break
            
    if not user_msg:
        user_msg = messages[0]
        
    user_query = user_msg.content
    
    workspace_context = ""
    thongtin_path = Path(ws) / "THONGTIN.md"
    if thongtin_path.exists():
        try:
            workspace_context = thongtin_path.read_text(encoding="utf-8")
        except Exception as e:
            workspace_context = f"Lỗi khi đọc file THONGTIN.md: {str(e)}"
            
    structured_llm = model.with_structured_output(TaskTriage, method="function_calling")
    
    system_prompt = (
        "Bạn là một điều phối viên Agent thông minh. Hãy phân tích yêu cầu của người dùng "
        "để xác định xem đây là một yêu cầu đơn giản (chỉ cần chỉnh sửa trực tiếp 1-2 file, không cần lập kế hoạch chi tiết) "
        "hay một yêu cầu phức tạp (cần lên kế hoạch khảo sát, phát triển nhiều bước).\n"
        "Ví dụ về tác vụ ĐƠN GIẢN: Thêm một trường/thuộc tính mới vào model, sửa lỗi giao diện nhỏ, "
        "ẩn/hiện một phần tử dựa trên điều kiện đầu vào của người dùng."
    )
    
    try:
        triage_output = structured_llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ])
        is_simple = getattr(triage_output, "is_simple", False)
        task_type = getattr(triage_output, "task_type", "development")
    except Exception:
        is_simple = False
        task_type = "development"
        
    plan_steps = []
    messages_to_append = []
    
    if is_simple:
        plan_steps = [f"Xử lý trực tiếp yêu cầu của người dùng: {user_query}"]
        messages_to_append.append(AIMessage(content=f"📋 Đã tải ngữ cảnh `THONGTIN.md`. Kích hoạt chế độ **Fast-Track (Nhiệm vụ đơn giản)**. Bỏ qua bước lập kế hoạch chi tiết."))
    else:
        messages_to_append.append(AIMessage(content="📋 Đã tải ngữ cảnh `THONGTIN.md`. Nhận diện tác vụ phức tạp, chuyển tiếp tới Planner để thiết kế kế hoạch."))
        
    return {
        "workspace_context": workspace_context,
        "plan": plan_steps,
        "task_type": task_type,
        "messages": messages_to_append
    }


def planner_node(state: AgentState) -> Dict[str, Any]:
    ws = state["workspace_path"]
    conversation_history = state["messages"]
    existing_plan = state.get("plan", [])
    workspace_context = state.get("workspace_context", "")
    
    if existing_plan:
        return {
            "current_step_idx": 0,
            "modified_files": [],
            "attempts": 0,
            "step_findings": ["__RESET__"],
        }
    
    structured_llm = model.with_structured_output(TaskPlan, method="function_calling")
    
    system_prompt = (
        "Bạn là một Kiến trúc sư phần mềm cấp cao. Hãy lập hoặc cập nhật kế hoạch thực hiện "
        f"cho dự án nằm trong thư mục '{ws}'.\n"
    )
    if workspace_context:
        system_prompt += f"\n--- NGỮ CẢNH HỆ THỐNG HIỆN TẠI (THONGTIN.md) ---\n{workspace_context}\n"
        
    system_prompt += (
        "Hãy phân tích kỹ lưỡng toàn bộ lịch sử trò chuyện của người dùng để sinh ra kế hoạch hành động tiếp theo thích hợp nhất.\n"
        "Đặc biệt lưu ý phân loại trường `task_type` chính xác:\n"
        "- Chọn 'analysis' nếu yêu cầu chỉ là đọc, khảo sát cấu trúc dự án hoặc viết báo cáo.\n"
        "- Chọn 'development' nếu yêu cầu có can thiệp sửa đổi, cập nhật hoặc viết mới mã nguồn."
    )
    
    plan_output = structured_llm.invoke([
        {"role": "system", "content": system_prompt},
        *conversation_history
    ])
    
    plan_steps = getattr(plan_output, "steps", [])
    task_type = getattr(plan_output, "task_type", "development")
    
    return {
        "plan": plan_steps,
        "task_type": task_type,
        "current_step_idx": 0,
        "modified_files": [],
        "attempts": 0,
        "step_findings": ["__RESET__"],
        "messages": [AIMessage(content=f"Đã lập kế hoạch hành động (Loại: {task_type.upper()}):\n" + "\n".join([f"- {s}" for s in plan_steps]))]
    }


def analysis_executor_node(state: AgentState) -> Dict[str, Any]:
    ws = state["workspace_path"]
    steps = state["plan"]
    current_idx = state["current_step_idx"]
    workspace_context = state.get("workspace_context", "")
    messages = list(state["messages"])
    
    if current_idx >= len(steps):
         return {
             "messages": [AIMessage(content="Đã hoàn thành khảo sát toàn bộ các bước.")]
         }
         
    current_step = steps[current_idx]
    
    read_files = ReadFilesTool(workspace_path=ws)
    list_directory = ListDirectoryTool(workspace_path=ws)
    tools = [read_files, list_directory]
    model_with_tools = model.bind_tools(tools)
    
    previous_findings_str = ""
    existing_findings = state.get("step_findings", [])
    if existing_findings:
        previous_findings_str = "\n\n--- CÁC KẾT QUẢ KHẢO SÁT BẠN ĐÃ THU THẬP ĐƯỢC Ở CÁC BƯỚC TRƯỚC ---\n" + "\n\n".join(existing_findings)
    
    system_prompt = (
        "Bạn là một kiến trúc sư chuyên khảo sát, đọc hiểu và phân tích cấu trúc mã nguồn (Read-Only Mode).\n"
        f"Nhiệm vụ: Bạn đang thực hiện bước khảo sát {current_idx + 1}/{len(steps)}: '{current_step}'.\n"
        f"Thư mục làm việc: {ws}\n"
    )
    if workspace_context:
        system_prompt += f"\n--- TỔNG QUAN VỀ DỰ ÁN (THONGTIN.md) ---\n{workspace_context}\n"
        
    system_prompt += (
        "\nHãy sử dụng các công cụ `read_files` và `list_directory` để đọc hiểu cấu trúc hệ thống.\n"
        "YÊU CẦU QUAN TRỌNG:\n"
        "- Bạn chỉ có quyền ĐỌC dữ liệu, tuyệt đối không chỉnh sửa mã nguồn hoặc tự ý tạo tệp tin trong bước này.\n"
        "- Trình bày chi tiết, chuyên nghiệp về kết quả phát hiện được của bạn ở tin nhắn phản hồi cuối cùng.\n"
        "- Dựa trên ngữ cảnh khảo sát từ các bước trước (nếu có) để không thực hiện lại các thao tác tìm kiếm thừa."
    )
    if previous_findings_str:
        system_prompt += previous_findings_str
    
    response = model_with_tools.invoke([SystemMessage(content=system_prompt)] + messages)
    
    # CHUYỂN DỊCH THỜI ĐIỂM TĂNG CHỈ MỤC: Tăng index ngay khi không còn yêu cầu gọi công cụ
    if not response.tool_calls:
        findings = []
        if response.content:
            findings = [f"### Khảo sát bước '{current_step}':\n{response.content}"]
        return {
            "messages": [response],
            "current_step_idx": current_idx + 1,  # <--- Tăng trực tiếp tại đây
            "step_findings": findings
        }
    else:
        return {
            "messages": [response]
        }


def development_executor_node(state: AgentState) -> Dict[str, Any]:
    ws = state["workspace_path"]
    steps = state["plan"]
    current_idx = state["current_step_idx"]
    error_logs = state.get("error_logs", "")
    workspace_context = state.get("workspace_context", "")
    messages = list(state["messages"])
        
    if current_idx >= len(steps):
         return {
             "messages": [AIMessage(content="Đã hoàn thành toàn bộ các bước phát triển.")]
         }
         
    current_step = steps[current_idx]
    
    read_files = ReadFilesTool(workspace_path=ws)
    write_file = WriteFileTool(workspace_path=ws)
    apply_patch = ApplyPatchTool(workspace_path=ws)
    list_directory = ListDirectoryTool(workspace_path=ws)
    run_terminal_command = RunTerminalTool(workspace_path=ws)
    
    tools = [read_files, write_file, apply_patch, list_directory, run_terminal_command]
    model_with_tools = model.bind_tools(tools)
    
    system_prompt = (
        "Bạn là một kỹ sư phần mềm thực thi chuyên nghiệp chuyên sửa lỗi và viết mới mã nguồn (Write-Access Mode).\n"
        f"Nhiệm vụ: Bạn đang thực hiện bước phát triển {current_idx + 1}/{len(steps)}: '{current_step}'.\n"
        f"Thư mục làm việc: {ws}\n"
    )
    if workspace_context:
        system_prompt += f"\n--- NGỮ CẢNH HỆ THỐNG HIỆN TẠI (THONGTIN.md) ---\n{workspace_context}\n"
        
    system_prompt += (
        "\nHãy sử dụng các công cụ `read_files`, `write_file`, `apply_search_replace_patch`, `list_directory` và `run_terminal_command` để giải quyết nhiệm vụ.\n"
        "\n⚠️ QUY TẮC QUAN TRỌNG VỀ SỬA ĐỔI FILE (BẮT BUỘC TUÂN THỦ):\n"
        "1. Đối với các file có sẵn trong hệ thống có độ dài TRÊN 300 DÒNG (hoặc tệp tin có kích thước lớn):\n"
        "   - Bạn TUYỆT ĐỐI KHÔNG ĐƯỢC sử dụng công cụ `write_file` để ghi đè lại toàn bộ tệp.\n"
        "   - Thay vào đó, bạn BẮT BUỘC PHẢI sử dụng công cụ `apply_search_replace_patch` để áp dụng bản vá cục bộ (Search-and-Replace).\n"
        "   - Điều này giúp tránh lỗi ngắt quãng đầu ra (output truncation) và giữ tính toàn vẹn của mã nguồn hiện hành.\n"
        "2. Công cụ `write_file` CHỈ được dùng để:\n"
        "   - Tạo mới các tệp tin chưa từng tồn tại trong workspace.\n"
        "   - Sửa đổi các tệp cực kỳ ngắn (dưới 300 dòng).\n"
        "3. Khi sử dụng `apply_search_replace_patch`:\n"
        "   - Hãy chắc chắn rằng khối văn bản SEARCH trùng khớp 100% từng ký tự, dấu cách và thụt lề dòng (indentation) với tệp gốc.\n"
        "\nYÊU CẦU PHÁT TRIỂN KHÁC:\n"
        "- Khi chỉnh sửa tệp đã có, luôn luôn dùng `read_files` đọc nội dung trước để hiểu ngữ cảnh và tránh làm mất mã nguồn cũ.\n"
        "- Bạn có thể cài đặt thư viện, build dự án, hoặc chạy kiểm thử bằng công cụ `run_terminal_command`.\n"
        "- Mô tả thật chi tiết các hành động, giải pháp và vị trí các dòng code bạn đã cập nhật ở tin nhắn phản hồi cuối cùng."
    )
    
    input_messages = [SystemMessage(content=system_prompt)]
    if error_logs:
        input_messages.append(HumanMessage(content=f"LƯU Ý SỬA LỖI TỪ LƯỢT CHẠY TRƯỚC (HỆ THỐNG KIỂM TRA BÁO LỖI):\n{error_logs}\nHãy tập trung khắc phục lỗi này triệt để."))
        
    response = model_with_tools.invoke(input_messages + messages)
    
    # CHUYỂN DỊCH THỜI ĐIỂM TĂNG CHỈ MỤC: Tăng index ngay khi không còn yêu cầu gọi công cụ
    if not response.tool_calls:
        return {
            "messages": [response],
            "current_step_idx": current_idx + 1,  # <--- Tăng trực tiếp tại đây
            "attempts": 0,                        # Reset số lần thử sửa lỗi cho bước cũ
            "error_logs": ""                      # Xóa vết log lỗi của bước cũ
        }
    else:
        return {
            "messages": [response]
        }


def tool_node(state: AgentState) -> Dict[str, Any]:
    ws = state["workspace_path"]
    read_files = ReadFilesTool(workspace_path=ws)
    write_file = WriteFileTool(workspace_path=ws)
    apply_patch = ApplyPatchTool(workspace_path=ws)
    list_directory = ListDirectoryTool(workspace_path=ws)
    run_terminal_command = RunTerminalTool(workspace_path=ws)
    
    tools_map = {
        "read_files": read_files,
        "write_file": write_file,
        "apply_search_replace_patch": apply_patch,
        "list_directory": list_directory,
        "run_terminal_command": run_terminal_command
    }
    
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {}
        
    tool_messages = []
    modified_files = list(state.get("modified_files", []))
    
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"] or {}
        tool_id = tool_call["id"]
        
        tool_instance = tools_map.get(tool_name)
        if not tool_instance:
            result = f"Lỗi: Không tìm thấy công cụ '{tool_name}'."
        else:
            try:
                result = tool_instance.invoke(tool_args)
                if tool_name in ["write_file", "apply_search_replace_patch"]:
                    file_path = tool_args.get("file_path")
                    if file_path:
                        try:
                            safe_path = sanitize_and_resolve_path(ws, file_path, create_parent=True)
                            if str(safe_path) not in modified_files:
                                modified_files.append(str(safe_path))
                        except Exception:
                            pass
            except Exception as e:
                result = f"Lỗi thực thi công cụ '{tool_name}': {str(e)}"
                
        tool_messages.append(ToolMessage(content=str(result), name=tool_name, tool_call_id=tool_id))
        
    return {
        "messages": tool_messages,
        "modified_files": modified_files
    }


def tester_node(state: AgentState) -> Dict[str, Any]:
    modified_files = state.get("modified_files", [])
    attempts = state.get("attempts", 0)
    current_idx = state["current_step_idx"]  # Đã là (idx + 1) do executor tự động tăng khi hoàn thành
    errors = []
    
    for file_path in modified_files:
        if file_path.endswith(".py"):
            res = subprocess.run(["python", "-m", "py_compile", file_path], capture_output=True, text=True)
            if res.returncode != 0:
                errors.append(f"Lỗi cú pháp tại {Path(file_path).name}:\n{res.stderr.strip()}")
                
        if errors:
            combined_error = "\n".join(errors)
            if attempts < 2:
                # Do lỗi kiểm thử thất bại, ta kéo lùi current_step_idx về bước đang sửa lỗi (current_idx - 1)
                # để chuẩn bị cho lượt sửa đổi tái hiện tại executor node
                return {
                    "error_logs": combined_error,
                    "attempts": attempts + 1,
                    "current_step_idx": current_idx - 1,  # <--- Đưa lùi chỉ mục về bước đang sửa lỗi
                    "messages": [AIMessage(content=f"⚠️ Phát hiện lỗi kiểm tra tại bước {current_idx}:\n{combined_error}\nHệ thống chuẩn bị quay lại sửa lỗi.")]
                }
            else:
                return {
                    "error_logs": "",
                    "attempts": 0,
                    # Giữ nguyên current_idx (đã tăng sẵn) vì chúng ta chấp nhận bỏ qua lỗi và đi tiếp sang bước kế tiếp/commit
                    "messages": [AIMessage(content=f"❌ Đã vượt quá 3 lần tự động sửa lỗi tại bước {current_idx}. Bỏ qua lỗi này để tiếp tục.")]
                }
                
        return {
            "error_logs": "",
            "attempts": 0,
            # Giữ nguyên current_idx (đã tăng sẵn) vì kiểm thử thành công
            "messages": [AIMessage(content=f"✅ Bước {current_idx} đã vượt qua vòng kiểm tra thành công. Tiến hành lưu vết.")]
        }


def synthesis_node(state: AgentState) -> Dict[str, Any]:
    ws = state["workspace_path"]
    findings = state.get("step_findings", [])
    
    if not findings:
        return {"messages": [AIMessage(content="Không thu thập được thông tin khảo sát để tổng hợp.")]}
        
    compiled_data = "\n\n---\n\n".join(findings)
    
    synthesis_prompt = (
        "Bạn là một Kiến trúc sư Hệ thống cao cấp chuyên biên soạn tài liệu kỹ thuật.\n"
        "Nhiệm vụ của bạn là đọc toàn bộ lịch sử khảo sát TUẦN TỰ bên dưới và tổng hợp thành một tài liệu 'THONGTIN.md' duy nhất.\n\n"
        "LƯU Ý QUAN TRỌNG: Do quá trình khảo sát diễn ra tuần tự, các bước phân tích sau có khả năng mở rộng, sửa chữa "
        "hoặc làm rõ thêm các nhận định của các bước phân tích trước đó. Hãy liên kết, đối chiếu logic một cách thống nhất, "
        "tránh lặp thông tin hoặc tạo ra mâu thuẫn trong báo cáo cuối cùng.\n\n"
        "⚠️ YÊU CẦU CỰC KỲ QUAN TRỌNG ĐỂ TRÁNH LỖI TRÀN TOKEN ĐẦU RA (OUTPUT TRUNCATION):\n"
        "- Hãy viết tài liệu thật SÚC TÍCH, CÔ ĐỌNG, tập trung vào cấu trúc, kiến trúc hệ thống và dependencies.\n"
        "- TUYỆT ĐỐI KHÔNG sao chép hay chèn các đoạn mã nguồn dài dòng.\n"
        "- KHÔNG chia nhỏ tài liệu thành các phần rác như 'Phần 1/3'. Tài liệu phải hoàn chỉnh trong lượt sinh này.\n"
        "- Sử dụng bảng biểu súc tích thay vì viết các đoạn mô tả dài dòng.\n\n"
        "Tài liệu phải bao gồm các mục tiêu chuẩn:\n"
        "1. Tổng quan & Môi trường phát triển.\n"
        "2. Cấu trúc thư mục chính.\n"
        "3. Danh sách Dependencies chính và vai trò.\n"
        "4. Mô tả các Module chính, Scrapers & Services.\n"
        "5. Đánh giá nhanh (Điểm mạnh, điểm yếu, đề xuất nâng cấp)."
    )
    
    try:
        response = model.invoke([
            SystemMessage(content=synthesis_prompt),
            HumanMessage(content=f"Dưới đây là toàn bộ thông tin thu thập tuần tự từ hệ thống:\n\n{compiled_data}\n\nHãy tạo tệp 'THONGTIN.md' hoàn chỉnh.")
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
            git_manager._run_cmd(["git", "add", "THONGTIN.md"], ignore_error=True)
            
            return {
                "workspace_context": cleaned_md,
                "messages": [AIMessage(content="**Tổng hợp tài liệu hoàn tất:** Hệ thống đã thu thập tri thức của toàn bộ các bước khảo sát tuần tự và biên dịch thành tệp `THONGTIN.md` thành công tại thư mục gốc của dự án.")]
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
    
    status = git_manager._run_cmd(["git", "status", "--porcelain"], ignore_error=True)
    
    if status and not status.startswith("ERROR"):
        commit_msg = f"feat(ai): automatic execution ({task_type}) \n\nSteps:\n" + "\n".join(state["plan"])
        git_manager.commit_changes(commit_msg)
        msg = f"Đã hoàn thành yêu cầu và commit các thay đổi lên nhánh `{state['git_branch']}`."
    else:
        msg = "Đã hoàn thành quy trình công việc. Không có thay đổi tệp tin vật lý nào cần commit lên Git."
        
    return {
        "messages": [AIMessage(content=msg)]
    }