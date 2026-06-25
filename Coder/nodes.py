# nodes.py
import subprocess
from pathlib import Path
from typing import Dict, Any, Union, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from config import model, sanitize_and_resolve_path
from state import AgentState, WorkspaceDetection, TaskPlan
from tools import (
    GitManager, WorkspaceTools, 
    ReadFilesTool, WriteFileTool, ListDirectoryTool, RunTerminalTool
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
    
    if current_idx >= len(steps):
         return {"messages": [AIMessage(content="Đã hoàn thành khảo sát toàn bộ các bước.")]}
         
    current_step = steps[current_idx]
    
    # Khởi tạo công cụ tĩnh một cách tối ưu
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
        "Hãy sử dụng các công cụ `read_files` và `list_directory` để đọc hiểu cấu trúc hệ thống.\n"
        "YÊU CẦU QUAN TRỌNG:\n"
        "- Bạn chỉ có quyền ĐỌC dữ liệu, tuyệt đối không chỉnh sửa mã nguồn hoặc tự ý tạo tệp tin trong bước này.\n"
        "- Trình bày chi tiết, chuyên nghiệp về kết quả phát hiện được của bạn ở tin nhắn phản hồi cuối cùng.\n"
        "- Dựa trên ngữ cảnh khảo sát từ các bước trước (nếu có) để không thực hiện lại các thao tác tìm kiếm thừa."
    )
    if previous_findings_str:
        system_prompt += previous_findings_str
    
    react_messages = [SystemMessage(content=system_prompt), HumanMessage(content=f"Yêu cầu: Hãy khảo sát bước này: '{current_step}'")]
    
    for _ in range(10):
        try:
            response = model_with_tools.invoke(react_messages)
        except Exception as e:
            react_messages.append(AIMessage(content=f"Lỗi gọi mô hình: {str(e)}"))
            break
            
        react_messages.append(response)
        if not response.tool_calls:
            break
            
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"] or {}
            
            try:
                if tool_name == "read_files":
                    result = read_files.invoke(tool_args)
                elif tool_name == "list_directory":
                    result = list_directory.invoke(tool_args)
                else:
                    result = f"Lỗi: Không tìm thấy công cụ '{tool_name}'."
            except Exception as tool_err:
                result = f"Lỗi thực thi công cụ '{tool_name}': {str(tool_err)}"
                
            react_messages.append(ToolMessage(content=str(result), name=tool_name, tool_call_id=tool_call["id"]))
            
    final_output = ""
    for msg in reversed(react_messages):
        if isinstance(msg, AIMessage) and msg.content and len(msg.content.strip()) > 15:
            final_output = msg.content.strip()
            break
            
    if not final_output:
        try:
            summary_prompt = "Hãy viết một báo cáo tóm tắt chi tiết các kết quả khảo sát bạn đã thu được ở bước này."
            react_messages.append(HumanMessage(content=summary_prompt))
            forced_response = model.invoke(react_messages)
            final_output = forced_response.content
        except Exception as e:
            final_output = f"Lỗi xảy ra trong quá trình tổng kết: {str(e)}"
        
    findings = [f"### Khảo sát bước '{current_step}':\n{final_output}"]
    
    return {
        "messages": [AIMessage(content=f"**[Phân tích] Thực thi bước '{current_step}':**\n\n{final_output}")],
        "current_step_idx": current_idx + 1,
        "step_findings": findings
    }


def development_executor_node(state: AgentState) -> Dict[str, Any]:
    ws = state["workspace_path"]
    steps = state["plan"]
    current_idx = state["current_step_idx"]
    error_logs = state.get("error_logs", "")
    
    if current_idx >= len(steps):
         return {"messages": [AIMessage(content="Đã hoàn thành toàn bộ các bước phát triển.")]}
         
    current_step = steps[current_idx]
    
    # Khởi tạo công cụ tĩnh
    read_files = ReadFilesTool(workspace_path=ws)
    write_file = WriteFileTool(workspace_path=ws)
    list_directory = ListDirectoryTool(workspace_path=ws)
    run_terminal_command = RunTerminalTool(workspace_path=ws)
    
    tools = [read_files, write_file, list_directory, run_terminal_command]
    model_with_tools = model.bind_tools(tools)
    
    system_prompt = (
        "Bạn là một kỹ sư phần mềm thực thi chuyên nghiệp chuyên sửa lỗi và viết mới mã nguồn (Write-Access Mode).\n"
        f"Nhiệm vụ: Bạn đang thực hiện bước phát triển {current_idx + 1}/{len(steps)}: '{current_step}'.\n"
        f"Thư mục làm việc: {ws}\n"
        "Hãy sử dụng các công cụ `read_files`, `write_file`, `list_directory` và `run_terminal_command` để giải quyết nhiệm vụ.\n"
        "YÊU CẦU QUAN TRỌNG:\n"
        "- Khi chỉnh sửa tệp đã có, luôn luôn dùng `read_files` đọc nội dung trước để hiểu ngữ cảnh và tránh làm mất mã nguồn cũ.\n"
        "- Bạn có thể cài đặt thư viện, build dự án, hoặc chạy kiểm thử bằng công cụ `run_terminal_command`.\n"
        "- Mô tả thật chi tiết các hành động, giải pháp và vị trí các dòng code bạn đã cập nhật ở tin nhắn phản hồi cuối cùng."
    )
    
    react_messages = [SystemMessage(content=system_prompt)]
    if error_logs:
        react_messages.append(HumanMessage(content=f"LƯU Ý SỬA LỖI TỪ LƯỢT CHẠY TRƯỚC (HỆ THỐNG KIỂM TRA BÁO LỖI):\n{error_logs}\nHãy tập trung khắc phục lỗi này triệt để."))
    react_messages.append(HumanMessage(content=f"Yêu cầu: Hãy thực hiện phát triển bước này: '{current_step}'"))
    
    # Tạo bản sao cục bộ để tránh mutate trực tiếp State List
    modified_files = list(state.get("modified_files", []))
    
    for _ in range(5):
        try:
            response = model_with_tools.invoke(react_messages)
        except Exception as e:
            react_messages.append(AIMessage(content=f"Lỗi gọi mô hình: {str(e)}"))
            break
            
        react_messages.append(response)
        
        if not response.tool_calls:
            break
            
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"] or {}
            
            try:
                if tool_name == "read_files":
                    result = read_files.invoke(tool_args)
                elif tool_name == "write_file":
                    file_path = tool_args.get("file_path")
                    if not file_path:
                        raise ValueError("Thiếu tham số 'file_path' bắt buộc trong lệnh gọi write_file.")
                    result = write_file.invoke(tool_args)
                    try:
                        safe_path = sanitize_and_resolve_path(ws, file_path, create_parent=True)
                        if str(safe_path) not in modified_files:
                            modified_files.append(str(safe_path))
                    except Exception:
                        pass
                elif tool_name == "list_directory":
                    result = list_directory.invoke(tool_args)
                elif tool_name == "run_terminal_command":
                    result = run_terminal_command.invoke(tool_args)
                else:
                    result = f"Lỗi: Không tìm thấy công cụ '{tool_name}'."
            except Exception as tool_err:
                result = f"Lỗi thực thi công cụ '{tool_name}': {str(tool_err)}"
                
            react_messages.append(ToolMessage(content=str(result), name=tool_name, tool_call_id=tool_call["id"]))
            
    final_output = ""
    for msg in reversed(react_messages):
        if isinstance(msg, AIMessage) and msg.content and len(msg.content.strip()) > 15:
            final_output = msg.content.strip()
            break
            
    if not final_output:
        try:
            summary_prompt = "Hãy viết một tóm tắt chi tiết về mã nguồn bạn đã chỉnh sửa hoặc tạo mới trong bước này."
            react_messages.append(HumanMessage(content=summary_prompt))
            forced_response = model.invoke(react_messages)
            final_output = forced_response.content
        except Exception as e:
            final_output = f"Lỗi xảy ra khi cố gắng tóm tắt mã nguồn: {str(e)}"
        
    # LƯU Ý: Không tự ý cộng `current_step_idx` tại đây nữa. 
    # Việc quản lý chuyển tiếp bước tiếp theo sẽ do `tester_node` xử lý sau khi kiểm tra xong.
    return {
        "messages": [AIMessage(content=f"**[Phát triển] Thực thi bước '{current_step}':**\n\n{final_output}")],
        "modified_files": modified_files
    }


def tester_node(state: AgentState) -> Dict[str, Any]:
    modified_files = state.get("modified_files", [])
    attempts = state.get("attempts", 0)
    current_idx = state["current_step_idx"]
    errors = []
    
    # 1. Kiểm tra biên dịch mã nguồn Python
    for file_path in modified_files:
        if file_path.endswith(".py"):
            res = subprocess.run(["python", "-m", "py_compile", file_path], capture_output=True, text=True)
            if res.returncode != 0:
                errors.append(f"Lỗi cú pháp tại {Path(file_path).name}:\n{res.stderr.strip()}")
                
    if errors:
        combined_error = "\n".join(errors)
        # Nếu gặp lỗi và chưa vượt quá 3 lần sửa lỗi liên tục cho bước này:
        if attempts < 2:  # Lần 0, 1 -> Tổng cộng tối đa 3 lần thực thi (1 lần gốc + 2 lần sửa lỗi)
            return {
                "error_logs": combined_error,
                "attempts": attempts + 1,
                "messages": [AIMessage(content=f"⚠️ Phát hiện lỗi kiểm tra tại bước {current_idx + 1}:\n{combined_error}\nHệ thống chuẩn bị quay lại sửa lỗi.")]
            }
        else:
            # Quá giới hạn sửa lỗi cho bước này. Bắt buộc bỏ qua để chuyển sang bước tiếp theo
            return {
                "error_logs": "",
                "attempts": 0,
                "current_step_idx": current_idx + 1,
                "messages": [AIMessage(content=f"❌ Đã vượt quá 3 lần tự động sửa lỗi tại bước {current_idx + 1}. Bỏ qua bước này để tiếp tục thực hiện các bước khác.")]
            }
            
    # 2. Vượt qua kiểm thử thành công: Reset lỗi, tăng chỉ mục bước, chuẩn bị cho bước tiếp theo
    return {
        "error_logs": "",
        "attempts": 0,
        "current_step_idx": current_idx + 1,
        "messages": [AIMessage(content=f"✅ Bước {current_idx + 1} đã vượt qua vòng kiểm tra thành công. Tiến hành lưu vết.")]
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