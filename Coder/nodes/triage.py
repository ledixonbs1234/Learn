# oder/nodes/triage.py
import os
import uuid
from pathlib import Path
from typing import Dict, Any
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langgraph.types import interrupt

from config import model, find_project_root_heuristic, sanitize_and_resolve_path
from state import AgentState, TaskTriage, Task
from prompts_loader import load_prompt
from skills_engine import AgentSkillsEngine
from .utils import (
    get_text_content_safely, extract_path_from_text, resolve_special_system_paths,
    find_extension_dir_heuristic, verify_workspace_safety, encode_image_to_data_uri
)

def triage_node_stateful(state: AgentState, catalog_summary: str, mcp_summary: str) -> TaskTriage:
    messages = state.get("messages", [])
    current_workspace = state.get("workspace_path", "")
    active_session_context = (
        "=== NGỮ CẢNH PHIÊN HOẠT ĐỘNG HIỆN TẠI (ACTIVE SESSION) ===\n"
        f"- Thư mục làm việc hiện hành (Workspace): `{current_workspace or 'Chưa thiết lập'}`\n"
    )
    context_messages = []
    if len(messages) > 1:
        recent_history = messages[-5:-1]
        context_messages.append(SystemMessage(content="--- LỊCH SỬ HỘI THOẠI GẦN NHẤT ĐỂ THAM CHIẾU NGỮ CẢNH ---"))
        context_messages.extend(recent_history)

    user_msg = messages[-1]
    user_query_text = get_text_content_safely(user_msg.content)

    system_prompt_template = load_prompt("triage_stateful.txt")
    system_prompt = system_prompt_template.replace(
        "{catalog_summary}", catalog_summary
    ).replace(
        "{mcp_summary}", mcp_summary
    )

    structured_llm = model.with_structured_output(TaskTriage, method="function_calling")
    return structured_llm.invoke([
        SystemMessage(content=system_prompt),
        SystemMessage(content=active_session_context),
        *context_messages,
        HumanMessage(content=f"Yêu cầu hiện tại của người dùng: {user_query_text}")
    ])

def detect_and_triage_node(state: AgentState) -> Dict[str, Any]:
    messages = state["messages"]
    user_msg = messages[-1]
    user_query_text = get_text_content_safely(user_msg.content)
    
    random_hex = uuid.uuid4().hex[:8]
    import tempfile
    temp_workspace_path = Path(tempfile.gettempdir()) / f"agent_{os.getpid()}_{random_hex}"
    temp_workspace_path.mkdir(parents=True, exist_ok=True)
    safe_temp_dir = str(temp_workspace_path.resolve())
    
    existing_workspace = state.get("workspace_path", "")
    if not existing_workspace or existing_workspace == ".":
        existing_workspace = safe_temp_dir

    detected_path_str = extract_path_from_text(user_query_text)
    if not detected_path_str:
        path_keywords = ["thư mục", "folder", "dự án", "project", "mở", "quét", "ls", "dir", "làm việc tại", "tại", "cd"]
        if any(kw in user_query_text.lower() for kw in path_keywords) or not existing_workspace:
            detected_path_str = resolve_special_system_paths(user_query_text)

    provisional_workspace = existing_workspace
    pivoted_msg = ""
    if detected_path_str:
        try:
            resolved_path = Path(detected_path_str).expanduser().resolve()
            if resolved_path.exists():
                provisional_workspace = str(find_project_root_heuristic(resolved_path))
                if provisional_workspace != existing_workspace and existing_workspace:
                    pivoted_msg = f"🔄 **[Chuyển đổi Workspace]**: Phát hiện yêu cầu chuyển đổi thư mục làm việc sang: `{provisional_workspace}`\n"
        except Exception:
            pass
    else:
        if existing_workspace:
            pivoted_msg = f"🔄 **[Kế thừa Workspace]**: Sử dụng lại thư mục làm việc hiện hành: `{existing_workspace}`\n"

    image_paths = state.get("image_paths", []) or []
    updated_messages = []
    if image_paths and (user_msg.type == "human" or isinstance(user_msg, HumanMessage)):
        multimodal_content = [{"type": "text", "text": user_query_text}]
        for path_str in image_paths:
            try:
                safe_img_path = sanitize_and_resolve_path(provisional_workspace, path_str)
                if safe_img_path.exists() and safe_img_path.is_file():
                    data_uri = encode_image_to_data_uri(safe_img_path)
                    multimodal_content.append({
                        "type": "image_url",
                        "image_url": {"url": data_uri, "detail": "low"}
                    })
            except Exception as e:
                print(f"[Cảnh báo hệ thống] Lỗi xử lý hình ảnh '{path_str}': {str(e)}")
        updated_user_msg = HumanMessage(content=multimodal_content, id=user_msg.id)
        updated_messages.append(updated_user_msg)

    temp_state = state.copy()
    temp_state["workspace_path"] = provisional_workspace
    skills_engine = AgentSkillsEngine(provisional_workspace)
    catalog = skills_engine.scan_catalog()
    catalog_summary = ""
    if catalog:
        catalog_summary = "\n=== 📚 DANH MỤC KỸ NĂNG HỆ THỐNG HIỆN CÓ ===\n"
        for item in catalog:
            catalog_summary += f"- Kỹ năng: `{item['name']}`\n  Điều kiện kích hoạt: {item['description']}\n"

    mcp_summary = ""
    try:
        from mcp_helper import MCPRegistryManager
        mcp_manager = MCPRegistryManager(provisional_workspace)
        raw_servers = mcp_manager.load_config()
        if raw_servers:
            mcp_summary = "\n=== 🔌 MÁY CHỦ MCP NGOẠI VI KHẢ DỤNG ===\n"
            for name, cfg in raw_servers.items():
                mcp_summary += f"- Máy chủ: `{name}` (Hành vi: {cfg.get('transport', 'stdio')})\n"
    except Exception:
        pass

    if provisional_workspace and provisional_workspace != existing_workspace:
        temp_ext = find_extension_dir_heuristic(Path(provisional_workspace))
        temp_state["extension_path"] = temp_ext or ""

    try:
        triage_output = triage_node_stateful(temp_state, catalog_summary, mcp_summary)
        task_type = triage_output.task_type
        is_simple = triage_output.is_simple
        detailed_analysis = triage_output.detailed_analysis
        recommended_skills = triage_output.recommended_skills or []
        recommended_mcp_servers = triage_output.recommended_mcp_servers or []
    except Exception as e:
        task_type = "clarify" if not provisional_workspace else "analysis"
        is_simple = True
        detailed_analysis = f"Lỗi hệ thống phân loại: {str(e)}"
        recommended_skills = []
        recommended_mcp_servers = []

    final_workspace = provisional_workspace
    if task_type == "clarify" or not final_workspace:
        interrupt_payload = {
            "type": "path_clarification",
            "prompt": "Hệ thống phát hiện bạn muốn làm việc với ứng dụng nhưng chưa cấu hình thư mục làm việc cụ thể.",
            "fields": [
                {
                    "name": "target_workspace_path",
                    "label": "Đường dẫn thư mục dự án của bạn",
                    "placeholder": "Ví dụ: C:/Users/Name/Projects/my-app",
                    "type": "text",
                    "required": True
                }
            ]
        }
        user_response = interrupt(interrupt_payload)
        if isinstance(user_response, dict) and "target_workspace_path" in user_response:
            detected_path_str = str(user_response["target_workspace_path"]).strip()
        elif isinstance(user_response, str):
            detected_path_str = user_response.strip()
        try:
            final_workspace = str(Path(detected_path_str).expanduser().resolve())
        except Exception:
            final_workspace = "."

    is_user_explicit = (detected_path_str is not None)
    if not is_simple and not verify_workspace_safety(final_workspace, allow_explicit=is_user_explicit):
        return {
            "workspace_path": final_workspace,
            "plan": [], "task_type": "analysis", "is_simple": True,
            "detailed_analysis": "Ngắt hoạt động do vi phạm rào chắn bảo mật Workspace dành cho tác vụ phức tạp.",
            "messages": [AIMessage(content="🚨 **[CẢNH BÁO BẢO MẬT]**: Tác vụ phát triển phức tạp yêu cầu một Workspace an toàn bên ngoài thư mục Agent. Vui lòng chỉ định một thư mục làm việc hợp lệ.")]
        }

    active_skills = {}
    analysis_triad = ["grill-with-docs", "write-a-prd", "domain-modeling"]
    requires_triad = any(skill in recommended_skills for skill in analysis_triad)
    plan = []
    if is_simple:
        plan = [Task(id="T1", description=f"Thực hiện trực tiếp tác vụ tại `{final_workspace}`: {user_query_text}", dependencies=[], status="pending")]
    else:
        survey_desc = (
            f"Bắt đầu pha khảo sát. Bạn hãy gọi công cụ `activate_agent_skill` để kích hoạt tuần tự và "
            f"thực thi các kỹ năng `grill-with-docs` (chất vấn làm rõ yêu cầu về: {user_query_text}), "
            f"`domain-modeling` (đồng bộ bảng thuật ngữ vào `CONTEXT.md`), "
            f"và `write-a-prd` (thiết lập tệp đặc tả `PRD.md`)."
        ) if requires_triad else f"Khảo sát cấu trúc file và mã nguồn tại `{final_workspace}` liên quan đến yêu cầu: {user_query_text} và thu thập dữ liệu để lập kế hoạch chi tiết."
        plan = [Task(id="T_SURVEY", description=survey_desc, dependencies=[], status="pending")]
        task_type = "analysis"

    skills_log_msg = f"- 📋 **Kỹ năng đề xuất (Đã nạp sẵn cho pha khảo sát):** {', '.join([f'`{s}`' for s in recommended_skills])}\n" if recommended_skills else ""
    mcp_log_msg = f"- 🔌 **Máy chủ MCP được AI kích hoạt:** {', '.join([f'`{s}`' for s in recommended_mcp_servers])}\n" if recommended_mcp_servers else ""
    triage_info_msg = (
        f"📊 **[Hệ thống Phân phối thông minh]**:\n{pivoted_msg}"
        f"- **Workspace hoạt động:** `{final_workspace}`\n"
        f"- **Chế độ kiểm soát:** {'Đơn giản (Fast-Track)' if is_simple else 'Phức tạp (Multi-Step Discovery)'}\n"
        f"- **Pha hoạt động khởi động:** `{task_type.upper()}`\n{skills_log_msg}{mcp_log_msg}\n"
        f"🎯 **[Phân tích mục tiêu kỹ thuật]**:\n{detailed_analysis}"
    )

    return {
        "workspace_path": final_workspace, "plan": plan, "task_type": task_type,
        "is_simple": is_simple, "detailed_analysis": detailed_analysis,
        "messages": updated_messages + [AIMessage(content=triage_info_msg)],
        "error_logs": "", "attempts": 0, "modified_files": [], "last_executed_task_ids": [],
        "replanning_count": 0, "step_findings": ["__RESET__"], "active_skills": active_skills,
        "recommended_skills": recommended_skills, "active_mcp_servers": recommended_mcp_servers
    }

def triage_node(state: AgentState) -> Dict[str, Any]:
    messages = state["messages"]
    user_msg = next((msg for msg in reversed(messages) if isinstance(msg, HumanMessage) or getattr(msg, "type", None) == "human"), messages[0])
    user_query_text = get_text_content_safely(user_msg.content)
    
    structured_llm = model.with_structured_output(TaskTriage, method="function_calling")
    system_prompt = load_prompt("triage_simple.txt")
    
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

    plan = [Task(id="T1", description=f"Thực hiện trực tiếp yêu cầu: {user_query_text}", dependencies=[], status="pending")] if is_simple else [
        Task(id="T_SURVEY", description="Khảo sát cấu trúc thư mục, tệp cấu hình manifest.json và mã nguồn chính của dự án bằng các công cụ khảo sát để nắm vững kiến trúc trước khi lập kế hoạch.", dependencies=[], status="pending")
    ]
    if not is_simple:
        task_type = "analysis"

    return {
        "plan": plan, "task_type": task_type, "is_simple": is_simple, "detailed_analysis": detailed_analysis,
        "messages": [AIMessage(content=f"📊 **[Phân loại tác vụ]**: Hệ thống xác định yêu cầu thuộc diện {'ĐƠN GIẢN (Fast-Track)' if is_simple else 'PHỨC TẠP (Multi-Step Discovery)'} | Pha hoạt động khởi động: `{task_type.upper()}`.\n\n🎯 **[Phân tích mục tiêu]**:\n{detailed_analysis}")]
    }