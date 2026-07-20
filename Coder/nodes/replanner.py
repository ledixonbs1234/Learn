# oder/nodes/replanner.py
import json
from typing import Dict, Any, List
from venv import logger
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import interrupt

from config import model, fast_model
from state import AgentState, Task, PlanUpdate
from prompts_loader import load_prompt
from .utils import sanitize_llm_response_content

def replanner_node(state: AgentState) -> Dict[str, Any]:
    replanning_count = state.get("replanning_count", 0)
    ws = state["workspace_path"]
    plan = state["plan"]
    messages = state["messages"]
    task_type = state.get("task_type", "development")
    workspace_context = state.get("workspace_context", "")
    error_logs = state.get("error_logs", "")
    active_skills = dict(state.get("active_skills", {}))
    
    active_skills_prompt = ""
    if active_skills:
        active_skills_prompt = "\n=== ⚡ CÁC KỸ NĂNG ĐANG HOẠT ĐỘNG (INSTRUCTIONS) ===\n"
        for s_name, s_instructions in active_skills.items():
            active_skills_prompt += f"\n--- CHỒNG CHỈ DẪN KỸ NĂNG `{s_name}` ---\n{s_instructions}\n"

    is_survey_transition = (
        len(plan) == 1 and 
        (plan[0].id if isinstance(plan[0], Task) else plan[0].get("id")) == "T_SURVEY" and
        (plan[0].status if isinstance(plan[0], Task) else plan[0].get("status")) == "completed"
    )
    if replanning_count >= 5 or (not error_logs and not is_survey_transition):
        action_msg = "bypass_limit" if replanning_count >= 5 else "bypass_no_error"
        proposal_message = AIMessage(content=json.dumps({"action": action_msg}, ensure_ascii=False), name="replanner_proposal")
        return {"messages": [proposal_message], "active_skills": active_skills}
    
    plan_str = "\n".join([
        f"- [{t.id if isinstance(t, Task) else t.get('id')}] {t.description if isinstance(t, Task) else t.get('description')} "
        f"(Trạng thái: {t.status if isinstance(t, Task) else t.get('status')}, "
        f"Phụ thuộc: {t.dependencies if isinstance(t, Task) else t.get('dependencies')})"
        for t in plan
    ])
    if is_survey_transition:
        system_prompt_template = load_prompt("replanner_survey.txt")
    else:
        system_prompt_template = load_prompt("replanner_bug.txt")
    
    system_prompt = system_prompt_template.replace("{ws}", ws) + active_skills_prompt
    if workspace_context:
        system_prompt += f"\n\n--- NGỮ CẢNH HỆ THỐNG (CONTEXT.md) ---\n{workspace_context}"
        
    user_prompt = f"Kế hoạch hiện tại:\n{plan_str}\n\n"
    if error_logs:
        user_prompt += f"🚨 LỖI BIÊN DỊCH CẦN SỬA ĐỔI KẾ HOẠCH:\n{error_logs}\n\n"
    user_prompt += "Hãy đưa ra phân tích và đề xuất cập nhật kế hoạch phù hợp thông qua cuộc gọi hàm."
    
    structured_llm = model.with_structured_output(PlanUpdate, method="function_calling")
    try:
        decision = structured_llm.invoke([{"role": "system", "content": system_prompt}] + messages + [{"role": "user", "content": user_prompt}])
        should_modify = getattr(decision, "should_modify_plan", False)
        explanation = getattr(decision, "explanation", "")
        updated_tasks = getattr(decision, "updated_tasks", [])
        updated_task_type = getattr(decision, "task_type", task_type)
        if not updated_tasks or not should_modify:
            proposal_message = AIMessage(content="📋 [Hệ thống tự động duyệt qua: Kế hoạch hiện tại đã tối ưu, không cần cập nhật thêm]", name="replanner_proposal", additional_kwargs={"proposal_payload": {"action": "bypass_no_error", "tasks": []}})
            return {"messages": [proposal_message], "active_skills": active_skills}
        refined_tasks = [task_data if isinstance(task_data, Task) else Task(**task_data) for task_data in updated_tasks]
        proposal_data = {"action": "propose", "explanation": explanation, "task_type": updated_task_type, "tasks": [t.model_dump() for t in refined_tasks]}
    except Exception as e:
        proposal_data = {"action": "bypass_no_error", "explanation": f"Kích hoạt cơ chế tự phục hồi do lỗi hệ thống: {str(e)}", "task_type": task_type, "tasks": []}
        
    proposal_message = AIMessage(content=f"🔄 **[Hệ thống đề xuất lộ trình hành động mới]**\n\n{proposal_data.get('explanation', 'Đang cập nhật lộ trình...')}", name="replanner_proposal", additional_kwargs={"proposal_payload": proposal_data})
    return {"replanning_count": replanning_count + 1, "messages": [proposal_message], "active_skills": active_skills}

def replanner_interrupt_node(state: AgentState) -> Dict[str, Any]:
    messages = state["messages"]
    plan = state["plan"]
    proposal_payload = {}
    for msg in reversed(messages):
        if getattr(msg, "name", None) == "replanner_proposal":
            proposal_payload = msg.additional_kwargs.get("proposal_payload", {})
            break
            
    if not proposal_payload:
        logger.warning("[Replanner Gate] Không tìm thấy dữ liệu đề xuất từ replanner_node.")
        return {"error_logs": "", "attempts": 0, "modified_files": []}
        
    action = proposal_payload.get("action", "bypass_no_error")
    proposed_tasks = proposal_payload.get("tasks", [])
    if action in ["bypass_limit", "bypass_no_error"] or not proposed_tasks:
        fallback_tasks = [t if isinstance(t, Task) else Task(**t) for t in plan]
        logger.info("[Replanner Gate] Tự động duyệt qua kế hoạch rỗng hoặc lệnh bypass.")
        return {"plan": fallback_tasks, "error_logs": "", "attempts": 0, "modified_files": [], "messages": [AIMessage(content="⏭️ **[Tự động điều phối]**: Kế hoạch hiện tại đã tối ưu, hệ thống tự động hoàn tất pha duyệt.")]}
        
    interrupt_payload = {
        "type": "replanner_interrupt",
        "title": "📋 ĐÁNH GIÁ & PHÊ DUYỆT KẾ HOẠCH HÀNH ĐỘNG",
        "explanation": proposal_payload.get("explanation", "Hệ thống phát hiện cần thay đổi lộ trình để tiếp tục thực hiện."),
        "proposed_tasks": proposed_tasks,
        "prompt": (
            "Hệ thống đề xuất điều chỉnh lộ trình như trên.\n"
            "- Nhấn Approve (hoặc gửi phản hồi 'yes', chuỗi rỗng) để ĐỒNG Ý lộ trình mới.\n"
            "- Gửi phản hồi 'skip' hoặc 'no' để BỎ QUA và giữ nguyên lộ trình cũ.\n"
            "- Bạn cũng có thể sửa đổi danh sách Task trực tiếp trên giao diện để cấu hình kế hoạch tùy chỉnh."
        )
    }
    user_input = interrupt(interrupt_payload)
    user_input_clean = str(user_input).strip().lower() if user_input else ""
    if user_input is None or user_input_clean in ["", "yes", "approve", "ok"]:
        refined_tasks = [Task(**t) for t in proposed_tasks]
        return {"plan": refined_tasks, "task_type": proposal_payload.get("task_type", "development"), "error_logs": "", "attempts": 0, "modified_files": [], "messages": [AIMessage(content="✅ **[Kế hoạch được duyệt]** Áp dụng lộ trình phát triển và kiểm thử mới thành công.")]}
        
    if user_input_clean in ["skip", "no", "cancel", "decline"]:
        fallback_tasks = [t if isinstance(t, Task) else Task(**t) for t in plan]
        has_pending = any(t.status == "pending" for t in fallback_tasks)
        if not has_pending and fallback_tasks:
            fallback_tasks[-1].status = "pending"
        logger.info("[Replanner Gate] Người dùng từ chối kế hoạch mới. Sử dụng kế hoạch cũ.")
        return {"plan": fallback_tasks, "error_logs": "", "attempts": 0, "modified_files": [], "messages": [AIMessage(content="⏭️ **[Người dùng bỏ qua kế hoạch mới]** Tiếp tục lộ trình thực thi cũ.")]}
        
    custom_tasks_raw = []
    if isinstance(user_input, list):
        custom_tasks_raw = user_input
    elif isinstance(user_input, dict) and "tasks" in user_input:
        custom_tasks_raw = user_input["tasks"]
    elif isinstance(user_input, str):
        try:
            parsed = json.loads(user_input)
            if isinstance(parsed, list):
                custom_tasks_raw = parsed
            elif isinstance(parsed, dict) and "tasks" in parsed:
                custom_tasks_raw = parsed["tasks"]
        except json.JSONDecodeError:
            pass

    if custom_tasks_raw:
        try:
            custom_tasks = [Task(**t) for t in custom_tasks_raw]
            logger.info(f"[Replanner Gate] Áp dụng thành công kế hoạch tùy chỉnh gồm {len(custom_tasks)} tasks.")
            return {"plan": custom_tasks, "task_type": proposal_payload.get("task_type", "development"), "error_logs": "", "attempts": 0, "modified_files": [], "messages": [AIMessage(content=f"✏️ **[Kế hoạch tùy chỉnh]** Đã áp dụng lộ trình gồm {len(custom_tasks)} bước do bạn thiết lập.")]}
        except Exception as parse_err:
            logger.error(f"[Replanner Gate] Lỗi định dạng dữ liệu Task tùy chỉnh: {str(parse_err)}")
            
    logger.info(f"[Replanner Gate] Ghi nhận phản hồi văn bản tự do: {user_input}")
    feedback_message = HumanMessage(content=f"⚠️ Ý kiến điều chỉnh lộ trình từ người dùng:\n'{user_input}'\nHãy phân tích và cập nhật lại kế hoạch hành động tương ứng dựa trên ý kiến này.")
    fallback_tasks = [t if isinstance(t, Task) else Task(**t) for t in plan]
    return {"plan": fallback_tasks, "error_logs": f"Người dùng yêu cầu thay đổi lộ trình: {user_input}", "attempts": 0, "messages": [feedback_message]}