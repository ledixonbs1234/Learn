# oder/nodes/compression.py
import json
import re
from pathlib import Path
from typing import Dict, Any
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage

from config import fast_model
from state import AgentState, Task
from .utils import sanitize_llm_response_content

def context_compressor_node(state: AgentState) -> Dict[str, Any]:
    messages = state.get("messages", [])
    ws = state.get("workspace_path", ".")
    active_skills = state.get("active_skills", {}) or {}
    workspace_root = Path(ws).expanduser().resolve()
    
    latest_task_summary = ""
    task_id = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc["name"] == "complete_agent_task":
                    latest_task_summary = tc["args"].get("summary", "")
                    task_id = tc["args"].get("task_id", "")
                    break
            if latest_task_summary:
                break

    new_summaries = [f"Task {task_id}: {latest_task_summary}"] if latest_task_summary else []
    triage_msg_id = next((msg.id for msg in messages if (msg.type == "ai" or isinstance(msg, AIMessage)) and ("Phân phối thông minh" in str(msg.content) or "Phân loại tác vụ" in str(msg.content))), None)
    final_summary_id = messages[-1].id if messages and (messages[-1].type == "ai" or isinstance(messages[-1], AIMessage)) and not getattr(messages[-1], "tool_calls", None) else None

    deletion_list = []
    has_kept_root_user_msg = False
    for msg in messages:
        if not msg.id:
            continue
        is_human = (msg.type == "human" or isinstance(msg, HumanMessage))
        if is_human and not has_kept_root_user_msg:
            has_kept_root_user_msg = True
            continue
        if not (is_human or msg.id == triage_msg_id or msg.id == final_summary_id):
            deletion_list.append(RemoveMessage(id=msg.id))

    compiled_context_parts = []
    prd_path = workspace_root / "PRD.md"
    if prd_path.exists():
        try:
            compiled_context_parts.append(f"### [YÊU CẦU SẢN PHẨM (PRD.md)]\n{prd_path.read_text(encoding='utf-8')}")
        except Exception:
            pass

    context_file_path = workspace_root / "CONTEXT.md"
    if context_file_path.exists():
        try:
            compiled_context_parts.append(f"### [KIẾN TRÚC HỆ THỐNG & THUẬT NGỮ (CONTEXT.md)]\n{context_file_path.read_text(encoding='utf-8')}")
        except Exception:
            pass

    historical_summaries = state.get("completed_task_summaries", []) + new_summaries
    if historical_summaries:
        history_block = "### [LỊCH SỬ THỰC THI PHIÊN CHẠY (BỘ NHỚ TẠM THỜI IN-MEMORY)]\n" + "\n".join([f"- {s}" for s in historical_summaries])
        compiled_context_parts.append(history_block)

    super_context = "\n\n---\n\n".join(compiled_context_parts)
    normalized_modified_set = set()
    for f in state.get("modified_files", []) or []:
        try:
            f_path = Path(f).resolve()
            if f_path.exists():
                try:
                    rel = str(f_path.relative_to(workspace_root))
                    normalized_modified_set.add(rel)
                except ValueError:
                    normalized_modified_set.add(str(f_path))
        except Exception:
            pass

    current_registry = state.get("file_registry", {}) or {}
    registry_eviction_updates = {cached_file: None for cached_file in current_registry.keys() if cached_file not in normalized_modified_set}

    cleaned_active_skills = dict(active_skills)
    discovery_triad = ["grill-with-docs", "write-a-prd", "domain-modeling"]
    for skill_name in discovery_triad:
        cleaned_active_skills.pop(skill_name, None)
    
    clean_checkpoint_msg = AIMessage(
        content=(
            f"🔄 **[Hệ thống nén ngữ cảnh]**:\n"
            f"- Đã lưu tóm tắt nhiệm vụ `{task_id or 'N/A'}` vào bộ đệm trạng thái tạm thời.\n"
            f"- Đã thu gọn và giải phóng thành công lịch sử tin nhắn rác khỏi bộ nhớ RAM.\n"
            f"- Giải phóng {len(registry_eviction_updates)} tệp tin không thay đổi khỏi File Registry."
        )
    )
    return {
        "workspace_context": super_context, "messages": deletion_list + [clean_checkpoint_msg],
        "active_skills": cleaned_active_skills, "completed_task_summaries": new_summaries,
        "file_registry": registry_eviction_updates
    }

def fluxmem_distillation_node(state: AgentState) -> Dict[str, Any]:
    messages = state.get("messages", [])
    ws = state["workspace_path"]
    plan = state.get("plan", [])
    
    latest_task_id = ""
    latest_task_desc = "Tác vụ thực thi hệ thống"
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc["name"] == "complete_agent_task":
                    latest_task_id = tc["args"].get("task_id", "")
                    break
            if latest_task_id:
                break
    if not latest_task_id:
        return {}
        
    for t in plan:
        t_id = t.id if isinstance(t, Task) else t.get("id")
        if t_id == latest_task_id:
            latest_task_desc = t.description if isinstance(t, Task) else t.get("description")
            break

    current_task_messages = []
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and "📊 [Hệ thống Phân phối thông minh]" in str(msg.content):
            break
        current_task_messages.insert(0, msg)
        
    steps_prompt_parts = []
    for m in current_task_messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                steps_prompt_parts.append(f"- **AI đã gọi công cụ:** `{tc['name']}` với các đối số: `{json.dumps(tc['args'], ensure_ascii=False)}`")
        elif m.type == "tool":
            content_str = str(m.content)
            if len(content_str) > 800:
                content_str = content_str[:800] + "... [Đã cắt bớt dữ liệu thô dài] ..."
            steps_prompt_parts.append(f"  -> *Kết quả phản hồi của công cụ:* {content_str}")

    steps_prompt = "\n".join(steps_prompt_parts) if steps_prompt_parts else "Không ghi nhận lượt gọi công cụ trực tiếp."
    system_prompt = (
        "Bạn là Chuyên gia chưng cất kỹ năng (FluxMem Distillation Agent).\n"
        "Dưới đây là vết lịch sử chạy thực tế thành công của Agent:\n\n"
        f"[MỤC TIÊU]: \"{latest_task_desc}\"\n"
        "[HÀNH ĐỘNG CHI TIẾT]:\n{steps_prompt}\n\n"
        "Hãy viết một quy trình (Procedural Skill) gồm 3-5 bước tổng quát hóa, mô tả chính xác cách thiết thiết kế, biên tập, các công cụ tối ưu cần gọi và cách phòng ngừa lỗi biên dịch cho mục tiêu này.\n"
        "Chỉ trả về nội dung quy trình dạng Markdown, không viết thêm lời mở đầu hay giải thích"
    ).replace("{steps_prompt}", steps_prompt)

    try:
        distilled_response = fast_model.invoke([SystemMessage(content=system_prompt)])
        procedural_skill_md = distilled_response.content
        wiki_dir = Path("~/.openwiki/wiki").expanduser().resolve()
        wiki_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            slug_prompt = (
                "Dựa vào mô tả nhiệm vụ sau, hãy tạo ra một định danh (slug) ngắn gọn từ 2-4 từ, viết thường, KHÔNG dấu, phân cách bằng duy nhất dấu gạch dưới, mô tả khái quát kỹ năng kỹ thuật cốt lõi của tác vụ này.\n"
                "⚠️ YÊU CẦU NGHIÊM NGẶT:\n- TUYỆT ĐỐI KHÔNG bao gồm bất kỳ đường dẫn thư mục, ổ đĩa C:/, tên file cục bộ, hoặc tên người dùng nào.\n- Nếu nhiệm vụ là cào dữ liệu, slug có thể là 'web_data_scraping'.\n- Nếu nhiệm vụ là cấu hình database, slug có thể là 'database_configuration'.\n\nMô tả nhiệm vụ ban đầu: {latest_task_desc}\n\nChỉ trả về chuỗi định danh duy nhất (ví dụ: 'setup_playwright_scraper', 'write_prd_specification'):"
            ).replace("{latest_task_desc}", latest_task_desc)
            slug_response = fast_model.invoke([SystemMessage(content=slug_prompt)])
            slug_response = sanitize_llm_response_content(slug_response)
            raw_slug_content = slug_response.content.strip().lower()
            slug_candidates = re.findall(r'\b[a-z0-9_]{3,40}\b', raw_slug_content)
            system_stopwords = {"thinking", "the", "user", "is", "asking", "me", "to", "create", "a", "slug", "identifier", "based", "on", "task", "description", "markdown", "python", "task_id", "procedural", "skill", "would", "be", "something", "like"}
            safe_candidates = [c for c in slug_candidates if c not in system_stopwords]
            safe_task_name = safe_candidates[-1][:45] if safe_candidates else f"procedural_task_{latest_task_id.lower()}"
        except Exception as slug_err:
            print(f"[Cảnh báo] Lỗi sinh slug bằng AI: {str(slug_err)}. Chuyển sang fallback phòng ngự.")
            safe_task_name = f"procedural_task_{latest_task_id.lower()}"
            
        file_name = f"skill_{safe_task_name}.md"
        dest_file = wiki_dir / file_name
        dest_file.write_text(procedural_skill_md, encoding="utf-8")
        return {"messages": [AIMessage(content=f"⚡ **[Global FluxMem Distillation]**: Đã chưng cất tri thức thành công! Lưu trữ vật lý tại OpenWiki toàn cục: `~/.openwiki/wiki/{file_name}`.")]}
    except Exception as e:
        print(f"[Cảnh báo] Lỗi trong quá trình chưng cất quy trình toàn cục: {str(e)}")
        return {}