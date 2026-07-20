# oder/nodes/synthesis.py
from pathlib import Path
from typing import Dict, Any
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

from config import fast_model
from state import AgentState
from prompts_loader import load_prompt

def synthesis_node(state: AgentState) -> Dict[str, Any]:
    ws = state["workspace_path"]
    findings = state.get("step_findings", [])
    git_branch = state.get("git_branch", "no_git")
    if not findings:
        return {"messages": [AIMessage(content="Không thu thập được thông tin khảo sát để tổng hợp.")]}
        
    compiled_data = "\n\n---\n\n".join(findings)
    is_non_coding = any(kw in compiled_data.lower() or kw in str(state.get("messages", [])).lower() for kw in ["notion", "figma", "jira", "trello", "google sheet", "excel", "devops", "ops", "tối ưu"])
    is_localized_edit = any(kw in compiled_data.lower() or "localized code edit" in compiled_data.lower() for kw in ["sửa lỗi cục bộ", "chỉnh sửa file", "localized"])
    
    if is_non_coding:
        return {"messages": [AIMessage(content="⏭️ **[Bypass Synthesis]**: Phát hiện tác vụ Phi lập trình / Vận hành (Ops). Hệ thống tự động bỏ qua bước thiết lập `CONTEXT.md` để bảo vệ bối cảnh đồ thị sạch.")]}
    if is_localized_edit and not (Path(ws) / "CONTEXT.md").exists():
        return {"messages": [AIMessage(content="⏭️ **[Bypass Synthesis]**: Phát hiện tác vụ chỉnh sửa tệp tin cục bộ. Bỏ qua việc tạo tệp `CONTEXT.md` mới để tối ưu bộ nhớ.")]}

    synthesis_prompt = load_prompt("synthesis.txt")
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
                from tools import WorkspaceTools, GitManager
                tools_mgr = WorkspaceTools(ws)
                tools_mgr.write_file("CONTEXT.md", cleaned_md)
                git_manager = GitManager(ws)
                git_manager._run_cmd(["git", "add", "CONTEXT.md"], ignore_error=True)
                message_content = "**Tổng hợp tài liệu hoàn tất:** Đã biên dịch tri thức khảo sát, lưu vật lý thành tệp `CONTEXT.md` và đưa vào Git staging thành công."
            else:
                message_content = "**Tổng hợp tài liệu hoàn tất (Chế độ In-Memory):** Tri thức khảo sát đã được tổng hợp và nạp trực tiếp vào ngữ cảnh trạng thái đồ thị (`workspace_context`). Tệp tin `CONTEXT.md` vật lý **không** được tạo trên ổ đĩa do hệ thống phát hiện không sử dụng Git."
            return {"workspace_context": cleaned_md, "messages": [AIMessage(content=message_content)]}
    except Exception as e:
        return {"messages": [AIMessage(content=f"Cảnh báo: Có lỗi xảy ra khi tổng hợp tệp CONTEXT.md: {str(e)}")]}
    return {}