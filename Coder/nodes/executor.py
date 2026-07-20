# oder/nodes/executor.py
from pathlib import Path
from typing import Dict, Any, List
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config import (
    model,  sanitize_and_resolve_path,
   
)
from state import AgentState, Task
from prompts_loader import load_prompt
from skills_engine import AgentSkillsEngine
from tools import (
    ActivateSkillTool, RunSkillScriptTool, SearchKeywordTool, CompleteTaskTool,
    QueryOpenWikiTool, WebAutonomousExecutorTool, ChromeDebuggerTool, ReadFilesTool,
    ListDirectoryTool, UniversalSymbolSearchTool, ReadFileLinesTool, AskQuestionsTool,
    WriteFileTool, ApplyPatchTool, WriteAndRunScriptTool, RunTerminalTool
)
from .utils import find_extension_dir_heuristic, compact_historical_file_messages

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
        if t_status == "pending" and all(dep in completed_ids for dep in t_deps):
            eligible.append(t)
    return eligible

def executor_node(state: AgentState) -> Dict[str, Any]:
    ws = state.get("workspace_path", ".")
    plan = state.get("plan", [])
    error_logs = state.get("error_logs", "")
    file_registry = state.get("file_registry", {})
    messages = list(state["messages"])
    task_type = state.get("task_type", "development")
    extension_path = state.get("extension_path", "")
    active_skills = state.get("active_skills", {}) or {}
    debugger_proposal = state.get("debugger_proposal", "")

    state_updates = {}
    git_branch = state.get("git_branch", "")
    workspace_context = state.get("workspace_context", "")
    
    if not git_branch:
        from tools import GitManager
        git_dir = Path(ws) / ".git"
        if git_dir.exists():
            try:
                git_manager = GitManager(ws)
                git_branch = git_manager.init_and_prepare_branch()
            except Exception:
                git_branch = "no_git"
        else:
            git_branch = "no_git"
        state_updates["git_branch"] = git_branch

    if not workspace_context:
        context_path = Path(ws) / "CONTEXT.md"
        if context_path.exists():
            try:
                workspace_context = context_path.read_text(encoding="utf-8")
            except Exception:
                workspace_context = "Không thể đọc CONTEXT.md"
        else:
            workspace_context = "📋 Chưa có tệp cấu hình CONTEXT.md."
        state_updates["workspace_context"] = workspace_context

    if not extension_path:
        ext_dir = find_extension_dir_heuristic(Path(ws))
        if ext_dir:
            extension_path = ext_dir
            state_updates["extension_path"] = extension_path

    parsed_plan = [t if isinstance(t, Task) else Task(**t) for t in plan]
    eligible_tasks = get_eligible_tasks(parsed_plan)
    if not eligible_tasks:
        pending_tasks = [t for t in parsed_plan if t.status == "pending"]
        if pending_tasks:
            eligible_tasks = [pending_tasks[0]]
            
    tasks_str = "\n".join([f"- [{t.id}] {t.description}" for t in eligible_tasks]) if eligible_tasks else "- [Khảo sát tổng thể]: Tìm hiểu cấu trúc và giải quyết yêu cầu người dùng."

    skills_engine = AgentSkillsEngine(ws)
    catalog = skills_engine.scan_catalog()
    catalog_prompt = ""
    if catalog:
        catalog_prompt = "\n=== 📚 THƯ VIỆN KỸ NĂNG KHẢ DỤNG (TIER 1: CATALOG) ===\n"
        for item in catalog:
            catalog_prompt += f"- **{item['name']}**: {item['description']}\n"

    current_turn_tool_messages = [msg for msg in reversed(messages) if getattr(msg, "type", None) == "tool"]
    last_ai_message = next((msg for msg in reversed(messages) if isinstance(msg, AIMessage)), None)
    if last_ai_message and last_ai_message.tool_calls:
        tool_results_map = {tm.tool_call_id: tm for tm in current_turn_tool_messages}
        for tc in last_ai_message.tool_calls:
            if tc["name"] == "activate_agent_skill":
                tc_id = tc["id"]
                associated_tool_msg = tool_results_map.get(tc_id)
                if associated_tool_msg and "Lỗi" not in str(associated_tool_msg.content):
                    requested_skill = tc["args"].get("skill_name")
                    if requested_skill:
                        active_skills[requested_skill] = str(associated_tool_msg.content)

    active_skills_prompt = ""
    if active_skills:
        active_skills_prompt = "\n=== ⚡ CÁC KỸ NĂNG ĐANG HOẠT ĐỘNG (TIER 2: ACTIVE STATUS) ===\n"
        for s_name in active_skills.keys():
            desc = next((item["description"] for item in catalog if item["name"] == s_name), "") if catalog else ""
            active_skills_prompt += f"- **{s_name}** (Trạng thái: Đã kích hoạt): {desc or 'Kích hoạt thành công.'}\n"

    activate_skill_tool = ActivateSkillTool(workspace_path=ws)
    run_skill_script_tool = RunSkillScriptTool(workspace_path=ws)
    search_keyword_tool = SearchKeywordTool(workspace_path=ws)
    complete_task_tool = CompleteTaskTool(workspace_path=ws)
    query_openwiki = QueryOpenWikiTool(workspace_path=ws)
    web_autonomous_executor = WebAutonomousExecutorTool(workspace_path=ws)
    chrome_debugger_tool = ChromeDebuggerTool(workspace_path=ws)

    mcp_tools = []
    active_mcp_servers = state.get("active_mcp_servers", [])
    try:
        from mcp_helper import MCPRegistryManager
        mcp_manager = MCPRegistryManager(ws)
        mcp_tools = mcp_manager.get_tools_sync(active_servers=active_mcp_servers)
    except Exception as e:
        print(f"[Dynamic MCP Warning] Không thể nạp công cụ MCP: {str(e)}")

    has_pending_tasks = any(t.status == "pending" for t in parsed_plan)
    if not has_pending_tasks:
        return {
            "plan": parsed_plan,
            "messages": [AIMessage(content="🎉 **[Hệ thống tự động duyệt hoàn thành]**: Toàn bộ nhiệm vụ trong lộ trình đã hoàn thành thành công.")]
        }

    if task_type == "analysis":
        read_files = ReadFilesTool(workspace_path=ws)
        list_directory = ListDirectoryTool(workspace_path=ws)
        search_symbols = UniversalSymbolSearchTool(workspace_path=ws)
        read_file_lines = ReadFileLinesTool(workspace_path=ws)
        ask_questions_tool = AskQuestionsTool(workspace_path=ws)
        write_file = WriteFileTool(workspace_path=ws)
        apply_patch = ApplyPatchTool(workspace_path=ws)
        tools = [
            activate_skill_tool, run_skill_script_tool, web_autonomous_executor, chrome_debugger_tool,
            read_files, list_directory, search_symbols, read_file_lines, ask_questions_tool, 
            search_keyword_tool, write_file, apply_patch, complete_task_tool, query_openwiki
        ] + mcp_tools
        system_prompt_template = load_prompt("executor_analysis.txt")
    else:
        read_files = ReadFilesTool(workspace_path=ws)
        write_file = WriteFileTool(workspace_path=ws)
        apply_patch = ApplyPatchTool(workspace_path=ws)
        search_symbols = UniversalSymbolSearchTool(workspace_path=ws)
        list_directory = ListDirectoryTool(workspace_path=ws)
        run_terminal_command = RunTerminalTool(workspace_path=ws)
        read_file_lines = ReadFileLinesTool(workspace_path=ws)
        write_and_run_script = WriteAndRunScriptTool(workspace_path=ws)
        ask_questions_tool = AskQuestionsTool(workspace_path=ws)
        tools = [
            activate_skill_tool, run_skill_script_tool, read_files, write_file, web_autonomous_executor,
            apply_patch, list_directory, run_terminal_command, search_symbols, chrome_debugger_tool,
            read_file_lines, ask_questions_tool, write_and_run_script, search_keyword_tool, complete_task_tool, query_openwiki
        ] + mcp_tools
        system_prompt_template = load_prompt("executor_development.txt")

    system_prompt = system_prompt_template.replace("{tasks_str}", tasks_str).replace("{ws}", ws)
    updated_file_registry = dict(file_registry or {})
    cache_updated = False
    for f_path_key in list(updated_file_registry.keys()):
        try:
            safe_path = sanitize_and_resolve_path(ws, f_path_key, create_parent=False)
            if safe_path.exists() and safe_path.is_file():
                disk_content = safe_path.read_text(encoding="utf-8")
                if updated_file_registry[f_path_key] != disk_content:
                    updated_file_registry[f_path_key] = disk_content
                    cache_updated = True
            else:
                updated_file_registry.pop(f_path_key, None)
                cache_updated = True
        except Exception:
            pass

    if cache_updated:
        state_updates["file_registry"] = updated_file_registry
        file_registry = updated_file_registry

    system_instructions = system_prompt + catalog_prompt + active_skills_prompt
    if git_branch and git_branch != "no_git":
        system_instructions += f"\n- Nhánh Git đang hoạt động: `{git_branch}`"

    model_with_tools = model.bind_tools(tools)
    optimized_history = compact_historical_file_messages(messages)
    input_messages = [SystemMessage(content=system_instructions)]

    if workspace_context:
        context_body = "=== NGỮ CẢNH DỰ ÁN HIỆN HÀNH (WORKSPACE STATE) ===\n"
        context_body += f"\n--- THÔNG TIN NỀN TẢNG (CONTEXT.md) ---\n{workspace_context}\n"
        input_messages.append(SystemMessage(content=context_body))

    if task_type == "development" and error_logs:
        input_messages.append(HumanMessage(content=f"LƯU Ý SỬA LỖI TỪ VÒNG KIỂM THỬ:\n{error_logs}\nHãy sửa triệt để."))
        
    response = model_with_tools.invoke(input_messages + optimized_history)
    from .utils import sanitize_llm_response_content
    response = sanitize_llm_response_content(response)
    
    if not response.tool_calls:
        if task_type == "analysis":
            findings = [f"### Báo cáo khảo sát chủ động:\n{response.content}"] if response.content else []
            eligible_ids = {t.id for t in eligible_tasks}
            updated_plan = [t.model_copy(update={"status": "completed"}) if t.id in eligible_ids else t.model_copy() for t in parsed_plan]
            state_updates.update({
                "messages": [response], "plan": updated_plan,
                "last_executed_task_ids": list(eligible_ids), "active_skills": active_skills,
                "debugger_proposal": debugger_proposal
            })
            if findings:
                state_updates["step_findings"] = findings
            return state_updates
        else:
            has_executed_action = any(
                getattr(msg, "type", None) == "tool" and msg.name in ["write_file", "apply_search_replace_patch", "run_terminal_command", "run_skill_script", "complete_agent_task"]
                for msg in reversed(messages)
            )
            content_lower = response.content.lower() if response.content else ""
            explicitly_finished = any(kw in content_lower for kw in ["hoàn thành", "hoàn tất", "done", "finished", "đã hoàn thành"])
            if has_executed_action or explicitly_finished:
                eligible_ids = {t.id for t in eligible_tasks}
                updated_plan = [t.model_copy(update={"status": "completed"}) if t.id in eligible_ids else t.model_copy() for t in parsed_plan]
                state_updates.update({
                    "messages": [response], "plan": updated_plan,
                    "last_executed_task_ids": list(eligible_ids), "active_skills": active_skills,
                    "debugger_proposal": debugger_proposal
                })
                return state_updates
            else:
                warning_feedback = HumanMessage(content="⚠️ Cảnh báo: Bạn chưa gọi công cụ 'complete_agent_task' để hoàn thành nhiệm vụ này. Hãy gọi công cụ đó để kết thúc công việc.")
                state_updates.update({
                    "messages": [response, warning_feedback], "active_skills": active_skills,
                    "debugger_proposal": debugger_proposal
                })
                return state_updates
                
    state_updates.update({
        "messages": [response], "active_skills": active_skills, "debugger_proposal": debugger_proposal
    })
    return state_updates