# oder/nodes/debugger.py
import os
import re
from pathlib import Path
from typing import Dict, Any
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

from config import model, GitIgnoreMatcher
from state import AgentState
from prompts_loader import load_prompt
from .utils import get_text_content_safely

def crawl_project_sources(workspace_path: Path) -> str:
    matcher = GitIgnoreMatcher(workspace_path)
    sources = []
    allowed_extensions = {
        '.py', '.js', '.jsx', '.ts', '.tsx', '.dart', '.go', '.rs', '.java',
        '.cpp', '.h', '.hpp', '.cs', '.kt', '.swift', '.json', '.yaml', '.yml',
        '.md', '.html', '.css', '.toml', '.xml', '.gradle', '.bat', '.sh'
    }
    try:
        for root, dirs, files in os.walk(str(workspace_path)):
            dirs[:] = [d for d in dirs if not matcher.is_ignored(Path(root) / d)]
            for file in files:
                file_path = Path(root) / file
                if matcher.is_ignored(file_path) or file_path.suffix.lower() not in allowed_extensions:
                    continue
                try:
                    content = file_path.read_text(encoding='utf-8', errors='replace')
                    rel_path = file_path.relative_to(workspace_path).as_posix()
                    sources.append(f"=== TỆP TIN: `{rel_path}` ===\n{content}\n")
                except Exception:
                    pass
    except Exception as e:
        print(f"[Cảnh báo Debugger] Lỗi duyệt thư mục: {str(e)}")
    return "\n\n".join(sources)

def run_isolated_debugger_agent(workspace_path: str, user_query: str) -> str:
    print("🤖 [Debugger Agent] Đang phân tích toàn bộ mã nguồn của dự án...")
    sources_data = crawl_project_sources(Path(workspace_path))
    if not sources_data:
        return "Không phát hiện mã nguồn hợp lệ hoặc thư mục dự án trống."
        
    prompt_template = load_prompt("isolated_debugger.txt")
    prompt = prompt_template.replace("{sources_data}", sources_data).replace("{user_query}", user_query)
    
    try:
        response = model.invoke([
            SystemMessage(content="Bạn đang thực hiện nhiệm vụ gỡ lỗi cô lập, một chu kỳ suy nghĩ duy nhất (One-shot debug)."),
            HumanMessage(content=prompt)
        ])
        cleaned_content = re.sub(r"<thinking>.*?</thinking>", "", response.content, flags=re.DOTALL)
        cleaned_content = re.sub(r"<thought>.*?</thought>", "", cleaned_content, flags=re.DOTALL)
        return cleaned_content.strip()
    except Exception as e:
        return f"Lỗi trong quá trình chạy Debugger Agent độc lập: {str(e)}"

def isolated_debugger_node(state: AgentState) -> Dict[str, Any]:
    ws = state.get("workspace_path", ".")
    messages = state.get("messages", [])
    debugger_proposal = state.get("debugger_proposal", "")
    if debugger_proposal:
        return {}

    user_query = ""
    for msg in reversed(messages):
        if msg.type == "human" or isinstance(msg, HumanMessage):
            user_query = get_text_content_safely(msg.content)
            break

    bug_keywords = ["lỗi", "error", "bug", "crash", "exception", "failed", "sửa lỗi", "fix", "hỏng"]
    detailed_analysis = state.get("detailed_analysis", "")
    error_logs = state.get("error_logs", "")
    is_bug_fixing_task = (
        any(kw in user_query.lower() for kw in bug_keywords) or
        any(kw in detailed_analysis.lower() for kw in bug_keywords) or
        bool(error_logs)
    )
    if not is_bug_fixing_task:
        return {"debugger_proposal": ""}

    proposal_result = run_isolated_debugger_agent(ws, user_query or detailed_analysis or error_logs)
    proposal_msg = AIMessage(
        content=(
            "=== [BẢN PHÂN TÍCH & PHƯƠNG ÁN SỬA LỖI TỐI ƯU CỦA TÔI] ===\n"
            "Sau khi phân tích toàn bộ mã nguồn của dự án một cách độc lập, tôi đã xác định được nguyên nhân và đúc kết được phương án giải quyết tối ưu dưới đây:\n\n"
            f"{proposal_result}\n\n"
            "Bây giờ, tôi sẽ bắt đầu gọi các công cụ chỉnh sửa tệp tin thích hợp để áp dụng giải pháp này."
        )
    )
    return {"debugger_proposal": proposal_result, "messages": [proposal_msg]}