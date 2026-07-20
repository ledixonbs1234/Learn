# oder/nodes/doubt.py
import os
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langgraph.types import interrupt

from config import fast_model, sanitize_and_resolve_path
from state import AgentState
from prompts_loader import load_prompt

def doubt_reviewer_node(state: AgentState) -> Dict[str, Any]:
    modified_files = state.get("modified_files", [])
    file_registry = state.get("file_registry", {})
    doubt_attempts = state.get("doubt_attempts", 0)
    if not modified_files or doubt_attempts >= 3:
        return {"doubt_findings": ""}
        
    latest_file = modified_files[-1]
    artifact_code = file_registry.get(latest_file, "")
    if not artifact_code:
        try:
            safe_path = sanitize_and_resolve_path(state["workspace_path"], latest_file)
            if safe_path.exists():
                artifact_code = safe_path.read_text(encoding="utf-8")
        except Exception:
            pass
    if not artifact_code:
        return {"doubt_findings": ""}

    adversarial_prompt_template = load_prompt("doubt_reviewer.txt")
    adversarial_prompt = adversarial_prompt_template.replace("{latest_file}", latest_file) + "\n```\n" + artifact_code + "\n```"
    response = fast_model.invoke([
        SystemMessage(content="Bạn đang thực thi quy trình thẩm định đối kháng thuộc kỹ năng `doubt-driven-development`."),
        HumanMessage(content=adversarial_prompt)
    ])
    return {"doubt_findings": response.content, "doubt_attempts": doubt_attempts + 1}

def doubt_gate_node(state: AgentState) -> Dict[str, Any]:
    doubt_findings = state.get("doubt_findings", "")
    modified_files = state.get("modified_files", [])
    if not doubt_findings or not modified_files:
        return {}
    latest_file = modified_files[-1]
    interrupt_payload = {
        "title": "🔍 THẨM ĐỊNH ĐỐI KHÁNG (DOUBT-DRIVEN DEVELOPMENT)",
        "file_under_review": latest_file,
        "single_model_findings": doubt_findings,
        "prompt": (
            "Hệ thống phát hiện một số rủi ro logic tiềm ẩn trong đoạn mã nguồn bạn vừa viết.\n"
            "- Nhấn Approve (hoặc gửi phản hồi rỗng/'yes'/'ok') để BỎ QUA và tiến hành commit.\n"
            "- Nhập 'gemini' hoặc 'codex' để kích hoạt THẨM ĐỊNH CHÉO ngoại vi thông qua mô hình độc lập (Cross-Model Escalation).\n"
            "- Nhập ý kiến phản hồi khác hoặc yêu cầu sửa lỗi để chuyển thông tin này quay lại cho Executor khắc phục lỗi."
        )
    }
    user_input = interrupt(interrupt_payload)
    user_action = str(user_input).strip().lower() if user_input else ""
    if user_action in ["", "yes", "approve", "ok", "skip"]:
        return {
            "error_logs": "", "doubt_findings": "",
            "messages": [AIMessage(content="✅ **[Doubt Bypassed]** Người dùng đã phê duyệt mã nguồn. Tiến hành hoàn tất tác vụ.")]
        }
    if user_action in ["gemini", "codex"]:
        cli_tool = user_action
        cli_executable = "gemini" if cli_tool == "gemini" else "codex"
        if not shutil.which(cli_executable):
            return {
                "error_logs": f"Không tìm thấy công cụ ngoại vi `{cli_executable}`",
                "messages": [HumanMessage(content=f"⚠️ Lỗi: Không tìm thấy thực thi CLI `{cli_executable}` trong biến môi trường PATH của bạn.")]
            }
        file_registry = state.get("file_registry", {})
        artifact_code = file_registry.get(latest_file, "")
        cross_prompt = f"Thẩm định đối kháng chéo (Adversarial Cross-Model Review) cho file {latest_file}.\nHãy tìm ra các lỗ hổng, lỗi logic hoặc điểm chưa tối ưu mà mô hình trước đã bỏ qua.\n\nMÃ NGUỒN:\n{artifact_code}"
        temp_prompt_file_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8", delete=False) as temp_prompt_file:
                temp_prompt_file.write(cross_prompt)
                temp_prompt_file_path = temp_prompt_file.name
            with open(temp_prompt_file_path, "r", encoding="utf-8") as stdin_file:
                env_copy = os.environ.copy()
                env_copy["PYTHONIOENCODING"] = "utf-8"
                env_copy["PYTHONUTF8"] = "1"
                cmd = [cli_executable]
                if cli_executable == "gemini":
                    cmd.extend(["--approval-mode", "plan", "-p", ""])
                elif cli_executable == "codex":
                    cmd.extend(["exec", "--sandbox", "read-only", "-"])
                res = subprocess.run(
                    cmd, stdin=stdin_file, capture_output=True, text=True,
                    encoding="utf-8", errors="replace", env=env_copy, timeout=45
                )
            combined_output = (res.stdout or "") + "\n" + (res.stderr or "")
            escalated_findings = combined_output.strip()
            return {
                "doubt_findings": f"🛡️ **[KẾT QUẢ THẨM ĐỊNH CHÉO TỪ {cli_executable.upper()}]**:\n\n{escalated_findings}",
                "messages": [AIMessage(content=f"🔍 Kích hoạt thành công rà soát chéo từ {cli_executable.upper()}. Đang chờ ý kiến phê duyệt cuối cùng.")]
            }
        except Exception as err:
            return {
                "error_logs": f"Lỗi hệ thống khi khởi chạy Cross-Model: {str(err)}",
                "messages": [AIMessage(content=f"❌ Thao tác gọi mô hình chéo thất bại: {str(err)}")]
            }
        finally:
            if temp_prompt_file_path and Path(temp_prompt_file_path).exists():
                try:
                    Path(temp_prompt_file_path).unlink(missing_ok=True)
                except Exception:
                    pass
    feedback_message = HumanMessage(
        content=f"⚠️ Yêu cầu sửa đổi mã nguồn dựa trên kết quả thẩm định đối kháng:\nÝ kiến người dùng: '{user_input}'\nCác lỗi cần khắc phục:\n{doubt_findings}"
    )
    return {"error_logs": f"Cần khắc phục lỗi logic thẩm định: {user_input}", "doubt_findings": "", "messages": [feedback_message]}