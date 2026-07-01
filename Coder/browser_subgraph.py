# browser_subgraph.py
import os
import atexit
import tempfile
import uuid
import re # ĐÃ BỔ SUNG ĐỂ SỬ DỤNG REGEX PARSER DỰ PHÒNG
from pathlib import Path
from typing import Dict, Any, Literal, Optional
from urllib.parse import urlparse
from cloakbrowser import launch_persistent_context 
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from state import WebInteractionState
from config import model

class ElementSelection(BaseModel):
    selected_id: int = Field(description="ID của phần tử phù hợp nhất.")
    reason: str = Field(description="Lý do lựa chọn.")

# SOM_SCRIPT trích xuất đầy đủ thông tin trạng thái
SOM_SCRIPT = """
() => {
    const oldMarks = document.querySelectorAll('.ai-som-mark');
    oldMarks.forEach(el => el.remove());

    const interactiveSelectors = 'a, button, select, textarea, input, [role="button"], [role="checkbox"], [role="switch"]';
    const elements = document.querySelectorAll(interactiveSelectors);
    const registry = {};
    let idCounter = 1;

    elements.forEach(el => {
        const rect = el.getBoundingClientRect();
        if (rect.width > 0 && rect.height > 0 && window.getComputedStyle(el).display !== 'none') {
            const id = idCounter++;
            
            el.style.outline = '2px solid #ff00ff';
            el.style.position = 'relative';
            
            const badge = document.createElement('span');
            badge.className = 'ai-som-mark';
            badge.innerText = `[${id}]`;
            badge.style.position = 'absolute';
            badge.style.top = '0px';
            badge.style.left = '0px';
            badge.style.backgroundColor = '#ff00ff';
            badge.style.color = 'white';
            badge.style.fontSize = '11px';
            badge.style.fontWeight = 'bold';
            badge.style.padding = '1px 3px';
            badge.style.zIndex = '100000';
            el.appendChild(badge);

            let selector = el.tagName.toLowerCase();
            if (el.id) {
                selector = `#${el.id}`;
            } else if (el.getAttribute('aria-label')) {
                selector = `[aria-label="${el.getAttribute('aria-label')}"]`;
            } else if (el.className) {
                const cleanClasses = Array.from(el.classList).filter(c => c !== 'ai-som-mark').join('.');
                if (cleanClasses) selector += `.${cleanClasses}`;
            }

            const isActiveElement = (document.activeElement === el) || 
                                    el.classList.contains('active') || 
                                    el.classList.contains('selected') ||
                                    el.getAttribute('aria-selected') === 'true';

            registry[id] = {
                "id": id,
                "tagName": el.tagName,
                "text": el.innerText ? el.innerText.trim() : "",
                "aria_label": el.getAttribute('aria-label') || "",
                "selector": selector,
                "value": el.value || "",
                "classes": el.className || "",
                "checked": el.checked || false,
                "aria_selected": el.getAttribute('aria-selected') || "false",
                "style": el.getAttribute('style') || "",
                "isActive": isActiveElement
            };
        }
    });
    return registry;
}
"""


class BrowserSessionManager:
    """
    Quản lý khởi tạo trình duyệt CloakBrowser độc lập cho từng luồng.
    Không lưu đệm tĩnh để tránh xung đột Event Loop giữa các luồng làm việc của LangGraph.
    """
    @classmethod
    def create_page(cls, workspace_path: str, extension_path: Optional[str] = None) -> tuple:
        """
        Khởi tạo một context và page mới sạch sẽ trong luồng hiện tại.
        Mọi cookies và cookies/localstorage của extension đều được bảo toàn thông qua profile_dir vật lý.
        """
        ws_resolved = str(Path(workspace_path).resolve())
        ext_resolved = str(Path(extension_path).resolve()) if extension_path else None
        
        profile_id = uuid.uuid5(uuid.NAMESPACE_URL, ws_resolved).hex
        profile_dir = Path.home() / ".cloak_profiles" / profile_id
        profile_dir.mkdir(parents=True, exist_ok=True)

        extension_paths_list = [ext_resolved] if ext_resolved else None

        context = launch_persistent_context(
            user_data_dir=str(profile_dir),
            headless=False,       
            humanize=True,        
            geoip=True,           
            extension_paths=extension_paths_list 
        )
        
        page = context.pages[0] if context.pages else context.new_page()
        return context, page


def run_browser_session_sync(state: WebInteractionState) -> Dict[str, Any]:
    workspace = state.get("workspace_path", ".") 
    url = state["url"]
    action_type = state["action_type"]
    target_desc = state["target_description"]
    js_code = state.get("js_code_to_test")
    extension_path = state.get("extension_path") 
    
    temp_dir = Path(tempfile.gettempdir()) / "ai_agent_browser_sessions"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    unique_id = str(uuid.uuid4())[:8]
    screenshot_name = f"web_som_{unique_id}.png"
    screenshot_path = str(temp_dir / screenshot_name)
    
    result = {
        "detected_selectors": None,
        "execution_success": False,
        "dom_state_after": None,
        "screenshot_path": None,
        "browser_console_logs": None,
        "error": None
    }
    
    context = None
    try:
        # Khởi tạo một phiên sạch sẽ thuộc luồng hiện hành
        context, page = BrowserSessionManager.create_page(workspace, extension_path)
        
        browser_logs = []
        
        def log_handler(msg):
            browser_logs.append(f"[{msg.type.upper()}] {msg.text}")
            
        def err_handler(err):
            browser_logs.append(f"[RUNTIME_ERROR] {err.message}")
            
        page.on("console", log_handler)
        page.on("pageerror", err_handler)
        
        try:
            # Điều hướng trang web
            page.goto(url, wait_until="networkidle", timeout=15000)
            
            if extension_path:
                page.wait_for_timeout(4000)
            
            if action_type == "explore":
                registry = page.evaluate(SOM_SCRIPT)
                page.screenshot(path=screenshot_path)
                
                compact_dom = ""
                for el_id, info in registry.items():
                    compact_dom += (
                        f"ID [{el_id}]: {info.get('tagName', 'UNKNOWN')} | Text: '{info.get('text', '')}' | Label: '{info.get('aria_label', '')}' "
                        f"| Classes: '{info.get('classes', '')}' | Active: {info.get('isActive', False)} | Checked: {info.get('checked', False)}\n"
                    )
                
                # NÂNG CẤP LẬP TRÌNH PHÒNG THỦ: Sử dụng include_raw=True để tránh sập đồ thị khi gọi Local LLM Proxy
                structured_llm = model.with_structured_output(
                    ElementSelection, 
                    method="function_calling", 
                    include_raw=True
                )
                
                prompt = (
                    f"Người dùng muốn tìm phần tử: '{target_desc}'.\n"
                    f"Dưới đây là cấu trúc DOM rút gọn kèm thuộc tính trạng thái:\n{compact_dom}\n"
                    "Hãy chọn ra ID chính xác nhất bằng cách gọi hàm ElementSelection."
                )
                
                llm_output = structured_llm.invoke([
                    SystemMessage(content="Bạn là chuyên gia định vị phần tử giao diện. Hãy trả về một cấu trúc ElementSelection hợp lệ."),
                    HumanMessage(content=prompt)
                ])
                
                parsed = llm_output.get("parsed")
                raw_msg = llm_output.get("raw")
                parsing_error = llm_output.get("parsing_error")
                
                selected_id = None
                reason = "Không rõ lý do"
                
                # 🛡️ HỆ THỐNG PHÂN TÍCH CÚ PHÁP DỰ PHÒNG ĐA TẦNG (MULTI-TIER FALLBACK PARSER)
                if parsed is not None:
                    selected_id = parsed.selected_id
                    reason = parsed.reason
                else:
                    # Tầng 1: Kiểm tra nếu có tool_calls trong raw_msg nhưng LangChain parse lỗi
                    tool_calls = getattr(raw_msg, "tool_calls", []) if raw_msg else []
                    if tool_calls:
                        try:
                            args = tool_calls[0].get("args", {})
                            if "selected_id" in args:
                                selected_id = int(args["selected_id"])
                                reason = args.get("reason", "Trích xuất thành công từ tool_calls thô")
                        except Exception:
                            pass
                    
                    # Tầng 2: Nếu vẫn None, quét Regex trên raw_msg.content
                    if selected_id is None and raw_msg and hasattr(raw_msg, "content"):
                        raw_content = str(raw_msg.content)
                        match = re.search(r'"selected_id"\s*:\s*(\d+)', raw_content) or \
                                re.search(r'selected_id\s*[:=]\s*(\d+)', raw_content, re.IGNORECASE) or \
                                re.search(r'ID\s*\[?(\d+)\]?', raw_content, re.IGNORECASE)
                        
                        if match:
                            selected_id = int(match.group(1))
                            reason = f"Trích xuất dự phòng qua Regex (Lỗi gốc: {parsing_error})"
                        else:
                            # Tầng 3: Quét ID hợp lệ đầu tiên xuất hiện trong văn bản thô
                            all_nums = [int(n) for n in re.findall(r'\b\d+\b', raw_content)]
                            valid_nums = [n for n in all_nums if str(n) in registry]
                            if valid_nums:
                                selected_id = valid_nums[0]
                                reason = f"Trích xuất số ID khả dụng đầu tiên (Lỗi gốc: {parsing_error})"
                
                selected_id_str = str(selected_id) if selected_id is not None else ""
                
                if selected_id_str in registry:
                    result_selectors = registry[selected_id_str]
                    result_selectors["selection_reason"] = reason
                    result["detected_selectors"] = result_selectors
                    result["screenshot_path"] = screenshot_path
                    result["execution_success"] = True
                else:
                    # Tầng 4: Trả về lỗi chi tiết kèm nội dung thô để gỡ lỗi thay vì làm sập đồ thị
                    raw_dump = raw_msg.content if (raw_msg and hasattr(raw_msg, "content")) else str(raw_msg)
                    result["error"] = (
                        f"Không thể phân tích hoặc trích xuất ID hợp lệ từ phản hồi của LLM.\n"
                        f"Lỗi cú pháp (Parsing Error): {parsing_error}\n"
                        f"Nội dung thô từ mô hình: {raw_dump}"
                    )
                    
            elif action_type == "test_js" and js_code:
                try:
                    page.evaluate(f"() => {{ {js_code} }}")
                    page.wait_for_timeout(1500)
                    
                    page.screenshot(path=screenshot_path)
                    result["execution_success"] = True
                    result["screenshot_path"] = screenshot_path
                    
                    result["dom_state_after"] = {
                        "url_after": page.url
                    }
                except Exception as js_err:
                    result["error"] = f"Lỗi chạy thử JS: {str(js_err)}"
                    result["execution_success"] = False
                    
            if browser_logs:
                result["browser_console_logs"] = "\n".join(browser_logs[-20:]) 
                
        finally:
            try:
                page.remove_listener("console", log_handler)
                page.remove_listener("pageerror", err_handler)
            except Exception:
                pass
                
    except Exception as launch_err:
        result["error"] = f"Không thể lấy hoặc khởi chạy CloakBrowser: {str(launch_err)}"
    finally:
        # Đảm bảo đóng trình duyệt và giải phóng Event Loop sau khi tác vụ kết thúc
        if context:
            try:
                context.close()
            except Exception:
                pass
        
    return result

# Thiết lập đồ thị con đồng bộ
def web_executor_node(state: WebInteractionState) -> Dict[str, Any]:
    return run_browser_session_sync(state)

sub_builder = StateGraph(WebInteractionState)
sub_builder.add_node("web_executor", web_executor_node)
sub_builder.add_edge(START, "web_executor")
sub_builder.add_edge("web_executor", END)

web_subgraph = sub_builder.compile()