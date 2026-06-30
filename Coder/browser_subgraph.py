# browser_subgraph.py
import os
import atexit
import tempfile
import uuid
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

# NÂNG CẤP SOM_SCRIPT ĐỂ TRÍCH XUẤT TRẠNG THÁI HOẠT ĐỘNG CỦA CÁC PHẦN TỬ
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

            registry[id] = {
                "id": id,
                "tagName": el.tagName,
                "text": el.innerText ? el.innerText.trim() : "",
                "aria_label": el.getAttribute('aria-label') || "",
                "selector": selector,
                "value": el.value || "",
                // BỔ SUNG CÁC TRƯỜNG TRẠNG THÁI PHỤC VỤ CƠ CHẾ ASSERTION
                "classes": el.className || "",
                "checked": el.checked || false,
                "aria_selected": el.getAttribute('aria-selected') || "false",
                "style": el.getAttribute('style') || ""
            };
        }
    });
    return registry;
}
"""


class BrowserSessionManager:
    """
    Quản lý vòng đời của các phiên trình duyệt CloakBrowser.
    Hỗ trợ nạp Chrome Extension và kiểm tra thay đổi cấu hình nạp giữa các lượt chạy.
    """
    _sessions: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def get_or_create_page(cls, workspace_path: str, extension_path: Optional[str] = None) -> tuple:
        """
        Lấy ra context và page đang hoạt động.
        Nếu thư mục extension_path thay đổi so với phiên trước, tự động đóng phiên cũ và khởi chạy lại.
        """
        ws_resolved = str(Path(workspace_path).resolve())
        ext_resolved = str(Path(extension_path).resolve()) if extension_path else None
        
        if ws_resolved in cls._sessions:
            session = cls._sessions[ws_resolved]
            # Nếu đường dẫn extension thay đổi so với phiên đang chạy, đóng phiên cũ để khởi chạy lại
            if session.get("loaded_extension_path") != ext_resolved:
                cls.close_session(ws_resolved)
            else:
                try:
                    _ = session["page"].url
                    return session["context"], session["page"], True
                except Exception:
                    cls.close_session(ws_resolved)

        profile_id = uuid.uuid5(uuid.NAMESPACE_URL, ws_resolved).hex
        profile_dir = Path.home() / ".cloak_profiles" / profile_id
        profile_dir.mkdir(parents=True, exist_ok=True)

        # Chuẩn bị tham số nạp extension cho cloakbrowser
        extension_paths_list = [ext_resolved] if ext_resolved else None

        # Khởi chạy trình duyệt CloakBrowser dưới dạng Persistent Context
        context = launch_persistent_context(
            user_data_dir=str(profile_dir),
            headless=False,       # Đặt False để hiển thị trực quan và kích hoạt Extension
            humanize=True,        # Giả lập hành vi con người chống bot
            geoip=True,           # Đồng bộ hóa múi giờ và locale
            extension_paths=extension_paths_list # Nạp trực tiếp qua thư viện cloakbrowser
        )
        
        page = context.pages[0] if context.pages else context.new_page()
        
        cls._sessions[ws_resolved] = {
            "context": context,
            "page": page,
            "loaded_extension_path": ext_resolved
        }
        return context, page, False

    @classmethod
    def close_session(cls, workspace_path: str):
        ws_resolved = str(Path(workspace_path).resolve())
        if ws_resolved in cls._sessions:
            session = cls._sessions[ws_resolved]
            try:
                session["context"].close()
            except Exception:
                pass
            del cls._sessions[ws_resolved]

    @classmethod
    def close_all(cls):
        for ws in list(cls._sessions.keys()):
            cls.close_session(ws)

atexit.register(BrowserSessionManager.close_all)


def run_browser_session_sync(state: WebInteractionState) -> Dict[str, Any]:
    workspace = state.get("workspace_path", ".") 
    url = state["url"]
    action_type = state["action_type"]
    target_desc = state["target_description"]
    js_code = state.get("js_code_to_test")
    extension_path = state.get("extension_path") # Lấy ra từ State
    
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
    
    try:
        # Lấy phiên trình duyệt hiện tại và nạp extension nếu có
        context, page, reused = BrowserSessionManager.get_or_create_page(workspace, extension_path)
        
        # Thiết lập cơ chế ghi nhận console logs để tránh rò rỉ bộ nhớ
        browser_logs = []
        
        def log_handler(msg):
            browser_logs.append(f"[{msg.type.upper()}] {msg.text}")
            
        def err_handler(err):
            browser_logs.append(f"[RUNTIME_ERROR] {err.message}")
            
        page.on("console", log_handler)
        page.on("pageerror", err_handler)
        
        try:
            should_navigate = not reused
            if reused:
                try:
                    current_parsed = urlparse(page.url)
                    target_parsed = urlparse(url)
                    if current_parsed.netloc != target_parsed.netloc:
                        should_navigate = True
                except Exception:
                    should_navigate = True
            
            if should_navigate:
                page.goto(url, wait_until="networkidle", timeout=15000)
            
            # TRỄ ĐỘNG (DYNAMIC WAIT): Đợi Content Script của Extension thực thi hành vi can thiệp DOM
            if extension_path:
                page.wait_for_timeout(4000)
            
            if action_type == "explore":
                # Vẽ nhãn SoM lên trang web
                registry = page.evaluate(SOM_SCRIPT)
                page.screenshot(path=screenshot_path)
                
                # Gọi LLM dạng cấu trúc để chọn phần tử phù hợp nhất
                compact_dom = ""
                for el_id, info in registry.items():
                    compact_dom += (
                        f"ID [{el_id}]: {info['tagName']} | Text: '{info['text']}' | Label: '{info['aria_label']}' "
                        f"| Classes: '{info['classes']}' | Active: {info['isActive']} | Checked: {info['checked']}\n"
                    )
                
                structured_llm = model.with_structured_output(ElementSelection, method="function_calling")
                prompt = (
                    f"Người dùng muốn tìm phần tử: '{target_desc}'.\n"
                    f"Dưới đây là cấu trúc DOM rút gọn kèm thuộc tính trạng thái:\n{compact_dom}\n"
                    "Hãy chọn ra ID chính xác nhất."
                )
                
                response = structured_llm.invoke([
                    SystemMessage(content="Bạn là chuyên gia định vị phần tử giao diện."),
                    HumanMessage(content=prompt)
                ])
                
                selected_id = response.selected_id
                selected_id_str = str(selected_id) 
                
                if selected_id_str in registry:
                    result["detected_selectors"] = registry[selected_id_str]
                    result["screenshot_path"] = screenshot_path
                    result["execution_success"] = True
                else:
                    result["error"] = f"LLM chọn ID {selected_id} (chuỗi: '{selected_id_str}') không tồn tại trong Registry. Các khóa hiện có: {list(registry.keys())}"
                    
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
                    
            # Trích xuất toàn bộ log của phiên chạy này
            if browser_logs:
                result["browser_console_logs"] = "\n".join(browser_logs[-20:]) # Chỉ lấy 20 dòng cuối cùng
                
        finally:
            # Ngăn ngừa Memory Leak: Gỡ bỏ Listener sau khi kết thúc vòng đời của request kiểm thử
            try:
                page.remove_listener("console", log_handler)
                page.remove_listener("pageerror", err_handler)
            except Exception:
                pass
                
    except Exception as launch_err:
        result["error"] = f"Không thể lấy hoặc khởi chạy CloakBrowser: {str(launch_err)}"
        
    return result

# Thiết lập đồ thị con đồng bộ
def web_executor_node(state: WebInteractionState) -> Dict[str, Any]:
    return run_browser_session_sync(state)

sub_builder = StateGraph(WebInteractionState)
sub_builder.add_node("web_executor", web_executor_node)
sub_builder.add_edge(START, "web_executor")
sub_builder.add_edge("web_executor", END)

web_subgraph = sub_builder.compile()