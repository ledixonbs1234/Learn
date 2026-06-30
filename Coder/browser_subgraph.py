# browser_subgraph.py
import os
import atexit
import tempfile
import uuid
from pathlib import Path
from typing import Dict, Any, Literal, Optional
from urllib.parse import urlparse
# Thay đổi quan trọng: Sử dụng launch của cloakbrowser thay cho sync_playwright
from cloakbrowser import launch 
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from state import WebInteractionState
from config import model

class ElementSelection(BaseModel):
    selected_id: int = Field(description="ID của phần tử phù hợp nhất.")
    reason: str = Field(description="Lý do lựa chọn.")

# (Giữ nguyên đoạn mã SOM_SCRIPT)
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
                "value": el.value || ""
            };
        }
    });
    return registry;
}
"""


class BrowserSessionManager:
    """
    Quản lý vòng đời của các phiên trình duyệt CloakBrowser.
    Duy trì kết nối Browser và Page liên tục theo từng workspace nhằm bảo toàn Cookies, Session và trạng thái DOM.
    """
    _sessions: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def get_or_create_page(cls, workspace_path: str) -> tuple:
        """
        Lấy ra browser và page đang hoạt động hoặc khởi chạy mới nếu chưa có.
        Trả về tuple: (browser_instance, page_instance, was_reused)
        """
        ws_resolved = str(Path(workspace_path).resolve())
        
        if ws_resolved in cls._sessions:
            session = cls._sessions[ws_resolved]
            try:
                # Kiểm tra kết nối đến page hoạt động ổn định
                _ = session["page"].url
                return session["browser"], session["page"], True
            except Exception:
                cls.close_session(ws_resolved)

        # Khởi chạy một trình duyệt CloakBrowser mới
        browser = launch(
            headless=False,       # Đặt False để hiển thị trực quan trên localhost
            humanize=True,        # Giả lập hành vi con người chống bot
            geoip=True            # Đồng bộ hóa múi giờ và locale
        )
        page = browser.new_page()
        
        cls._sessions[ws_resolved] = {
            "browser": browser,
            "page": page
        }
        return browser, page, False

    @classmethod
    def close_session(cls, workspace_path: str):
        """Đóng phiên trình duyệt tương ứng với workspace cụ thể."""
        ws_resolved = str(Path(workspace_path).resolve())
        if ws_resolved in cls._sessions:
            session = cls._sessions[ws_resolved]
            try:
                session["browser"].close()
            except Exception:
                pass
            del cls._sessions[ws_resolved]

    @classmethod
    def close_all(cls):
        """Dọn dẹp toàn bộ trình duyệt đang chạy trong bộ nhớ."""
        for ws in list(cls._sessions.keys()):
            cls.close_session(ws)

# Đăng ký đóng trình duyệt an toàn khi tiến trình python bị tắt đột ngột
atexit.register(BrowserSessionManager.close_all)


def run_browser_session_sync(state: WebInteractionState) -> Dict[str, Any]:
    workspace = state.get("workspace_path", ".") 
    url = state["url"]
    action_type = state["action_type"]
    target_desc = state["target_description"]
    js_code = state.get("js_code_to_test")
    
    # GIẢI PHÁP: Lưu trữ tệp tin tạm trong thư mục tạm của Hệ điều hành thay vì Workspace
    temp_dir = Path(tempfile.gettempdir()) / "ai_agent_browser_sessions"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Tạo tên ảnh duy nhất theo từng phiên chạy để tránh xung đột ghi đè
    unique_id = str(uuid.uuid4())[:8]
    screenshot_name = f"web_som_{unique_id}.png"
    screenshot_path = str(temp_dir / screenshot_name)
    
    result = {
        "detected_selectors": None,
        "execution_success": False,
        "dom_state_after": None,
        "screenshot_path": None,
        "error": None
    }
    
    try:
        # Lấy phiên trình duyệt hiện tại từ Manager dựa trên workspace làm định danh
        browser, page, reused = BrowserSessionManager.get_or_create_page(workspace)
        
        try:
            # QUY TẮC ĐIỀU HƯỚNG THÔNG MINH (SMART NAVIGATION)
            should_navigate = not reused
            if reused:
                # Nếu tái sử dụng, chỉ tải lại trang khi domain của URL đích khác với trang hiện thời.
                try:
                    current_parsed = urlparse(page.url)
                    target_parsed = urlparse(url)
                    if current_parsed.netloc != target_parsed.netloc:
                        should_navigate = True
                except Exception:
                    should_navigate = True
            
            if should_navigate:
                page.goto(url, wait_until="networkidle", timeout=15000)
            
            if action_type == "explore":
                # Vẽ nhãn SoM lên trang web
                registry = page.evaluate(SOM_SCRIPT)
                page.screenshot(path=screenshot_path)
                
                # Gọi LLM dạng cấu trúc để chọn phần tử phù hợp nhất
                compact_dom = ""
                for el_id, info in registry.items():
                    compact_dom += f"ID [{el_id}]: {info['tagName']} | Text: '{info['text']}' | Label: '{info['aria_label']}'\n"
                
                structured_llm = model.with_structured_output(ElementSelection, method="function_calling")
                prompt = (
                    f"Người dùng muốn tìm phần tử: '{target_desc}'.\n"
                    f"Dưới đây là cấu trúc DOM rút gọn:\n{compact_dom}\n"
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
                else:
                    result["error"] = f"LLM chọn ID {selected_id} (chuỗi: '{selected_id_str}') không tồn tại trong Registry. Các khóa hiện có: {list(registry.keys())}"
                    
            elif action_type == "test_js" and js_code:
                # Chạy thử nghiệm mã Javascript điều khiển
                try:
                    page.evaluate(f"() => {{ {js_code} }}")
                    page.wait_for_timeout(1500) # Chờ giao diện cập nhật trạng thái
                    
                    page.screenshot(path=screenshot_path)
                    result["execution_success"] = True
                    result["screenshot_path"] = screenshot_path
                    
                    # Lấy lại trạng thái sau khi thao tác
                    result["dom_state_after"] = {
                        "url_after": page.url
                    }
                except Exception as js_err:
                    result["error"] = f"Lỗi chạy thử JS: {str(js_err)}"
                    result["execution_success"] = False
                    
        except Exception as run_err:
            result["error"] = f"Sự cố tương tác trang web: {str(run_err)}"
            
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