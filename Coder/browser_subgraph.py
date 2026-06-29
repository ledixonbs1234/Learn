# browser_subgraph.py
import os
from typing import Dict, Any, Literal
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

def run_browser_session_sync(state: WebInteractionState) -> Dict[str, Any]:
    url = state["url"]
    action_type = state["action_type"]
    target_desc = state["target_description"]
    js_code = state.get("js_code_to_test")
    
    screenshot_name = "web_marked_som.png"
    screenshot_path = os.path.abspath(screenshot_name)
    
    result = {
        "detected_selectors": None,
        "execution_success": False,
        "dom_state_after": None,
        "screenshot_path": None,
        "error": None
    }
    
    # KHỞI CHẠY CLOAKBROWSER THAY THẾ CHO PLAYWRIGHT
    # Bật chế độ 'humanize=True' để mô phỏng chính xác hành vi di chuột, cuộn trang của con người.
    # Đặt headless=False (nếu muốn quan sát trực quan trên localhost) hoặc headless=True tùy nhu cầu của bạn.
    try:
        browser = launch(
            headless=False,       # Đặt False để trình duyệt hiện lên trực quan trên máy của bạn
            humanize=True,        # Kích hoạt giả lập hành vi con người cấp độ C++
            geoip=True            # Tự động đồng bộ hóa múi giờ và locale phù hợp
        )
        
        # Tạo trang mới giống hệt như API chuẩn của Playwright
        page = browser.new_page()
        
        try:
            page.goto(url, wait_until="networkidle", timeout=25000)
            
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
                if selected_id in registry:
                    result["detected_selectors"] = registry[selected_id]
                    result["screenshot_path"] = screenshot_path
                else:
                    result["error"] = f"LLM chọn ID {selected_id} không tồn tại trong Registry."
                    
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
        finally:
            # Đóng trình duyệt CloakBrowser sau khi xử lý xong
            browser.close()
            
    except Exception as launch_err:
        result["error"] = f"Không thể khởi chạy CloakBrowser: {str(launch_err)}"
        
    return result

# Thiết lập đồ thị con đồng bộ
def web_executor_node(state: WebInteractionState) -> Dict[str, Any]:
    return run_browser_session_sync(state)

sub_builder = StateGraph(WebInteractionState)
sub_builder.add_node("web_executor", web_executor_node)
sub_builder.add_edge(START, "web_executor")
sub_builder.add_edge("web_executor", END)

web_subgraph = sub_builder.compile()