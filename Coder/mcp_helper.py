# mcp_helper.py
import json
import os
import sys
import platform
import asyncio
from typing import List
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage, AIMessage

from config import sanitize_tool_result_content

def get_default_browser_profile_dir() -> str:
    """Tự động phát hiện đường dẫn User Data an toàn dựa trên Hệ điều hành."""
    system = platform.system()
    home = os.path.expanduser("~")
    
    if system == "Windows":
        # Ưu tiên Edge hoặc Chrome mặc định
        edge_path = os.path.join(home, "AppData", "Local", "Microsoft", "Edge", "User Data")
        if os.path.exists(edge_path):
            return edge_path
        return os.path.join(home, "AppData", "Local", "Google", "Chrome", "User Data")
    elif system == "Darwin":  # macOS
        return os.path.join(home, "Library", "Application Support", "Google", "Chrome")
    else:  # Linux
        return os.path.join(home, ".config", "google-chrome")
async def run_agent_with_flutter_skill_mcp(model, prompt_message: str, chat_history: List[BaseMessage] = None, workspace_path: str = "."):
    """
    Khởi chạy flutter_skill dưới dạng MCP Server và nạp các công cụ E2E 
    vào ngữ cảnh của AI Agent để tương tác trực tiếp với ứng dụng.
    """
    # Các công cụ có khả năng kích hoạt kết nối mới để mở khóa các công cụ tương tác sâu
    CONNECTION_TOOLS = {
        "connect_app", 
        "launch_app", 
        "scan_and_connect", 
        "connect_cdp", 
        "connect_openclaw_browser", 
        "connect_webmcp"
    }

    server_params = StdioServerParameters(
        command="flutter_skill",
        args=["server"]
    )
    
    if chat_history is None:
        chat_history = []
        
    system_prompt = (
        "Bạn là một chuyên gia kiểm thử tự động hóa sử dụng flutter-skill MCP.\n"
        "Bạn có quyền tương tác với ứng dụng (Flutter/React Native/Web) thông qua cây hỗ trợ tiếp cận (Accessibility Tree).\n"
        "Hãy thực hiện các bước kiểm thử theo yêu cầu bằng ngôn ngữ tự nhiên thông qua các công cụ có sẵn.\n"
        "Ưu tiên sử dụng cây phần tử để tìm chính xác nút bấm hoặc ô nhập liệu thay vì đoán mò."
    )
    
    messages = [SystemMessage(content=system_prompt)] + chat_history
    messages.append(HumanMessage(content=prompt_message))
    
    # Hàm hỗ trợ phân tích đệ quy kết quả trả về để xác định kết nối thành công
    def is_connection_successful(result) -> bool:
        if not result:
            return False
        
        # Nếu là danh sách (đầu ra của LangChain/MCP thường là list của các đối tượng nội dung)
        if isinstance(result, list):
            return any(is_connection_successful(item) for item in result)
        
        # Nếu là đối tượng có thuộc tính 'text' hoặc 'content'
        if hasattr(result, 'text'):
            return is_connection_successful(getattr(result, 'text'))
        if hasattr(result, 'content'):
            return is_connection_successful(getattr(result, 'content'))
            
        # Nếu là dictionary
        if isinstance(result, dict):
            # Xử lý trường hợp dạng {'type': 'text', 'text': '...'} như kết quả của bạn
            if 'text' in result:
                return is_connection_successful(result['text'])
            
            # Kiểm tra trực tiếp các cờ trạng thái thành công
            if result.get("success") is True or result.get("connected") is True:
                return True
            
            # Đề phòng trường hợp giá trị của success là chuỗi "true" thay vì boolean
            if str(result.get("success")).lower() == "true" or str(result.get("connected")).lower() == "true":
                return True
                
            return False
            
        # Nếu là chuỗi JSON hoặc chuỗi thường
        if isinstance(result, str):
            trimmed = result.strip()
            if trimmed.startswith('{') and trimmed.endswith('}'):
                try:
                    parsed = json.loads(trimmed)
                    return is_connection_successful(parsed)
                except Exception:
                    pass
            # Kiểm tra fallback bằng từ khóa trong chuỗi
            lower_str = trimmed.lower()
            return ("success" in lower_str and "true" in lower_str) or "connected to" in lower_str
            
        return False

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Nạp danh sách tool ban đầu (13 tools)
            mcp_tools = await load_mcp_tools(session)
            tools_map = {tool.name: tool for tool in mcp_tools}
            model_with_tools = model.bind_tools(mcp_tools)
            
            # Giới hạn tối đa 10 lượt suy luận/hành động tương tác cho một phiên kiểm thử
            for i in range(20):
                response = await model_with_tools.ainvoke(messages)
                messages.append(response)
                
                if not response.tool_calls:
                    break
                
                connection_established = False
                    
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_id = tool_call["id"]
                    
                    if tool_name in tools_map:
                        try:
                            # Thực thi công cụ
                            tool_result = await tools_map[tool_name].ainvoke(tool_args)
                            print(f"🛠️ Công cụ '{tool_name}' đã được thực thi với kết quả:\n {tool_result}")
                            
                            # Kiểm tra xem công cụ kết nối có chạy thành công không
                            if tool_name in CONNECTION_TOOLS and is_connection_successful(tool_result):
                                connection_established = True

                            # Áp dụng bộ lọc dọn dẹp ảnh chụp thô Base64
                            clean_result = sanitize_tool_result_content(tool_name, tool_result, workspace_path)

                            messages.append(ToolMessage(
                                content=str(clean_result),
                                name=tool_name,
                                tool_call_id=tool_id
                            ))
                        except Exception as e:
                            messages.append(ToolMessage(
                                content=f"Lỗi thực thi công cụ {tool_name}: {str(e)}",
                                name=tool_name,
                                tool_call_id=tool_id
                            ))
                    else:
                        messages.append(ToolMessage(
                            content=f"Không tìm thấy công cụ '{tool_name}' trên hệ thống.",
                            name=tool_name,
                            tool_call_id=tool_id
                        ))
                
                # Nếu phát hiện kết nối thành công, tiến hành nạp lại toàn bộ công cụ mới (Dynamic Re-binding)
                if connection_established:
                    print("🔄 Đã phát hiện kết nối thành công. Tiến hành nạp lại danh sách công cụ từ MCP...")
                    await asyncio.sleep(1.0)  # Chờ 1 giây để server đồng bộ trạng thái kết nối và phản hồi cổng
                    
                    mcp_tools = await load_mcp_tools(session)
                    tools_map = {tool.name: tool for tool in mcp_tools}
                    model_with_tools = model.bind_tools(mcp_tools)
                    
                    print(f"✅ Đã cập nhật thành công! Tổng số công cụ khả dụng hiện tại: {len(mcp_tools)}")
            
            return messages[-1].content
async def run_agent_with_devtools_mcp(model, prompt_message: str, chat_history: List[BaseMessage] = None):
    profile_dir = get_default_browser_profile_dir()
    
    # Đảm bảo khởi tạo thư mục profile nếu chưa tồn tại vật lý
    os.makedirs(profile_dir, exist_ok=True)

    server_params = StdioServerParameters(
        command="npx",
        args=[
            "-y", 
            "chrome-devtools-mcp@latest", 
            "--autoConnect", 
            "--no-usage-statistics",
            f"--user-data-dir={profile_dir}" # Sử dụng đường dẫn động an toàn [2]
        ]
    )
    
    if chat_history is None:
        chat_history = []
        
    system_prompt = (
        "Bạn là một chuyên gia gỡ lỗi Chrome Extension chuyên nghiệp.\n"
        "Bạn có quyền truy cập trực tiếp vào Chrome DevTools thông qua các công cụ được cung cấp.\n"
        "Hãy sử dụng chúng để phân tích mã lỗi, kiểm tra các yêu cầu mạng (network requests) "
        "và đọc logs console nhằm xác định chính xác nguyên nhân gây lỗi của Extension.\n"
        "Hãy thực hiện các hành động tuần tự (Navigate -> Interact -> Get Logs) trên cùng một trình duyệt."
    )
    
    messages = [SystemMessage(content=system_prompt)] + chat_history
    messages.append(HumanMessage(content=prompt_message))
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools = await load_mcp_tools(session)
            tools_map = {tool.name: tool for tool in mcp_tools}
            model_with_tools = model.bind_tools(mcp_tools)
            
            for _ in range(8):
                response = await model_with_tools.ainvoke(messages)
                messages.append(response)
                
                if not response.tool_calls:
                    break
                    
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_id = tool_call["id"]
                    
                    if tool_name in tools_map:
                        try:
                            tool_result = await tools_map[tool_name].ainvoke(tool_args)
                            
                            # Áp dụng bộ lọc dọn dẹp ảnh chụp thô Base64
                            clean_result = sanitize_tool_result_content(tool_name, tool_result, ".")

                            messages.append(ToolMessage(
                                content=str(clean_result),
                                name=tool_name,
                                tool_call_id=tool_id
                            ))
                        except Exception as e:
                            messages.append(ToolMessage(
                                content=f"Lỗi khi thực thi công cụ {tool_name}: {str(e)}",
                                name=tool_name,
                                tool_call_id=tool_id
                            ))
                    else:
                        messages.append(ToolMessage(
                            content=f"Không tìm thấy công cụ '{tool_name}' trên hệ thống MCP.",
                            name=tool_name,
                            tool_call_id=tool_id
                        ))
            
            return messages[-1].content