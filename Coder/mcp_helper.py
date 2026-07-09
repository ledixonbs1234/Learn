# mcp_helper.py
import os
import sys
import platform
import asyncio
from typing import List
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage, AIMessage

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
    Khởi chạy flutter-skill dưới dạng MCP Server và nạp các công cụ E2E 
    vào ngữ cảnh của AI Agent để tương tác trực tiếp với ứng dụng.
    """
    # Khởi chạy flutter-skill thông qua CLI đã được cài đặt trên máy Host
    server_params = StdioServerParameters(
        command="flutter-skill",
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
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools = await load_mcp_tools(session)
            tools_map = {tool.name: tool for tool in mcp_tools}
            model_with_tools = model.bind_tools(mcp_tools)
            
            # Giới hạn tối đa 10 lượt suy luận/hành động tương tác cho một phiên kiểm thử
            for _ in range(10):
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
                            messages.append(ToolMessage(
                                content=str(tool_result),
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
                            messages.append(ToolMessage(
                                content=str(tool_result),
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