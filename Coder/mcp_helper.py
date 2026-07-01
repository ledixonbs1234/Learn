# mcp_helper.py
import asyncio
from typing import List
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage, AIMessage

async def run_agent_with_devtools_mcp(model, prompt_message: str, chat_history: List[BaseMessage] = None):
    """
    Khởi chạy phiên làm việc đồng thời của LLM và Chrome DevTools MCP.
    Đảm bảo kết nối Stdio được giữ hoạt động trong suốt quá trình xử lý tác vụ.
    """
    # 1. Định cấu hình khởi chạy MCP server qua npx (Node.js)
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "chrome-devtools-mcp@latest", "--autoConnect", "--no-usage-statistics"]
    )
    
    if chat_history is None:
        chat_history = []
        
    system_prompt = (
        "Bạn là một chuyên gia gỡ lỗi Chrome Extension chuyên nghiệp.\n"
        "Bạn có quyền truy cập trực tiếp vào Chrome DevTools thông qua các công cụ được cung cấp.\n"
        "Hãy sử dụng chúng để phân tích mã lỗi, kiểm tra các yêu cầu mạng (network requests) "
        "và đọc logs console nhằm xác định chính xác nguyên nhân gây lỗi của Extension."
    )
    
    messages = [SystemMessage(content=system_prompt)] + chat_history
    
    # 2. Tạo kết nối Stdio Client
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Khởi tạo phiên kết nối MCP
            await session.initialize()
            
            # Tải động tất cả các công cụ của Chrome DevTools MCP sang dạng LangChain Tools
            mcp_tools = await load_mcp_tools(session)
            tools_map = {tool.name: tool for tool in mcp_tools}
            
            # Liên kết các công cụ này vào LLM của bạn (Hỗ trợ function calling của bạn)
            model_with_tools = model.bind_tools(mcp_tools)
            
            # Lần gọi đầu tiên gửi yêu cầu của người dùng
            messages.append(AIMessage(content=prompt_message))
            
            # Vòng lặp phản hồi của Agent (tối đa 8 bước lặp công cụ để tránh lặp vô hạn)
            for _ in range(8):
                response = await model_with_tools.ainvoke(messages)
                messages.append(response)
                
                # Nếu mô hình không yêu cầu gọi công cụ nào nữa, kết thúc vòng lặp
                if not response.tool_calls:
                    break
                    
                # Thực thi tuần tự các công cụ do mô hình yêu cầu
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_id = tool_call["id"]
                    
                    if tool_name in tools_map:
                        try:
                            # Thực thi công cụ của DevTools ngay trong phiên đang kết nối
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