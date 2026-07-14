# oder/mcp_helper.py
import json
import os
import sys
import platform
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from config import sanitize_tool_result_content

# =====================================================================
# HÀM HỖ TRỢ CHẠY ĐỒNG BỘ AN TOÀN TRONG GRAPH NODES
# =====================================================================
def run_sync(coro):
    """
    Thực thi coroutine bất đồng bộ một cách đồng bộ và an toàn.
    Sử dụng nest_asyncio nếu event loop hiện tại đang chạy để tránh lỗi xung đột luồng.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():
        import nest_asyncio
        nest_asyncio.apply()
        
    return loop.run_until_complete(coro)


# =====================================================================
# BỘ ĐIỀU PHỐI MULTI-SERVER MCP TRUNG TÂM
# =====================================================================
class MCPRegistryManager:
    """
    Bộ quản lý đăng ký và kết nối MCP Server tập trung có cơ chế quản lý vòng đời tài nguyên.
    Sử dụng mẫu thiết kế Persistent Client để tránh rò rỉ tiến trình con của hệ thống.
    """
    # Khai báo các thuộc tính tĩnh (Static variables) để duy trì kết nối duy nhất
    _active_client: Optional[MultiServerMCPClient] = None
    _loaded_servers_hash: Optional[str] = None

    def __init__(self, workspace_path: str):
        self.workspace_path = Path(workspace_path).expanduser().resolve()
        self.local_config_path = self.workspace_path / "mcp_config.json"
        
        # Cấu hình Global: Nằm cùng cấp với tệp mã nguồn chính của LangGraph (thư mục Coder/)
        self.global_config_path = Path(__file__).parent.resolve() / "mcp_config.json"
        
        # Cấu hình Parent: Đề phòng trường hợp mã nguồn được đóng gói trong thư mục con của Coder/
        self.parent_config_path = Path(__file__).parent.parent.resolve() / "mcp_config.json"
        
        self._tools_cache: Optional[List[Any]] = None
        
    def load_config(self) -> Dict[str, Any]:
        """Tải cấu hình mcp_config.json theo thứ tự phân cấp ưu tiên."""
        config_path = None
        if self.local_config_path.exists():
            config_path = self.local_config_path
        elif self.global_config_path.exists():
            config_path = self.global_config_path
        elif self.parent_config_path.exists():
            config_path = self.parent_config_path
            
        config_data = {}
        if config_path:
            try:
                config_data = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"[Dynamic MCP Warning] Lỗi đọc {config_path.name}: {str(e)}")
                
        servers = config_data.get("mcpServers", config_data.get("servers", config_data))
        if not isinstance(servers, dict):
            servers = {}
            return servers

    def build_client_config(self, raw_servers: Dict[str, Any]) -> Dict[str, Any]:
        """Chuẩn hóa cấu hình thành chuẩn của MultiServerMCPClient."""
        client_config = {}
        for name, cfg in raw_servers.items():
            if not isinstance(cfg, dict):
                continue
            url = cfg.get("url") or cfg.get("serverUrl")
            command = cfg.get("command")
            args = cfg.get("args", [])
            env = cfg.get("env")
            headers = cfg.get("headers")
            transport = cfg.get("transport", cfg.get("type"))
            if not transport:
                if url:
                    transport = "sse"
                elif command:
                    transport = "stdio"
                    
            if transport in ["sse", "http", "streamable_http"] and url:
                client_config[name] = {
                    "transport": "sse" if transport == "sse" else "streamable_http",
                    "url": url
                }
                if headers:
                    client_config[name]["headers"] = headers
            elif transport == "stdio" and command:
                client_config[name] = {
                    "transport": "stdio",
                    "command": command,
                    "args": args
                }
                if env is not None:
                    client_config[name]["env"] = env
        return client_config

    async def get_tools_async(self, active_servers: Optional[List[str]] = None) -> List[Any]:
        """Tải động các công cụ từ các MCP Server, tái sử dụng kết nối cũ để bảo vệ tài nguyên."""
        raw_servers = self.load_config()
        if not raw_servers:
            return []
            
        # Nạp máy chủ được chỉ định bởi AI Supervisor ở bước Triage
        filtered_servers = {}
        for name, cfg in raw_servers.items():
            if active_servers is not None:
                if name not in active_servers:
                    continue
            filtered_servers[name] = cfg
            
        client_config = self.build_client_config(filtered_servers)
        if not client_config:
            return []
            
        # Tạo mã băm (hash) cấu hình hiện tại để kiểm tra xem danh sách server có thay đổi không
        current_hash = json.dumps(sorted(list(filtered_servers.keys())))
        
        try:
            # Nếu đã có Client hoạt động và cấu hình không đổi, tái sử dụng kết nối cũ
            if MCPRegistryManager._active_client and MCPRegistryManager._loaded_servers_hash == current_hash:
                return await MCPRegistryManager._active_client.get_tools()
                
            # Ngược lại, tiến hành khởi tạo mới một lần duy nhất
            client = MultiServerMCPClient(client_config)
            tools = await client.get_tools()
            
            # Cập nhật tham chiếu tĩnh toàn cục
            MCPRegistryManager._active_client = client
            MCPRegistryManager._loaded_servers_hash = current_hash
            return tools
        except Exception as e:
            print(f"[Dynamic MCP Error] Lỗi khởi tạo hoặc tái sử dụng MultiServerMCPClient: {str(e)}")
            return []

    def get_tools_sync(self, active_servers: Optional[List[str]] = None) -> List[Any]:
        """Phương thức đồng bộ hóa để tích hợp trực tiếp vào LangGraph Sync Nodes."""
        if active_servers is None and self._tools_cache is not None:
            return self._tools_cache
            
        try:
            tools = run_sync(self.get_tools_async(active_servers))
            if active_servers is None:
                self._tools_cache = tools
            return tools
        except Exception as e:
            print(f"[Dynamic MCP Error] Lỗi tải công cụ đồng bộ: {str(e)}")
            return []


# =====================================================================
# CÁC HÀM CŨ ĐỂ ĐẢM BẢO TƯƠNG THÍCH NGƯỢC (BACKWARD COMPATIBILITY)
# =====================================================================
def get_default_browser_profile_dir() -> str:
    """Tự động phát hiện đường dẫn User Data an toàn dựa trên Hệ điều hành."""
    system = platform.system()
    home = os.path.expanduser("~")
    
    if system == "Windows":
        edge_path = os.path.join(home, "AppData", "Local", "Microsoft", "Edge", "User Data")
        if os.path.exists(edge_path):
            return edge_path
        return os.path.join(home, "AppData", "Local", "Google", "Chrome", "User Data")
    elif system == "Darwin":  # macOS
        return os.path.join(home, "Library", "Application Support", "Google", "Chrome")
    else:  # Linux
        return os.path.join(home, ".config", "google-chrome")

async def run_agent_with_flutter_skill_mcp(model, prompt_message: str, chat_history: List[BaseMessage] = None, workspace_path: str = "."):
    """Khởi chạy flutter_skill dưới dạng MCP Server và nạp các công cụ E2E."""
    CONNECTION_TOOLS = {
        "connect_app", "launch_app", "scan_and_connect", "connect_cdp", 
        "connect_openclaw_browser", "connect_webmcp"
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
    
    def is_connection_successful(result) -> bool:
        if not result:
            return False
        if isinstance(result, list):
            return any(is_connection_successful(item) for item in result)
        if hasattr(result, 'text'):
            return is_connection_successful(getattr(result, 'text'))
        if hasattr(result, 'content'):
            return is_connection_successful(getattr(result, 'content'))
        if isinstance(result, dict):
            if 'text' in result:
                return is_connection_successful(result['text'])
            if result.get("success") is True or result.get("connected") is True:
                return True
            if str(result.get("success")).lower() == "true" or str(result.get("connected")).lower() == "true":
                return True
            return False
        if isinstance(result, str):
            trimmed = result.strip()
            if trimmed.startswith('{') and trimmed.endswith('}'):
                try:
                    parsed = json.loads(trimmed)
                    return is_connection_successful(parsed)
                except Exception:
                    pass
            lower_str = trimmed.lower()
            return ("success" in lower_str and "true" in lower_str) or "connected to" in lower_str
        return False

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            mcp_tools = await load_mcp_tools(session)
            tools_map = {tool.name: tool for tool in mcp_tools}
            model_with_tools = model.bind_tools(mcp_tools)
            
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
                            tool_result = await tools_map[tool_name].ainvoke(tool_args)
                            print(f"🛠️ Công cụ '{tool_name}' đã được thực thi với kết quả:\n {tool_result}")
                            if tool_name in CONNECTION_TOOLS and is_connection_successful(tool_result):
                                connection_established = True

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
                
                if connection_established:
                    print("🔄 Đã phát hiện kết nối thành công. Tiến hành nạp lại danh sách công cụ từ MCP...")
                    await asyncio.sleep(1.0)
                    mcp_tools = await load_mcp_tools(session)
                    tools_map = {tool.name: tool for tool in mcp_tools}
                    model_with_tools = model.bind_tools(mcp_tools)
            
            return messages[-1].content

async def run_agent_with_devtools_mcp(model, prompt_message: str, chat_history: List[BaseMessage] = None):
    profile_dir = get_default_browser_profile_dir()
    os.makedirs(profile_dir, exist_ok=True)

    server_params = StdioServerParameters(
        command="npx",
        args=[
            "-y", 
            "chrome-devtools-mcp@latest", 
            "--autoConnect", 
            "--no-usage-statistics",
            f"--user-data-dir={profile_dir}"
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