import asyncio
from typing import Optional, Union
from contextlib import AsyncExitStack, suppress

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app.tool import ToolCollection
from app.logger import logger
from app.mcp import MCPServer
from app.mcp.transport import MessageStream, custom_client
from app.tool.base import BaseTool, ToolResult 
from app.schema import CLIENT_CHOICE_TYPE


class MCPClientTool(BaseTool):
    """Wrapper for MCP tools to execute them within a session."""

    session: Optional[ClientSession] = None
    # server_id: str = ""  # Add server identifier
    original_name: str = ""

    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool by making a remote call to the MCP server."""
        if not self.session:
            return ToolResult(error="Not connected to MCP server")

        # TODO: Image types supports required
        try:
            logger.info(f"Executing tool: {self.original_name}")
            result = await self.session.call_tool(self.original_name, kwargs)

            if getattr(result, "isError", False):
                error_text = "\n".join(block.text for block in result.content if hasattr(block, "text"))
                return ToolResult(error=error_text or "Unknown error")
            
            output_text = "\n".join(item.text for item in result.content if hasattr(item, "text"))
            logger.info(f"Tool {self.original_name} execution done.")
            return ToolResult(output=output_text or "No output returned.")
        
        except Exception as e:
            logger.exception(f"Exception when calling tool {self.original_name}")
            return ToolResult(error=f"Error executing tool: {type(e).__name__}: {str(e)}")


class MCPClient(ToolCollection):
    """Create MCP client within a tool collection and connect to specified server using customed transport method."""
    
    def __init__(self, client_type: CLIENT_CHOICE_TYPE = "tool_call"): # type: ignore
        super().__init__()
        # Initialize session and objects
        self.client_type: CLIENT_CHOICE_TYPE = client_type # type: ignore
        self.session: Optional[ClientSession] = None
        self.exit_stack: Optional[AsyncExitStack] = None
        self._server: Optional[MCPServer] = None
        self._server_task: Optional[asyncio.Task] = None
        self._custom_client_ctx: Optional[MessageStream] = None

    async def connect_to_server(self, server_source: Optional[Union[str, MCPServer]] = None) -> None:
        """Supports incoming script file or an MCP server instance, default MCP server"""
        # Always ensure clean disconnection before new connection
        if self.session:
            await self.disconnect()

        if server_source is None:
            if self._server is None:
                self._server = MCPServer()
        else:
            self._server = server_source

        if self.exit_stack is None:
            self.exit_stack = AsyncExitStack()
        try:
            if isinstance(self._server, str):
                logger.info(f"Connecting to MCP server script: {self._server}")
                is_python = self._server.endswith('.py')
                is_js = self._server.endswith('.js')

                # connect to server script using standard stdio transport
                if (is_python or is_js):
                    command = "python" if is_python else "node"
                    server_params = StdioServerParameters(
                        command=command,
                        args=[self._server],
                        env=None
                    )

                    stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
                    stdio, write = stdio_transport
                    self.session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))

                    await self._initialize_and_list_tools()
            elif isinstance(self._server, MCPServer):
                # logger.info(f"Connecting to MCP server instance: {self._server}")
                await self._connect_to_instance(self._server)

            else:
                raise Exception("Invalid server source")
            
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            raise

    async def _connect_to_instance(self, server: MCPServer) -> None:
        """Connet to MCP server instance in same pipe"""
        from .transport import MemoryTransport

        transport = MemoryTransport()

        # start server
        ready = asyncio.Event()
        self._server_task = asyncio.create_task(
            server.run_custom_async(transport, ready_event=ready)
        )
        logger.info(f"Awaiting server ready")
        await ready.wait()
        
        logger.info(f"Server is ready, connecting client")
        self._custom_client_ctx = custom_client(transport)
        client_reader, client_writer = await self._custom_client_ctx.__aenter__()
        self.session = await self.exit_stack.enter_async_context(ClientSession(client_reader, client_writer))
        await self._initialize_and_list_tools()

    async def _initialize_and_list_tools(self) -> None:
        """Initialize session and populate tool map."""
        if not self.session:
            raise RuntimeError(f"Session not initialized for server in client {self.client_type}")

        logger.info(f"Initializing session")
        await self.session.initialize()
        try:
            logger.info(f"Listing tools in client {self.client_type}")
            response = await self.session.list_tools()
            logger.info(f"Received {len(response.tools)} tools from server")
            # clear the tool map
            self.tool_map.clear()
            
            # Populate tool map with MCPClientTool instances
            for tool in response.tools:
                original_name = tool.name
                tool_name = f"mcp_{original_name}"
                mcp_tool = MCPClientTool(
                    name=tool_name,
                    description=tool.description,
                    parameters=tool.inputSchema,
                    session=self.session,
                    original_name=original_name,
                )
                self.tool_map[tool.name] = mcp_tool
            
            # update tools tuple
            self.tools = tuple(self.tool_map.values())
            
            tool_names = [tool.name for tool in response.tools]
            logger.info(f"Registered {len(tool_names)} tools: {tool_names}")
            
        except Exception as e:
            logger.error(f"Failed to initialize tools: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect server and do cleanup."""
        self.tool_map.clear()
        self.tools = tuple()
        
        if self.exit_stack:
            try:
                await self.exit_stack.aclose()
                logger.info(f"Successfully closed exit stack")
            except Exception as e:
                logger.error(f"Error closing exit stack: {e}")
                # raise
            finally:
                self.exit_stack = None
        
        self.session = None
        
        if self._custom_client_ctx:
            await self._custom_client_ctx.__aexit__(None, None, None)
            self._custom_client_ctx = None
            
        if self._server_task:
            self._server_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._server_task
            self._server_task = None
        
        if self._server:
            self._server = None
