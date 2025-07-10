import asyncio
import time
import uuid
from app.llm import LLM
from pydantic import Field, model_validator
from typing import Optional, List, Union

from app.llm import LLM
from app.agent.toolcall import ToolCallAgent, StreamToolCallAgent
from app.prompt.styles import *
from app.mcp import MCPClient
from app.prompt.toolcall import NEXT_STEP_PROMPT
from app.tool import CreateChatCompletion, Terminate, ToolCollection, AskHuman, BrowserUseTool
from app.logger import logger


class ChaosStar(ToolCallAgent):
    """DeepResearch agent for traditional HTTP requests."""

    name: str = Field(..., description="the name of agent")

    system_prompt: Optional[str] = Field(
        EXPLANATORY_STYLE['prompt'], 
        description="System-level instruction prompt"
    )
    next_step_prompt: str = NEXT_STEP_PROMPT

    _initialized: bool = False

    @model_validator(mode="after")
    def initialize_agent(self) -> "ChaosStar":
        """Initialize agent with default settings if not provided."""
        if self.llm is None:
            self.llm = LLM.for_config(config_name=self.llm_name.lower())
        return self
    
    async def initialize_client_and_server(self):
        """Initialize mcp client synchronously."""
        if not self.client:
            self.client = MCPClient()
        await self.client.connect_to_server(self.server)
        await self.add_tools()
    
    async def add_tools(self):
        """get available tools from server."""
        if self.available_tools.tool_map:
            self.available_tools.cleanup()
            logger.info("Cleared existing tools from available tools collection.")
        new_tools = [tool for tool in self.client.tools]
        self.available_tools.add_tools(*new_tools)
        
    @classmethod
    async def create(cls, **kwargs):
        instance = cls(**kwargs)
        await instance.initialize_client_and_server()
        instance._initialized = True
        return instance
    
    async def cleanup(self):
        if self._initialized:
            await self.client.disconnect()
            logger.info("ChaosStar client disconnected.")
            self._initialized = False

    async def think(self) -> any:
        if not self._initialized:
            await self.initialize_client_and_server()
            self._initialized = True

        result = await super().think()

        return result
    

class StreamChaosStar(StreamToolCallAgent):
    """DeepResearch agent for stream WebSocket requests."""
    name: str = Field(..., description="the name of agent")

    system_prompt: Optional[str] = Field(
        EXPLANATORY_STYLE['prompt'], 
        description="System-level instruction prompt"
    )
    next_step_prompt: str = NEXT_STEP_PROMPT

    _initialized: bool = False

    @model_validator(mode="after")
    def initialize_agent(self) -> "ChaosStar":
        """Initialize agent with default settings if not provided."""
        if self.llm is None:
            self.llm = LLM.for_config(config_name=self.llm_name.lower())
        return self
    
    async def initialize_client_and_server(self):
        """Initialize mcp client synchronously."""
        if not self.client:
            self.client = MCPClient()
        await self.client.connect_to_server(self.server)
        await self.add_tools()
    
    async def add_tools(self):
        """get available tools from server."""
        if self.available_tools.tool_map:
            self.available_tools.cleanup()
            logger.info("Cleared existing tools from available tools collection.")
        new_tools = [tool for tool in self.client.tools]
        self.available_tools.add_tools(*new_tools)
        
    @classmethod
    async def create(cls, **kwargs):
        instance = cls(**kwargs)
        await instance.initialize_client_and_server()
        instance._initialized = True
        return instance
    
    async def cleanup(self):
        if self._initialized:
            await self.client.disconnect()
            logger.info("StreamChaosStar client disconnected.")
            self._initialized = False

    async def think(self, stream: bool = True, **kwargs) -> any:
        if not self._initialized:
            await self.initialize_client_and_server()
            self._initialized = True

        response = await super().think(stream, **kwargs)

        return response