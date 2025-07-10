from abc import ABC, abstractmethod
from typing import AsyncGenerator

from app.agent.base import BaseAgent


class StreamAgent(BaseAgent, ABC):
    """General abstract class for the interleaved-thinking mode of an agent.
    
    Where think process represents the response to input and 
    action function represents the result of executing a tool call.
    """

    @abstractmethod
    async def think(self, stream: bool = True, **kwargs):
        ...

    @abstractmethod
    async def act(self, tool_uses: any):
        ...

    async def stream(self, stream: bool = True, **kwargs) -> AsyncGenerator[dict, None]:
        """Simplified process."""
        response = await self.think(stream, **kwargs)
        async with response as stream:
            async for chunk in stream:
                pass
            async for tool_event in self.act():
                yield tool_event
            async for next_event in self.stream(stream, **kwargs):
                yield next_event

    async def step(self):
        pass