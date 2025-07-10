from abc import ABC, abstractmethod

from pydantic import Field

from app.agent.base import BaseAgent
from app.schema import AgentState


class ReActAgent(BaseAgent, ABC):
    """General abstract class for the reaction of an agent.
    
    Where think process represents the action to input and decide whether to execute a tool call or not,
    and act process is the excution of tools.
    """
    
    reasoning_type: str = Field(default="react", description="Type of reasoning used by the agent")

    @abstractmethod
    async def think(self) -> bool:
        """Process current state and decide next action"""

    @abstractmethod
    async def act(self) -> str:
        """Execute decided actions"""

    async def step(self) -> dict:
        """Execute a single step: think and act."""
        response = {"think": "", "act": ""}
        think_response, should_act = await self.think()
        response["think"] = think_response
        if should_act:
            act_response = await self.act()
            response["act"] = act_response
        else:
            self.state = AgentState.FINISHED
        return response
    
    async def stream(self):
        pass