from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import List, Optional, Union, overload

from pydantic import BaseModel, Field

from app.logger import logger
from app.llm import LLM
from app.schema import ROLE_TYPE, AgentState, Memory, Message
from app.mcp import MCPClient, MCPServer


class BaseAgent(BaseModel, ABC):
    """Abstract base class for managing agent state and action.

    Provides foundational functionality for state transitions, memory management,
    and a step-based execution loop. Subclasses must implement the `step` method.
    """

    # Core attributes
    llm_name: str = Field(..., description="which llm this agent uses")
    description: Optional[str] = Field(None, description="Optional agent description")

    # Dependencies
    llm: Optional[LLM] = Field(None, description="Language model instance")
    memory: Memory = Field(default_factory=Memory, description="Agent's memory store")
    state: AgentState = Field(
        default=AgentState.IDLE, description="Current agent state"
    )
    client: Optional[MCPClient] = Field(None, description="MCP client instance")
    server: Optional[Union[str, MCPServer]] = Field(None, description="Path or instance of MCP server")

    # Prompts
    system_prompt: Optional[str] = Field(
        None, description="System-level instruction prompt"
    )
    next_step_prompt: Optional[str] = Field(
        None, description="Prompt for determining next action"
    )
    
    # Execution control
    max_steps: int = Field(default=10, description="Maximum steps before termination")
    current_step: int = Field(default=0, description="Current step in execution")

    duplicate_threshold: int = 2

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"
    
    @asynccontextmanager
    async def state_context(self, new_state: AgentState):
        """Context manager for safe agent state transitions.

        Args:
            new_state: The state to transition to during the context.

        Yields:
            None: Allows execution within the new state.

        Raises:
            ValueError: If the new_state is invalid.
        """
        if not isinstance(new_state, AgentState):
            raise ValueError(f"Invalid state: {new_state}")

        previous_state = self.state
        self.state = new_state
        try:
            yield
        except Exception as e:
            logger.info(f"Error during execution: {e}")
            self.state = AgentState.ERROR  # Transition to ERROR on failure
            raise e
        finally:
            self.state = previous_state  # Revert to previous state

    def update_memory(
        self,
        role: ROLE_TYPE,  # type: ignore
        content: str,
        base64_image: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Add a message to the agent's memory.

        Args:
            role: The role of the message sender (user, system, assistant, tool).
            content: The message content.
            base64_image: Optional base64 encoded image.
            **kwargs: Additional arguments (e.g., tool_call_id for tool messages).

        Raises:
            ValueError: If the role is unsupported.
        """
        message_map = {
            "user": Message.user_message,
            "system": Message.system_message,
            "assistant": Message.assistant_message,
            "tool": lambda content, **kw: Message.tool_message(content, **kw),
        }

        if role not in message_map:
            raise ValueError(f"Unsupported message role: {role}")

        # Create message with appropriate parameters based on role
        kwargs = {"base64_image": base64_image, **(kwargs if role == "tool" else {})}
        self.memory.add_message(message_map[role](content, **kwargs))

    @overload
    async def run(self, request: str) -> AsyncGenerator[dict, None]: ...

    @overload
    async def run(self, request: str, stream: bool) -> AsyncGenerator[dict, None]: ...

    async def run(self, request: str, stream: bool = False) -> AsyncGenerator[dict, None]:
        """Execute the agent's main loop asynchronously.

        Args:
            request: Optional initial user request to process(RESTful API).
            step_results: Optional flag to return results of each step(WebSocket API). Defaults to False.

        Returns:
            A string summarizing the execution results.

        Raises:
            RuntimeError: If the agent is not in IDLE state at start.
        """
        if self.state != AgentState.IDLE:
            raise RuntimeError(f"Cannot run agent from state: {self.state}")

        if request:
            self.update_memory("user", request)

        if request is None:
            logger.warning("No request provided, using default prompt")
            
        async with self.state_context(AgentState.RUNNING):
            if stream:
                async for event in self.stream(stream):
                    yield event

            elif not stream:
                while self.current_step < self.max_steps and self.state != AgentState.FINISHED:
                    self.current_step += 1
                    step_result = await self.step()

                    if self.is_stuck():
                        self.handle_stuck_state()
                    yield {"type": "step", "data": step_result}
                if self.current_step >= self.max_steps:
                    self.current_step = 0
                    self.state = AgentState.IDLE
    
    @abstractmethod
    async def step(self) -> dict:
        """Execute a single step in the agent's workflow.

        Must be implemented by subclasses to define specific behavior.
        """

    @abstractmethod
    async def stream(self, stream: bool = True, **kwargs) -> AsyncGenerator[dict, None]:
        """Designed for streaming responses.

        Must be implemented by subclasses to define logic of handling stream responses.
        """

    def handle_stuck_state(self):
        """Handle stuck state by adding a prompt to change strategy"""
        stuck_prompt = "\
        Observed duplicate responses. Consider new strategies and avoid repeating ineffective paths already attempted."
        self.next_step_prompt = f"{stuck_prompt}\n{self.next_step_prompt}"
        logger.warning(f"Agent detected stuck state. Added prompt: {stuck_prompt}")

    def is_stuck(self) -> bool:
        """Check if the agent is stuck in a loop by detecting duplicate content"""
        if len(self.memory.messages) < 2:
            return False

        last_message = self.memory.messages[-1]
        if not last_message.content:
            return False

        # Count identical content occurrences
        duplicate_count = sum(
            1
            for msg in reversed(self.memory.messages[:-1])
            if msg.role == "assistant" and msg.content == last_message.content
        )

        return duplicate_count >= self.duplicate_threshold

    async def get_final_step(self) -> dict:
        if hasattr(self, "step_results"):
            return self.step_results[-1]
        return {"think": "任务完成", "act": "无操作"}

    @property
    def messages(self) -> List[Message]:
        """Retrieve a list of messages from the agent's memory."""
        return self.memory.messages

    @messages.setter
    def messages(self, value: List[Message]):
        """Set the list of messages in the agent's memory."""
        self.memory.messages = value

