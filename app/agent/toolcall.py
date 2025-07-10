import asyncio
import json
from typing import Any, AsyncGenerator, ClassVar, List, Optional, Union

from pydantic import Field

from app.agent.react import ReActAgent
from app.agent.stream import StreamAgent
from app.exceptions import TokenLimitExceeded
from app.logger import logger
from app.prompt.toolcall import NEXT_STEP_PROMPT, SYSTEM_PROMPT, Tool_use_system_prompt
from app.schema import TOOL_CHOICE_TYPE, AgentState, Message, MockMessage, ToolCall, ToolChoice, Claude
from app.tool import CreateChatCompletion, Terminate, ToolCollection
from app.llm import Parameters

TOOL_CALL_REQUIRED = "Tool calls required but none provided"


class ToolCallAgent(ReActAgent):
    """Base agent class for handling tool/function calls with enhanced abstraction"""

    name: str = Field(default="toolcall", description="Execution mode for the agent")

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    available_tools: ToolCollection = ToolCollection(
        # CreateChatCompletion(), Terminate()
    )
    tool_choices: TOOL_CHOICE_TYPE = ToolChoice.AUTO.value  # type: ignore
    special_tool_names: List[str] = Field(default_factory=lambda: [Terminate().name])

    tool_calls: List[ToolCall] = Field(
        default_factory=list,
        description="List of tool calls to execute"
    )
    _current_base64_image: Optional[str] = None

    max_steps: int = 30
    max_observe: int = 100

    async def think(self) -> tuple[str, bool]:
        """Process current state and decide whether next actions using tools or not"""
        if self.next_step_prompt:
            assistant_msg = Message.assistant_message(self.next_step_prompt)
            self.messages += [assistant_msg]

        logger.info(f"Start to get response with messages: {self.messages}")

        # record thinking response content
        content = ""
        try:
            # Get response with tool options
            response = await self.llm.ask_tool(
                messages=self.messages,
                system_msgs=(
                    [Message.system_message(self.system_prompt)]
                    if self.system_prompt
                    else None
                ),
                tools=self.available_tools.to_params(),
                tool_choice=self.tool_choices,
            )
        except ValueError:
            raise
        except Exception as e:
            # Check if this is a RetryError containing TokenLimitExceeded
            if hasattr(e, "__cause__") and isinstance(e.__cause__, TokenLimitExceeded):
                token_limit_error = e.__cause__
                logger.error(
                    f"🚨 Token limit error (from RetryError): {token_limit_error}"
                )
                content = f"Maximum token limit reached, cannot continue execution: {str(token_limit_error)}"
                self.memory.add_message(
                    Message.assistant_message(
                        content=content,
                    )
                )
                # forcing the agent to finish
                self.state = AgentState.FINISHED
                return content, False
            raise

        self.tool_calls = tool_calls = (
            response.tool_calls if response and response.tool_calls else []
        )
        content = response.content if response and response.content else ""

        # Log response info
        logger.info(f"✨ {self.name}'s thoughts: {content}")
        logger.info(
            f"🛠️ {self.name} selected {len(tool_calls) if tool_calls else 0} tools to use"
        )
        if tool_calls:
            logger.info(
                f"🧰 Tools being prepared: {[call.function.name for call in tool_calls]}"
            )
            logger.info(f"🔧 Tool arguments: {[call.function.arguments for call in tool_calls]}")

        # decide next action
        try:
            if response is None:
                raise RuntimeError("No response received from the LLM")

            # Handle different tool_choices modes
            if self.tool_choices == ToolChoice.NONE.value:
                if tool_calls:
                    logger.warning(
                        f"🤔 Hmm, {self.name} tried to use tools when they weren't available!"
                    )
                if content:
                    self.memory.add_message(Message.assistant_message(content))
                return content, False

            # Create and add assistant message
            assistant_msg = (
                Message.from_tool_calls(content=content, tool_calls=self.tool_calls)
                if self.tool_calls
                else Message.assistant_message(content)
            )
            self.memory.add_message(assistant_msg)

            if self.tool_choices == ToolChoice.REQUIRED.value:
                if self.tool_calls:
                    return content, True
                else:
                    logger.warning(
                        f"🤔 {self.name} didn't use any tools, but they were required!"
                    )
                    raise ValueError(TOOL_CALL_REQUIRED)

            if self.tool_choices == ToolChoice.AUTO.value and self.tool_calls:
                return content, True

            return content, False
        except Exception as e:
            logger.error(f"🚨 Oops! The {self.name}'s thinking process hit a snag: {e}")
            self.memory.add_message(
                Message.assistant_message(
                    f"Error encountered while processing: {str(e)}"
                )
            )
            return content, False

    async def act(self) -> str:
        """Execute tool calls and handle their results"""
        results = []
        for command in self.tool_calls:
            # Reset base64_image for each tool call
            self._current_base64_image = None

            result = await self.execute_tool(command)

            logger.info(
                f"🎯 Tool '{command.function.name}' completed its mission! Result: {result}"
            )

            # Add tool response to memory
            tool_msg = Message.tool_message(
                content=result,
                tool_call_id=command.id,
                name=command.function.name,
                base64_image=self._current_base64_image,
            )
            self.memory.add_message(tool_msg)
            
            if self.max_observe:
                result = result[:self.max_observe] + "..."
            results.append(result)

        return "\n\n".join(results)

    async def execute_tool(self, command: ToolCall) -> str:
        """Execute a single tool call with robust error handling"""
        if not command or not command.function or not command.function.name:
            return "Error: Invalid command format"

        name = command.function.name
        if name not in self.available_tools.tool_map:
            return f"Error: Unknown tool '{name}'"

        try:
            logger.info(f"🔍 Preparing to execute tool arguments: '{command.function.arguments}'...")
            # Parse arguments
            args = json.loads(command.function.arguments or "{}")
            logger.info(f"✅ Arguments parsed successfully: {args}")

            # Execute the tool
            logger.info(f"🔧 Activating tool: '{name}'...")
            result = await self.available_tools.execute(name=name, tool_input=args)

            # Handle special tools
            await self._handle_special_tool(name=name, result=result)

            # Check if result is a ToolResult with base64_image
            if hasattr(result, "base64_image") and result.base64_image:
                # Store the base64_image for later use in tool_message
                self._current_base64_image = result.base64_image

            return str(result)
        except json.JSONDecodeError:
            error_msg = f"Error parsing arguments for {name}: Invalid JSON format"
            logger.error(
                f"📝 Oops! The arguments for '{name}' don't make sense - invalid JSON, arguments:{command.function.arguments}"
            )
            return f"Error: {error_msg}"
        except Exception as e:
            error_msg = f"⚠️ Tool '{name}' encountered a problem: {str(e)}"
            logger.exception(error_msg)
            return f"Error: {error_msg}"

    async def _handle_special_tool(self, name: str, result: Any, **kwargs):
        """Handle special tool execution and state changes"""
        if not self._is_special_tool(name):
            return

        if self._should_finish_execution(name=name, result=result, **kwargs):
            # Set agent state to finished
            logger.info(f"🏁 Special tool '{name}' has completed the task!")
            self.state = AgentState.FINISHED

    @staticmethod
    def _should_finish_execution(**kwargs) -> bool:
        """Determine if tool execution should finish the agent"""
        return True

    def _is_special_tool(self, name: str) -> bool:
        """Check if tool name is in special tools list"""
        return name.lower() in [n.lower() for n in self.special_tool_names]


class StreamToolCallAgent(StreamAgent):
    """An agent that uses tool calls to interact with external tools."""
    
    name: str = Field(default="toolcall", description="Execution mode for the agent")

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    available_tools: ToolCollection = ToolCollection(
        # CreateChatCompletion(), Terminate()
    )
    tool_choices: TOOL_CHOICE_TYPE = ToolChoice.AUTO.value  # type: ignore

    tool_calls: List[ToolCall] = Field(
        default_factory=list,
        description="List of tool calls to execute"
    )
    _current_base64_image: Optional[str] = None

    async def think(self, stream: bool = True, **kwargs):
        try:
            response: function = await self.llm.ask_tool(
                messages=self.messages,
                system_msgs=(
                    [Message.system_message(self.system_prompt)]
                    if self.system_prompt
                    else None
                ),
                tools=self.available_tools.to_params(),
                tool_choice=self.tool_choices,
                stream=stream,
                **kwargs,
            )
            return response
        except Exception as e:
            logger.error(f"Error encountered while processing: {str(e)}")
            raise

    async def act(self, tool_use: dict) -> None:
        name = tool_use.name; args = json.dumps(tool_use.input, ensure_ascii=False)
        result = await self.execute_tool(name, args)
        # yield {"type": "tool_result", "data": {"name": name, "result": result}}
        content = []
        content.append(
            {
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": result
             }
        )
        user_msg = Message.user_message(content)
        self.memory.add_message(user_msg)
    
    async def execute_tool(self, name: str, args: dict) -> str:
        try:
            args = json.loads(args or "{}")
            result = await self.available_tools.execute(name=name, tool_input=args)

            # Check if result is a ToolResult with base64_image
            if hasattr(result, "base64_image") and result.base64_image:
                # Store the base64_image for later use in tool_message
                self._current_base64_image = result.base64_image

            return str(result)
        except json.JSONDecodeError:
            error_msg = f"Error parsing arguments for {name}: Invalid JSON format"
            logger.error(
                f"📝 Oops! The arguments for '{name}' don't make sense - invalid JSON, arguments:{args}"
            )
            return f"Error: {error_msg}"
        except Exception as e:
            error_msg = f"⚠️ Tool '{name}' encountered a problem: {str(e)}"
            logger.exception(error_msg)
            return f"Error: {error_msg}"
    
    async def stream(self, stream: bool = True, **kwargs) -> AsyncGenerator[dict, None]:
        try:
            response = await self.think(stream, **kwargs)
            async with response as streaming:
                thinking_started, response_started = False, False
                # Handle the stream events
                async for chunk in streaming:
                    if chunk.type == 'message_start':
                        continue
                    elif chunk.type == 'content_block_start':
                        thinking_started, response_started = False, False
                        continue
                    elif chunk.type == 'content_block_delta':
                        if chunk.delta.type == "thinking_delta":
                            if not thinking_started:
                                yield {"type": "thinking", "data": ""}
                                thinking_started = True
                            yield {"type": "thinking", "data": chunk.delta.thinking}
                        elif chunk.delta.type == "text_delta":
                            if not response_started:
                                yield {"type": "response", "data": ""}
                                response_started = True
                            yield {"type": "response", "data": chunk.delta.text}
                    elif chunk.type == 'content_block_stop':
                        yield {"type": "block_stop", "data": "\n"}
                        continue
                    elif chunk.type == 'message_delta':
                        continue
                    elif chunk.type == 'message_stop':
                        break
                
                # Get the final message
                final_message = await streaming.get_final_message()
                # yield {"type": "completion", "data": final_message.content}
                
                # Check if there are tool uses that need to be handled
                tool_uses = [block for block in final_message.content if block.type == 'tool_use']
                
                assistant_msg = Message.assistant_message(final_message.content)
                self.memory.add_message(assistant_msg)

                if tool_uses:
                    # Process each tool use
                    for tool_use in tool_uses:
                        await self.act(tool_use)

                    # Continue the conversation with tool results - trigger more thinking
                    async for next_event in self.stream(stream, **kwargs):
                        yield next_event
            
        except Exception as e:
            yield {"type": "error", "data": str(e)}
        
