import json
import math
import inspect
import httpx
import tiktoken

from typing import Dict, List, Optional, Union, Any, Tuple
from functools import lru_cache, partial

from openai import (
    APIError,
    AuthenticationError,
    OpenAIError,
    RateLimitError,
)
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from anthropic.types import Message as AnthropicMessage
from anthropic.types.content_block import ContentBlock
from anthropic import AnthropicError

from tenacity import (
    retry,
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from app.config import LLMSettings, config, API_PROVIDERS
from app.exceptions import TokenLimitExceeded
from app.logger import logger
from app.schema import (
    ROLE_VALUES,
    TOOL_CHOICE_TYPE,
    TOOL_CHOICE_VALUES,
    Function,
    Message,
    MockMessage,
    ToolCall,
    ToolChoice,
)

REASONING_MODELS = ["o1", "o3-mini"]
MULTIMODAL_MODELS = [
    "gpt-4-vision-preview",
    "gpt-4o",
    "gpt-4o-mini",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
]

ResponseType = Union[ChatCompletion, AnthropicMessage]
MessageContentType = Union[ChatCompletionMessage, List[ContentBlock]]


def build_llm_client(llm_config: LLMSettings) -> Tuple[Any, Any]:
    """
    Returns (client, tokenizer) instantiated for the given settings.
    Caches so repeated calls with same config_name/settings reuse.
    """
    client = make_llm_client(llm_config, llm_config.api_type)
    # Wrap client.tokenizer if needed, or None
    return client, _build_tokenizer(llm_config.model)
    

def make_llm_client(llm_config: LLMSettings, api_type: Any) -> Any:
    sig = inspect.signature(api_type.__init__)
    init_params = set(sig.parameters) - {"self"}
    cfg = llm_config.model_dump(exclude={"api_type", "proxy"})
    kwargs = {k: v for k, v in cfg.items() if k in init_params and v is not None}
    proxy_client = httpx.AsyncClient(
        proxy=httpx.Proxy(url=llm_config.proxy),
        timeout=30,
    )
    kwargs["http_client"] = proxy_client
    # logger.info(f"Get llm client kwargs: {kwargs}")
    return api_type(**kwargs)


def _build_tokenizer(model: Any) -> Any:
    try:
        tokenizer = tiktoken.encoding_for_model(model)
    except KeyError:
        # If the model is not in tiktoken's presets, use cl100k_base as default
        tokenizer = tiktoken.get_encoding("cl100k_base")
    return tokenizer


# general retry decorator
def with_retry(
    *,
    retries: int = 6,
    min_wait: float = 1,
    max_wait: float = 60,
):
    retryable = retry_if_exception_type((OpenAIError, AnthropicError, ValueError, Exception))
    non_retryable = retry_if_not_exception_type(TokenLimitExceeded)

    def _decorator(func):
        @retry(
            stop=stop_after_attempt(retries),
            wait=wait_random_exponential(min=min_wait, max=max_wait),
            retry=retryable & non_retryable,
            reraise=True,
        )
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)
        return wrapper
    return _decorator


class TokenCounter:
    """manage token counting"""

    # Token constants
    BASE_MESSAGE_TOKENS = 4
    FORMAT_TOKENS = 2
    LOW_DETAIL_IMAGE_TOKENS = 85
    HIGH_DETAIL_TILE_TOKENS = 170

    # Image processing constants
    MAX_SIZE = 2048
    HIGH_DETAIL_TARGET_SHORT_SIDE = 768
    TILE_SIZE = 512

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def count_text(self, text: str) -> int:
        """Calculate tokens for a text string"""
        return 0 if not text else len(self.tokenizer.encode(text))

    def count_image(self, image_item: dict) -> int:
        """
        Calculate tokens for an image based on detail level and dimensions

        For "low" detail: fixed 85 tokens
        For "high" detail:
        1. Scale to fit in 2048x2048 square
        2. Scale shortest side to 768px
        3. Count 512px tiles (170 tokens each)
        4. Add 85 tokens
        """
        detail = image_item.get("detail", "medium")

        # For low detail, always return fixed token count
        if detail == "low":
            return self.LOW_DETAIL_IMAGE_TOKENS

        # For medium detail (default in OpenAI), use high detail calculation
        # OpenAI doesn't specify a separate calculation for medium

        # For high detail, calculate based on dimensions if available
        if detail == "high" or detail == "medium":
            # If dimensions are provided in the image_item
            if "dimensions" in image_item:
                width, height = image_item["dimensions"]
                return self._calculate_high_detail_tokens(width, height)

        return (
            self._calculate_high_detail_tokens(1024, 1024) if detail == "high" else 1024
        )

    def _calculate_high_detail_tokens(self, width: int, height: int) -> int:
        """Calculate tokens for high detail images based on dimensions"""
        # Step 1: Scale to fit in MAX_SIZE x MAX_SIZE square
        if width > self.MAX_SIZE or height > self.MAX_SIZE:
            scale = self.MAX_SIZE / max(width, height)
            width = int(width * scale)
            height = int(height * scale)

        # Step 2: Scale so shortest side is HIGH_DETAIL_TARGET_SHORT_SIDE
        scale = self.HIGH_DETAIL_TARGET_SHORT_SIDE / min(width, height)
        scaled_width = int(width * scale)
        scaled_height = int(height * scale)

        # Step 3: Count number of 512px tiles
        tiles_x = math.ceil(scaled_width / self.TILE_SIZE)
        tiles_y = math.ceil(scaled_height / self.TILE_SIZE)
        total_tiles = tiles_x * tiles_y

        # Step 4: Calculate final token count
        return (
            total_tiles * self.HIGH_DETAIL_TILE_TOKENS
        ) + self.LOW_DETAIL_IMAGE_TOKENS

    def count_content(self, content: Union[str, List[Union[str, dict]]]) -> int:
        """Calculate tokens for message content"""
        if not content:
            return 0

        if isinstance(content, str):
            return self.count_text(content)

        token_count = 0
        for item in content:
            if isinstance(item, str):
                token_count += self.count_text(item)
            elif isinstance(item, dict):
                if "text" in item:
                    token_count += self.count_text(item["text"])
                elif "image_url" in item:
                    token_count += self.count_image(item)
        return token_count

    def count_tool_calls(self, tool_calls: List[dict]) -> int:
        """Calculate tokens for tool calls"""
        token_count = 0
        for tool_call in tool_calls:
            if "function" in tool_call:
                function = tool_call["function"]
                token_count += self.count_text(function.get("name", ""))
                token_count += self.count_text(function.get("arguments", ""))
        return token_count

    def count_message_tokens(self, messages: List[dict]) -> int:
        """Calculate the total number of tokens in a message list"""
        total_tokens = self.FORMAT_TOKENS  # Base format tokens

        for message in messages:
            tokens = self.BASE_MESSAGE_TOKENS  # Base tokens per message

            # Add role tokens
            tokens += self.count_text(message.get("role", ""))

            # Add content tokens
            if "content" in message:
                tokens += self.count_content(message["content"])

            # Add tool calls tokens
            if "tool_calls" in message:
                tokens += self.count_tool_calls(message["tool_calls"])

            # Add name and tool_call_id tokens
            tokens += self.count_text(message.get("name", ""))
            tokens += self.count_text(message.get("tool_call_id", ""))

            total_tokens += tokens

        return total_tokens


class LLM:
    """provide management and ask of llm"""

    @classmethod
    @lru_cache(maxsize=None)
    def for_config(cls, config_name: str = "regular") -> "LLM":
        return cls(config_name)
    
    def __init__(self, config_name: str = "regular"):
        try:
            llm_config: LLMSettings = config.llm[config_name]
        except KeyError:
            raise ValueError(f"No LLM configuration found for {config_name}")
        self.client_name = config_name
        self.client, self.tokenizer = build_llm_client(llm_config)

        params = llm_config.model_dump(exclude={"base_url", "api_key", "api_type", "proxy"})
        if config_name == "regular":
            params.pop("extra_headers", None)
        self.base_params = {}
        for k, v in params.items():
            if v is not None:
                self.base_params[k] = v

        # logger.info(f"Current base params: {self.base_params}")
        # for k, v in self.__dict__.items():
        #     print(f"{k}: {v}")
            
        # Add token counting related attributes
        self.total_input_tokens = 0
        self.total_completion_tokens = 0
        self.max_input_tokens = params.get("max_input_tokens")
        
        self.model = self.base_params.get("model")
        self.token_counter = TokenCounter(self.tokenizer)

    def count_tokens(self, text: str) -> int:
        """Calculate the number of tokens in a text"""
        if not text:
            return 0
        return len(self.tokenizer.encode(text))

    def count_message_tokens(self, messages: List[dict]) -> int:
        return self.token_counter.count_message_tokens(messages)

    def update_token_count(self, input_tokens: int, completion_tokens: int = 0) -> None:
        """Update token counts"""
        # Only track tokens if max_input_tokens is set
        self.total_input_tokens += input_tokens
        self.total_completion_tokens += completion_tokens
        logger.info(
            f"Token usage: Input={input_tokens}, Completion={completion_tokens}, "
            f"Cumulative Input={self.total_input_tokens}, Cumulative Completion={self.total_completion_tokens}, "
            f"Total={input_tokens + completion_tokens}, Cumulative Total={self.total_input_tokens + self.total_completion_tokens}"
        )

    def check_token_limit(self, input_tokens: int) -> bool:
        """Check if token limits are exceeded"""
        if self.max_input_tokens is not None:
            return (self.total_input_tokens + input_tokens) >= self.max_input_tokens
        # If max_input_tokens is not set, always return False
        return False

    def get_limit_error_message(self, input_tokens: int) -> str:
        """Generate error message for token limit exceeded"""
        if (
            self.max_input_tokens is not None
            and (self.total_input_tokens + input_tokens) > self.max_input_tokens
        ):
            return f"Request may exceed input token limit (Current: {self.total_input_tokens}, \
                Needed: {input_tokens}, Max: {self.max_input_tokens})"

        return "Token limit exceeded"

    @staticmethod
    def format_messages(
        messages: List[Union[dict, Message]], supports_images: bool = False
    ) -> List[dict]:
        """
        Format messages for LLM by converting them to OpenAI message format.

        Args:
            messages: List of messages that can be either dict or Message objects
            supports_images: Flag indicating if the target model supports image inputs

        Returns:
            List[dict]: List of formatted messages in OpenAI format

        Raises:
            ValueError: If messages are invalid or missing required fields
            TypeError: If unsupported message types are provided

        Examples:
            >>> msgs = [
            ...     Message.system_message("You are a helpful assistant"),
            ...     {"role": "user", "content": "Hello"},
            ...     Message.user_message("How are you?")
            ... ]
            >>> formatted = LLM.format_messages(msgs)
        """
        formatted_messages = []

        for message in messages:
            # Convert Message objects to dictionaries
            if isinstance(message, Message):
                message = message.to_dict()

            if isinstance(message, dict):
                # If message is a dict, ensure it has required fields
                if "role" not in message:
                    raise ValueError("Message dict must contain 'role' field")

                # Process base64 images if present and model supports images
                if supports_images and message.get("base64_image"):
                    # Initialize or convert content to appropriate format
                    if not message.get("content"):
                        message["content"] = []
                    elif isinstance(message["content"], str):
                        message["content"] = [
                            {"type": "text", "text": message["content"]}
                        ]
                    elif isinstance(message["content"], list):
                        # Convert string items to proper text objects
                        message["content"] = [
                            (
                                {"type": "text", "text": item}
                                if isinstance(item, str)
                                else item
                            )
                            for item in message["content"]
                        ]

                    # Add the image to content
                    message["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{message['base64_image']}"
                            },
                        }
                    )

                    # Remove the base64_image field
                    del message["base64_image"]
                # If model doesn't support images but message has base64_image, handle gracefully
                elif not supports_images and message.get("base64_image"):
                    # Just remove the base64_image field and keep the text content
                    del message["base64_image"]

                if "content" in message or "tool_calls" in message:
                    formatted_messages.append(message)
                # else: do not include the message
            else:
                raise TypeError(f"Unsupported message type: {type(message)}")

        # Validate all messages have required fields
        for msg in formatted_messages:
            if msg["role"] not in ROLE_VALUES:
                raise ValueError(f"Invalid role: {msg['role']}")

        return formatted_messages

    @with_retry()
    async def ask(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = True,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Send a prompt to the LLM and get the response.

        Args:
            messages: List of conversation messages
            system_msgs: Optional system messages to prepend
            stream (bool): Whether to stream the response
            temperature (float): Sampling temperature for the response

        Returns:
            str: The generated response

        Raises:
            TokenLimitExceeded: If token limits are exceeded
            ValueError: If messages are invalid or response is empty
            OpenAIError: If API call fails after retries
            Exception: For unexpected errors
        """
        try:
            # Check if the model supports images
            supports_images = self.model in MULTIMODAL_MODELS

            # Format system and user messages with image support check
            if system_msgs:
                system_msgs = self.format_messages(system_msgs, supports_images)
                messages = system_msgs + self.format_messages(messages, supports_images)
            else:
                messages = self.format_messages(messages, supports_images)

            # Calculate input token count
            input_tokens = self.count_message_tokens(messages)

            # Check if token limits are exceeded
            if self.check_token_limit(input_tokens):
                error_message = self.get_limit_error_message(input_tokens)
                # Raise a special exception that won't be retried
                raise TokenLimitExceeded(error_message)

            params = {
                "model": self.model,
                "messages": messages,
            }

            if self.model in REASONING_MODELS:
                params["max_completion_tokens"] = self.max_tokens
            else:
                params["max_tokens"] = self.max_tokens
                params["temperature"] = (
                    temperature if temperature is not None else self.temperature
                )

            # Non-streaming request
            if not stream:
                response = await self.client.chat.completions.create(
                    **params, stream=False
                )

                if not response.choices or not response.choices[0].message.content:
                    raise ValueError("Empty or invalid response from LLM")

                # Update token counts
                self.update_token_count(
                    response.usage.prompt_tokens, response.usage.completion_tokens
                )

                return response.choices[0].message.content

            # Streaming request, For streaming, update estimated token count before making the request
            self.update_token_count(input_tokens)

            response = await self.client.chat.completions.create(**params, stream=True)

            collected_messages = []
            completion_text = ""
            async for chunk in response:
                chunk_message = chunk.choices[0].delta.content or ""
                collected_messages.append(chunk_message)
                completion_text += chunk_message
                print(chunk_message, end="", flush=True)

            print()  # Newline after streaming
            full_response = "".join(collected_messages).strip()
            if not full_response:
                raise ValueError("Empty response from streaming LLM")

            # estimate completion tokens for streaming response
            completion_tokens = self.count_tokens(completion_text)
            logger.info(
                f"Estimated completion tokens for streaming response: {completion_tokens}"
            )
            self.total_completion_tokens += completion_tokens

            return full_response

        except TokenLimitExceeded:
            # directly raise token limit errors without logging and retry
            raise
        except ValueError:
            logger.exception(f"Validation error")
            raise
        except OpenAIError as oe:
            logger.exception(f"OpenAI API error")
            if isinstance(oe, AuthenticationError):
                logger.error("Authentication failed. Check API key.")
            elif isinstance(oe, RateLimitError):
                logger.error("Rate limit exceeded. Consider increasing retry attempts.")
            elif isinstance(oe, APIError):
                logger.error(f"API error: {oe}")
            raise
        except Exception:
            logger.exception(f"Unexpected error in ask")
            raise

    @with_retry()
    async def ask_with_images(
        self,
        messages: List[Union[dict, Message]],
        images: List[Union[str, dict]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Send a prompt with images to the LLM and get the response.

        Args:
            messages: List of conversation messages
            images: List of image URLs or image data dictionaries
            system_msgs: Optional system messages to prepend
            stream (bool): Whether to stream the response
            temperature (float): Sampling temperature for the response

        Returns:
            str: The generated response

        Raises:
            TokenLimitExceeded: If token limits are exceeded
            ValueError: If messages are invalid or response is empty
            OpenAIError: If API call fails after retries
            Exception: For unexpected errors
        """
        try:
            # For ask_with_images, we always set supports_images to True because
            # this method should only be called with models that support images
            if self.model not in MULTIMODAL_MODELS:
                raise ValueError(
                    f"Model {self.model} does not support images. Use a model from {MULTIMODAL_MODELS}"
                )

            # Format messages with image support
            formatted_messages = self.format_messages(messages, supports_images=True)

            # Ensure the last message is from the user to attach images
            if not formatted_messages or formatted_messages[-1]["role"] != "user":
                raise ValueError(
                    "The last message must be from the user to attach images"
                )

            # Process the last user message to include images
            last_message = formatted_messages[-1]

            # Convert content to multimodal format if needed
            content = last_message["content"]
            multimodal_content = (
                [{"type": "text", "text": content}]
                if isinstance(content, str)
                else content
                if isinstance(content, list)
                else []
            )

            # Add images to content
            for image in images:
                if isinstance(image, str):
                    multimodal_content.append(
                        {"type": "image_url", "image_url": {"url": image}}
                    )
                elif isinstance(image, dict) and "url" in image:
                    multimodal_content.append({"type": "image_url", "image_url": image})
                elif isinstance(image, dict) and "image_url" in image:
                    multimodal_content.append(image)
                else:
                    raise ValueError(f"Unsupported image format: {image}")

            # Update the message with multimodal content
            last_message["content"] = multimodal_content

            # Add system messages if provided
            if system_msgs:
                all_messages = (
                    self.format_messages(system_msgs, supports_images=True)
                    + formatted_messages
                )
            else:
                all_messages = formatted_messages

            # Calculate tokens and check limits
            input_tokens = self.count_message_tokens(all_messages)
            if self.check_token_limit(input_tokens):
                raise TokenLimitExceeded(self.get_limit_error_message(input_tokens))

            # Set up API parameters
            params = {
                "model": self.model,
                "messages": all_messages,
                "stream": stream,
            }

            # Add model-specific parameters
            if self.model in REASONING_MODELS:
                params["max_completion_tokens"] = self.max_tokens
            else:
                params["max_tokens"] = self.max_tokens
                params["temperature"] = (
                    temperature if temperature is not None else self.temperature
                )

            # Handle non-streaming request
            if not stream:
                response = await self.client.chat.completions.create(**params)

                if not response.choices or not response.choices[0].message.content:
                    raise ValueError("Empty or invalid response from LLM")

                self.update_token_count(response.usage.prompt_tokens)
                return response.choices[0].message.content

            # Handle streaming request
            self.update_token_count(input_tokens)
            response = await self.client.chat.completions.create(**params)

            collected_messages = []
            async for chunk in response:
                chunk_message = chunk.choices[0].delta.content or ""
                collected_messages.append(chunk_message)
                print(chunk_message, end="", flush=True)

            print()  # Newline after streaming
            full_response = "".join(collected_messages).strip()

            if not full_response:
                raise ValueError("Empty response from streaming LLM")

            return full_response

        except TokenLimitExceeded:
            raise
        except ValueError as ve:
            logger.error(f"Validation error in ask_with_images: {ve}")
            raise
        except OpenAIError as oe:
            logger.error(f"OpenAI API error: {oe}")
            if isinstance(oe, AuthenticationError):
                logger.error("Authentication failed. Check API key.")
            elif isinstance(oe, RateLimitError):
                logger.error("Rate limit exceeded. Consider increasing retry attempts.")
            elif isinstance(oe, APIError):
                logger.error(f"API error: {oe}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in ask_with_images: {e}")
            raise

    @with_retry()
    async def ask_tool(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        timeout: int = 300,
        tools: Optional[List[dict]] = None,
        tool_choice: TOOL_CHOICE_TYPE = ToolChoice.AUTO.value,  # type: ignore
        stream: Optional[bool] = False,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> MessageContentType | None:
        """
        Ask LLM using functions/tools and return the response.

        Args:
            messages: List of conversation messages
            system_msgs: Optional system messages to prepend
            timeout: Request timeout in seconds
            tools: List of tools to use
            tool_choice: Tool choice strategy
            temperature: Sampling temperature for the response
            **kwargs: Additional completion arguments

        Returns:
            MessageContentType: The model's response

        Raises:
            TokenLimitExceeded: If token limits are exceeded
            ValueError: If tools, tool_choice, or messages are invalid
            OpenAIError: If API call fails after retries
            Exception: For unexpected errors
        """
        try:
            # Validate tool_choice
            if tool_choice not in TOOL_CHOICE_VALUES:
                raise ValueError(f"Invalid tool_choice: {tool_choice}")

            # Check if the model supports images
            supports_images = self.model in MULTIMODAL_MODELS

            # We set client name for both regular and streaming mode directly here
            self.client_name = "anthropic"
            
            # Format messages
            if system_msgs and self.client_name != "anthropic":
                system_msgs = self.format_messages(system_msgs, supports_images)
                messages = system_msgs + self.format_messages(messages, supports_images)
            else:
                messages = self.format_messages(messages, supports_images)

            # logger.info(f"Formatted messages for tool request: {messages}")
            # Calculate input token count
            input_tokens = self.count_message_tokens(messages)

            # If there are tools, calculate token count for tool descriptions
            tools_tokens = 0
            if tools:
                for tool in tools:
                    tools_tokens += self.count_tokens(str(tool))

            input_tokens += tools_tokens

            # Check if token limits are exceeded
            if self.check_token_limit(input_tokens):
                error_message = self.get_limit_error_message(input_tokens)
                # Raise a special exception that won't be retried
                raise TokenLimitExceeded(error_message)

            # Validate tools if provided
            if tools:
                for tool in tools:
                    if not isinstance(tool, dict) or "type" not in tool:
                        raise ValueError("Each tool must be a dict with 'type' field")

            # logger.info(f"Successfully get base params: {self.base_params}")

            # Set up the completion request, "stream" shouldn't be set
            params = {
                **self.base_params,
                "messages": messages,
                "tools": tools,
                "tool_choice": tool_choice,
                "timeout": timeout,
                **kwargs,
            }

            if self.model in REASONING_MODELS:
                params["max_completion_tokens"] = params.get("max_tokens")
            else:
                params["max_tokens"] = params.get("max_tokens")
                params["temperature"] = (
                    temperature if temperature is not None else params.get("temperature")
                )

            if self.client_name == "anthropic":
                if system_msgs:
                    params["system"] = system_msgs[0].content

            # logger.info(f"Preparing to send request with parameters: {params}")
            # Ovverride parameters for different clients
            override_params = Parameters(self.client_name, stream, **params)
            client = override_params.request(self.client, stream)
            # logger.info(f"Client ready for request: {client}")
            # logger.info(f"Parameters for request: {override_params._parameters}")

            # Wrapper for streaming ctx and return directly
            if stream:
                response = client(**override_params._parameters)
                return response
            
            response: ChatCompletion | AnthropicMessage = await client(**override_params._parameters)
            # Override response
            override_response = override_params.response_format(response)
            if not override_response:
                logger.info(f"Invalid response from LLM: {response}")
                # raise ValueError("Invalid or empty response from LLM")
                return None

            # Update token counts
            prompt_tokens, completion_tokens = override_params.get_usage_info(response)
            self.update_token_count(prompt_tokens, completion_tokens)

            logger.info(f"Successfully get response: {override_response}")

            # Build final response for external interface in agent-level
            response = override_params.build_response(override_response)
            
            return response

        except TokenLimitExceeded:
            # Re-raise token limit errors without logging
            raise
        except ValueError as ve:
            logger.error(f"Validation error in ask_tool: {ve}")
            raise
        except OpenAIError as oe:
            logger.error(f"OpenAI API error: {oe}")
            if isinstance(oe, AuthenticationError):
                logger.error("Authentication failed. Check API key.")
            elif isinstance(oe, RateLimitError):
                logger.error("Rate limit exceeded. Consider increasing retry attempts.")
            elif isinstance(oe, APIError):
                logger.error(f"API error: {oe}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in ask_tool: {e}")
            raise


class Parameters:
    """Override neccessary parameters for LLM calls"""

    def __init__(self, client: str, stream: bool = False, **kwargs):
        providers = list(API_PROVIDERS.keys())
        self.client: str = client
        if self.client.lower() not in providers:
            raise ValueError(f"Invalid client: {client}. Must be one of {providers}")
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.transform_parameters(stream)

    def transform_parameters(self, stream: bool = False):
        """Transform parameters to match the LLM client's expected format."""
        self._parameters = self.__dict__.copy()
        self._parameters.pop("client", None)

        # logger.info(f"Transforming parameters for client {self.client}: {self._parameters}")
        if self.client.lower() == "openai":
            pass
        elif self.client.lower() == "anthropic":
            # handle tool choice
            tool_choice = self._parameters.get("tool_choice", None)
            if tool_choice == "auto":
                self._parameters["tool_choice"] = {"type": "auto"}
            elif tool_choice == "required":
                self._parameters["tool_choice"] = {"type": "any"}
            elif tool_choice == "none":
                self._parameters.pop("tool_choice", None)

            # Convert tools format
            tools = self._parameters.get("tools", None)
            if tools:
                anthropic_tools = []
                for tool in tools:
                    if tool.get("type") == "function":
                        function_def = tool["function"]
                        anthropic_tool = {
                            "name": function_def["name"],
                            "description": function_def["description"],
                            "input_schema": function_def["parameters"]
                        }
                        anthropic_tools.append(anthropic_tool)
                self._parameters["tools"] = anthropic_tools

            # handle reasoning models
            self._parameters.pop("max_completion_tokens", None)

            # For stream Claude request, directly returns
            # As we have already converted the system messages to Anthropic format
            if stream:
               return
            
            # handle Anthropic-type messages
            messages = self._parameters.get("messages", [])
            converted_msgs = []
            for msg in messages:
                if msg["role"] == "system":
                    continue
                elif msg["role"] == "assistant":
                    content = []
                    if msg.get("content"):
                        content.append({"type": "text", "text": msg["content"]})
                    
                    if msg.get("tool_calls"):
                        for tool_call in msg["tool_calls"]:
                            content.append({
                                "type": "tool_use",
                                "id": tool_call["id"],
                                "name": tool_call["function"]["name"],
                                "input": json.loads(tool_call["function"]["arguments"])
                            })

                    converted_msgs.append({
                        "role": "assistant", 
                        "content": content
                    })
                elif msg["role"] == "tool":
                    converted_msgs.append({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": msg["tool_call_id"],
                            "content": msg["content"]
                        }]
                    })
                else:
                    converted_msgs.append(msg)

            self._parameters["messages"] = converted_msgs

    def request(self, client: Any, stream: bool) -> Any:
        if self.client.lower() == "openai":
            return client.chat.completions.create
        elif self.client.lower() == "anthropic":
            if stream:
                return client.messages.stream
            return client.messages.create

    def response_format(self, response: ResponseType) -> Union[MessageContentType, None]:
        """Return response format if available."""
        try:
            if self.client.lower() == "openai":
                return response.choices[0].message.content if response.choices else None
            elif self.client.lower() == "anthropic":
                return response.content if hasattr(response, 'content') and response.content else None
        except (AttributeError, IndexError) as e:
            logger.error(f"Error formatting response: {e}")
            raise e
        
    def get_usage_info(self, response: ResponseType) -> tuple[int, int]:
        """Extract token usage from response."""
        try:
            if self.client.lower() == "openai":
                usage = response.usage
                return usage.prompt_tokens, usage.completion_tokens
            elif self.client.lower() == "anthropic":
                usage = response.usage
                return usage.input_tokens, usage.output_tokens
        except AttributeError:
            logger.warning("Could not extract usage information from response")
            return 0, 0
        
    def build_response(
            self, 
            override_reponse: MessageContentType
        ) -> ChatCompletionMessage | MockMessage:
        """Build final response for external interface in agent-level."""
        if self.client.lower() == "openai":
            return override_reponse
        elif self.client.lower() == "anthropic":
            if isinstance(override_reponse, list):
                tool_calls = []
                content = ""
                for block in override_reponse:
                    if hasattr(block, 'type') and block.type == 'text':
                        content = block.text
                    elif hasattr(block, 'type') and block.type == 'tool_use':
                        tool_calls.append(ToolCall(
                            id=block.id,
                            type="function",
                            function=Function(
                                name=block.name,
                                arguments=json.dumps(block.input, ensure_ascii=False)   # garantee False ascii
                            )
                        ))
                return MockMessage(
                    content=content,
                    tool_calls=tool_calls
                )
            return override_reponse
        else:
            raise ValueError(f"Unsupported client: {self.client}")