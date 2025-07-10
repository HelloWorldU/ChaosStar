import json
import threading
import tomllib
import torch
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal, Union
from pydantic import BaseModel, Field, TypeAdapter, field_validator, model_validator
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic, Anthropic

def get_project_root() -> Path:
    """Get the project root directory"""
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = get_project_root()
WORKSPACE_ROOT = PROJECT_ROOT / "workspace"

API_PROVIDERS: dict[str, type] = {
    "openai":    AsyncOpenAI,
    "anthropic": AsyncAnthropic,
}

BETA_HEADERS   = ",".join([
    "interleaved-thinking-2025-05-14",
    "fine-grained-tool-streaming-2025-05-14",
])

def beta_headers() -> dict[str, str]:
    """Return extra headers for every call."""
    return {"anthropic-beta": BETA_HEADERS}


class LLMSettings(BaseModel):
    model: str = Field(..., description="Model name")
    base_url: str = Field(..., description="API base URL")
    api_key: str = Field(..., description="API key")
    max_tokens: int = Field(4096, description="Maximum number of tokens per request")
    max_input_tokens: Optional[int] = Field(
        None,
        description="Maximum input tokens to use across all requests (None for unlimited)",
    )
    temperature: float = Field(1.0, description="Sampling temperature")
    api_type: str = Field(..., description="Azure, Openai, or Anthropic")
    proxy: str = Field(..., description="Http/Local port url for eable vpn")
    extra_headers: dict[str, str] = Field(
        default_factory=beta_headers, 
        description="Extra headers for Anthropic call"
    )
    thinking: Optional[Union[int, dict]] = Field(
        None, description="Thinking config tokens to use (Anthropic API only)"
    )
    
    @field_validator("api_type", mode="after")
    def bind_api_type(cls, v) -> "type":
        try:
            return API_PROVIDERS[v.lower()]
        except KeyError:
            raise ValueError(f"Unknown api_type: {v!r}")

    @model_validator(mode="after")
    def process_api_dependent_fields(self):
        api_type_str = None
        for key, val in API_PROVIDERS.items():
            if val == self.api_type:
                api_type_str = key
                break
        
        if api_type_str != "anthropic":
            self.extra_headers = None
        
        if self.thinking is not None and api_type_str == "anthropic":
            if isinstance(self.thinking, int):
                self.thinking = {"type": "enabled", "budget_tokens": self.thinking}
        else:
            self.thinking = None
        
        return self
        

# load llm settings for llm client
LLMAdapter = TypeAdapter(LLMSettings)


class ProxySettings(BaseModel):
    server: Optional[str] = Field(None, description="Proxy server address")
    username: Optional[str] = Field(None, description="Proxy username")
    password: Optional[str] = Field(None, description="Proxy password")


class SearchSettings(BaseModel):
    engine: str = Field(default="Google", description="Search engine the llm to use")
    fallback_engines: List[str] = Field(
        default_factory=lambda: ["DuckDuckGo", "Baidu", "Bing"],
        description="Fallback search engines to try if the primary engine fails",
    )
    retry_delay: int = Field(
        default=60,
        description="Seconds to wait before retrying all engines again after they all fail",
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of times to retry all engines when all fail",
    )
    lang: str = Field(
        default="en",
        description="Language code for search results (e.g., en, zh, fr)",
    )
    country: str = Field(
        default="us",
        description="Country code for search results (e.g., us, cn, uk)",
    )


class BrowserSettings(BaseModel):
    headless: bool = Field(False, description="Whether to run browser in headless mode")
    disable_security: bool = Field(
        True, description="Disable browser security features"
    )
    extra_chromium_args: List[str] = Field(
        default_factory=list, description="Extra arguments to pass to the browser"
    )
    chrome_instance_path: Optional[str] = Field(
        None, description="Path to a Chrome instance to use"
    )
    wss_url: Optional[str] = Field(
        None, description="Connect to a browser instance via WebSocket"
    )
    cdp_url: Optional[str] = Field(
        None, description="Connect to a browser instance via CDP"
    )
    proxy: Optional[ProxySettings] = Field(
        None, description="Proxy settings for the browser"
    )
    max_content_length: int = Field(
        2000, description="Maximum length for content retrieval operations"
    )


class SandboxSettings(BaseModel):
    """Configuration for the execution sandbox"""

    use_sandbox: bool = Field(False, description="Whether to use the sandbox")
    image: str = Field("python:3.12-slim", description="Base image")
    work_dir: str = Field("/workspace", description="Container working directory")
    memory_limit: str = Field("512m", description="Memory limit")
    cpu_limit: float = Field(1.0, description="CPU limit")
    timeout: int = Field(300, description="Default command timeout (seconds)")
    network_enabled: bool = Field(
        False, description="Whether network access is allowed"
    )


class MCPServerConfig(BaseModel):
    """Configuration for a single MCP server"""

    type: str = Field(..., description="Server connection type (sse or stdio)")
    url: Optional[str] = Field(None, description="Server URL for SSE connections")
    command: Optional[str] = Field(None, description="Command for stdio connections")
    args: List[str] = Field(
        default_factory=list, description="Arguments for stdio command"
    )


class MCPSettings(BaseModel):
    """Configuration for MCP (Model Context Protocol)"""

    server_reference: str = Field(
        default="app.mcp.server", description="Module reference for the MCP server"
    )
    servers: Dict[str, MCPServerConfig] = Field(
        default_factory=dict, description="MCP server configurations"
    )

    @classmethod
    def load_server_config(cls) -> Dict[str, MCPServerConfig]:
        """Load MCP server configuration from JSON file"""
        config_path = PROJECT_ROOT / "config" / "mcp.example.json"

        try:
            config_file = config_path if config_path.exists() else None
            if not config_file:
                return {}

            with config_file.open() as f:
                data = json.load(f)
                servers = {}

                for server_id, server_config in data.get("mcpServers", {}).items():
                    servers[server_id] = MCPServerConfig(
                        type=server_config["type"],
                        url=server_config.get("url"),
                        command=server_config.get("command"),
                        args=server_config.get("args", []),
                    )
                return servers
        except Exception as e:
            raise ValueError(f"Failed to load MCP server config: {e}")


class AppConfig(BaseModel):
    llm: Dict[str, LLMSettings]
    sandbox: Optional[SandboxSettings] = Field(
        None, description="Sandbox configuration"
    )
    browser_config: Optional[BrowserSettings] = Field(
        None, description="Browser configuration"
    )
    search_config: Optional[SearchSettings] = Field(
        None, description="Search configuration"
    )
    mcp_config: Optional[MCPSettings] = Field(
        None, description="MCP configuration"
    )

    class Config:
        arbitrary_types_allowed = True


class Config:
    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self._config = None
                    self._load_initial_config()
                    self._initialized = True

    @staticmethod
    def _get_config_path() -> Path:
        root = PROJECT_ROOT
        config_path = root / "config" / "config.toml"
        if config_path.exists():
            return config_path
        example_path = root / "config" / "config.example.test.toml"
        if example_path.exists():
            return example_path
        raise FileNotFoundError("No configuration file found in config directory")

    def _load_config(self) -> dict:
        config_path = self._get_config_path()
        with config_path.open("rb") as f:
            return tomllib.load(f)

    def _load_initial_config(self) -> None:
        """Load the initial configuration from the config file and save in self._config"""
        raw_config = self._load_config()
        base_llm = raw_config.get("llm", {})
        llm_cfg: Dict[str, LLMSettings] = {
            name: LLMAdapter.validate_python(block)
            for name, block in base_llm.items()
            if isinstance(block, dict)
        }

        # print(llm_cfg)
        # handle browser config.
        browser_config = raw_config.get("browser", {})
        browser_settings = None

        if browser_config:
            # handle proxy settings.
            proxy_config = browser_config.get("proxy", {})
            proxy_settings = None

            if proxy_config and proxy_config.get("server"):
                proxy_settings = ProxySettings(
                    **{
                        k: v
                        for k, v in proxy_config.items()
                        if k in ["server", "username", "password"] and v
                    }
                )

            # filter valid browser config parameters.
            valid_browser_params = {
                k: v
                for k, v in browser_config.items()
                if k in BrowserSettings.__annotations__ and v is not None
            }

            # if there is proxy settings, add it to the parameters.
            if proxy_settings:
                valid_browser_params["proxy"] = proxy_settings

            # only create BrowserSettings when there are valid parameters.
            if valid_browser_params:
                browser_settings = BrowserSettings(**valid_browser_params)

        search_config = raw_config.get("search", {})
        search_settings = None
        if search_config:
            search_settings = SearchSettings(**search_config)

        sandbox_config = raw_config.get("sandbox", {})
        sandbox_settings = None
        if sandbox_config:
            sandbox_settings = SandboxSettings(**sandbox_config)

        mcp_config = raw_config.get("mcp", {})
        mcp_settings = None
        if mcp_config:
            # Load server configurations from JSON
            mcp_config["servers"] = MCPSettings.load_server_config()
            mcp_settings = MCPSettings(**mcp_config)
        else:
            mcp_settings = MCPSettings(servers=MCPSettings.load_server_config())

        config_dict = {
            "sandbox": sandbox_settings,
            "browser_config": browser_settings,
            "search_config": search_settings,
            "mcp_config": mcp_settings,
        }

        self._config = AppConfig(llm=llm_cfg, **config_dict)

    @property
    def llm(self) -> Dict[str, LLMSettings]:
        if self._config is None:
            raise RuntimeError("Configuration not loaded")
        return self._config.llm

    @property
    def sandbox(self) -> Optional[SandboxSettings]:
        if self._config is None:
            raise RuntimeError("Configuration not loaded")
        return self._config.sandbox

    @property
    def browser_config(self) -> Optional[BrowserSettings]:
        if self._config is None:
            raise RuntimeError("Configuration not loaded")
        return self._config.browser_config

    @property
    def search_config(self) -> Optional[SearchSettings]:
        if self._config is None:
            raise RuntimeError("Configuration not loaded")
        return self._config.search_config

    @property
    def mcp_config(self) -> Optional[MCPSettings]:
        if self._config is None:
            raise RuntimeError("Configuration not loaded")
        return self._config.mcp_config

    @property
    def workspace_root(self) -> Path:
        return WORKSPACE_ROOT

    @property
    def root_path(self) -> Path:
        return PROJECT_ROOT


config = Config()
