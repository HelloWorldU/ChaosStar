class ToolError(Exception):
    """Raised when a tool encounters an error."""

    def __init__(self, message):
        super().__init__(message)
        self.message = message


class DeepResearchError(Exception):
    """Base exception for all DeepResearch errors"""


class TokenLimitExceeded(DeepResearchError):
    """Exception raised when the token limit is exceeded"""
