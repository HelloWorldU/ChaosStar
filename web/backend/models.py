from pydantic import BaseModel, Field
from typing import Literal, Union, Any, List
from typing_extensions import Annotated, TypeAlias

class RegularChatRequest(BaseModel):
    message: str
    max_steps: int = 2
    stream: Literal[False] = False


class StreamChatRequest(BaseModel):
    message: str
    stream: Literal[True] = True


ChatRequest: TypeAlias = Annotated[
    Union[RegularChatRequest, StreamChatRequest],
    Field(discriminator="stream"),
]


# How to valide and transform the dict automatically?
class ReActResult(BaseModel):
    think: str
    act: str


class StepResult(BaseModel):
    data: ReActResult
    type: Literal["step"]


class StreamResult(BaseModel):
    data: Any
    type: Literal["thinking", "response", "block_stop", "error"]


ResponseResult: TypeAlias = Annotated[
    Union[StepResult, StreamResult],
    Field(discriminator="type"),
]


class ChatResponse(BaseModel):
    response: Union[List[ResponseResult], str]
    status: str