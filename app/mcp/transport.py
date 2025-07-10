from contextlib import asynccontextmanager
import anyio
import asyncio
from typing import Tuple

from app.logger import logger
from mcp.types import JSONRPCMessage
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

MessageStream = tuple[
    MemoryObjectReceiveStream[JSONRPCMessage | Exception],
    MemoryObjectSendStream[JSONRPCMessage],
]


class QueueStreamReader:
    def __init__(self, receive_stream: anyio.abc.ObjectReceiveStream[bytes]):
        self._recv = receive_stream

    async def read(self, n: int = -1) -> bytes:
        return await self._recv.receive()

    async def readexactly(self, n: int) -> bytes:
        return await self.read(n)

    def at_eof(self) -> bool:
        return False


class QueueStreamWriter:
    def __init__(self, send_stream: anyio.abc.ObjectSendStream[bytes]):
        self._send = send_stream

    def write(self, data: bytes) -> None:
        asyncio.create_task(self._send.send(data))

    async def drain(self) -> None:
        # no-op
        await asyncio.sleep(0)

    async def close(self) -> None:
        await self._send.aclose()


class MemoryTransport:
    """Simulate a bidirectional stdio pipe via anyio.Queues."""

    def __init__(self, capacity: int = 0):
        self.client_send, self.client_receive = anyio.create_memory_object_stream[JSONRPCMessage | Exception](capacity)
        self.server_send, self.server_receive = anyio.create_memory_object_stream[JSONRPCMessage | Exception](capacity)

    def get_client_streams(self) -> Tuple[MessageStream, MessageStream]:
        return self.server_receive, self.client_send

    def get_server_streams(self) -> Tuple[MessageStream, MessageStream]:
        return self.client_receive, self.server_send
    

@asynccontextmanager
async def custom_client(transport: MemoryTransport):
    """Client transport for custom: this will connect to a server by memory queue transport."""

    read_stream, write_stream = transport.get_client_streams()

    try:
        yield read_stream, write_stream
    finally:
        await transport.client_send.aclose()


@asynccontextmanager
async def custom_server(transport: MemoryTransport):
    """Server transport for custom: this will connect to a client by memory queue transport."""

    read_stream, write_stream = transport.get_server_streams()

    try:
        yield read_stream, write_stream
    finally:
        await transport.server_send.aclose()