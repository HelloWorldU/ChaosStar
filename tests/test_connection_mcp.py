import asyncio
import pytest
import importlib.util
import pathlib

from mcp import ClientSession

root = pathlib.Path(__file__).parent.parent
spec_t = importlib.util.spec_from_file_location("transport", root/"app"/"mcp"/"transport.py")
mod_t = importlib.util.module_from_spec(spec_t); spec_t.loader.exec_module(mod_t)

spec_s = importlib.util.spec_from_file_location("server", root/"app"/"mcp"/"server.py")
mod_s = importlib.util.module_from_spec(spec_s); spec_s.loader.exec_module(mod_s)

spec_c = importlib.util.spec_from_file_location("client", root/"app"/"mcp"/"client.py")
mod_c = importlib.util.module_from_spec(spec_c); spec_c.loader.exec_module(mod_c)

MemoryTransport = mod_t.MemoryTransport
custom_client    = mod_t.custom_client
custom_server    = mod_t.custom_server
MCPServer        = mod_s.MCPServer
MCPClient        = mod_c.MCPClient

@pytest.mark.asyncio
async def test_memory_mcp_echo_tool():
    transport = MemoryTransport()
    server = MCPServer(name="TestServer")

    async def echo_tool(data: str):
        return data + "_echo"

    server.server.tool()(echo_tool)

    async with asyncio.TaskGroup() as tg:
        ready = asyncio.Event()
        tg.create_task(server.run_custom_async(transport, ready_event=ready))

        async def client_job():
            mcp_client = MCPClient()
            async with custom_client(transport) as (r, w):
                sess = await ClientSession(r, w).__aenter__()
                await sess.initialize()

                resp = await sess.call_tool("echo_tool", {"data": "hello"})
                assert resp.content[0].text == "hello_echo"

                await sess.__aexit__(None, None, None)

        tg.create_task(client_job())
        