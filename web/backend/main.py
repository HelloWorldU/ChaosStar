import sys
import asyncio

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

sys.path.append("../..")
import json

from json import JSONDecodeError
from app.agent import ChaosStar, StreamChaosStar
from app.logger import logger
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from models import ChatRequest, ChatResponse


class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}
        self.stream_tasks: dict[str, asyncio.Task] = {}
    
    async def connect(self, websocket: WebSocket, connection_id: str):
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        logger.info(f"WebSocket connected: {connection_id}")
    
    def disconnect(self, connection_id: str):
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
            logger.info(f"WebSocket disconnected: {connection_id}")
    
    async def send_personal_message(self, message: dict, connection_id: str):
        if connection_id in self.active_connections:
            try:
                await self.active_connections[connection_id].send_text(
                    json.dumps(message, ensure_ascii=False)
                )
            except Exception as e:
                logger.error(f"Error sending message to {connection_id}: {e}")
                self.disconnect(connection_id)


class AgentManager:
    def __init__(self):
        self.regular_agent = None
        self.stream_agent = None
        self.lock = asyncio.Lock()
        self.connection_manager = ConnectionManager()
    
    async def initialize(self):
        try:
            self.regular_agent = await ChaosStar.create(
                llm_name="regular", 
                name="ChaosStar", 
                max_steps=5
            )
            self.stream_agent = await StreamChaosStar.create(
                llm_name="streaming", 
                name="StreamChaosStar"
            )
            logger.info("Both agents initialized")
        except Exception as e:
            logger.error(f"Failed to initialize agents: {e}")
            raise
    
    async def get_agent(self, stream: bool = False):
        return self.stream_agent if stream else self.regular_agent
    
    async def cleanup(self):
        if self.regular_agent:
            try:
                await self.regular_agent.cleanup()
            except Exception as e:
                logger.error(f"Error during regular agent cleanup: {e}")
        if self.stream_agent:
            try:
                await self.stream_agent.cleanup()
            except Exception as e:
                logger.error(f"Error during stream agent cleanup: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    agent_manager = AgentManager()
    try:
        await agent_manager.initialize()
        app.state.agent_manager = agent_manager
        yield
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
    finally:
        if agent_manager:
            try:
                await agent_manager.cleanup()
                logger.info("Shutdown: agent cleaned up")
            except Exception as e:
                logger.error(f"Shutdown: cleanup error: {e}")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:80"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "ChaosStar API is running"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# RESTful API endpoint for chat interaction
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    agent_manager: AgentManager = app.state.agent_manager
    async with agent_manager.lock:
        try:
            regular_agent = await agent_manager.get_agent()
            results = []
            async for event in regular_agent.run(request.message):
                if event["type"] == "step":
                    results.append(event)
            return ChatResponse(response=results, status="success")
        except Exception as e:
            return ChatResponse(response=f"Error: {str(e)}", status="error")


# WebSocket transport for real-time chat streaming
@app.websocket("/ws/{connection_id}")
async def websocket_endpoint(websocket: WebSocket, connection_id: str):
    agent_manager: AgentManager = app.state.agent_manager
    try:
        await agent_manager.connection_manager.connect(websocket, connection_id)
    except Exception as e:
        logger.error(f"Failed to connect to WebSocket: {e}")

    try:
        while True:
            raw_data = await websocket.receive_text()
            # logger.info(f"WS {connection_id} raw message: {raw_data!r}")
            try:
                message_data = json.loads(raw_data)
            except JSONDecodeError:
                # Continue instead of disconnecting connection
                logger.warning(f"Invalid JSON from {connection_id!r}: {raw_data!r}")
                continue
            
            if message_data["type"] == "chat":
                task = asyncio.create_task(
                    chat_stream(
                        agent_manager,
                        connection_id,
                        message_data["message"],
                    )
                )
                agent_manager.connection_manager.stream_tasks[connection_id] = task
            elif message_data["type"] == "stop":
                task = agent_manager.connection_manager.stream_tasks.get(connection_id)
                if task and not task.done():
                    task.cancel()
                await agent_manager.connection_manager.send_personal_message(
                    {"type": "done", "data": ""}, connection_id
                )
            elif message_data["type"] == "ping":
                await agent_manager.connection_manager.send_personal_message(
                    {"type": "pong", "data": ""}, connection_id
                )
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {connection_id}")
        
    except Exception as e:
        logger.error(f"WebSocket error for {connection_id}: {e}")
        await agent_manager.connection_manager.send_personal_message(
            {"type": "error", "data": str(e)}, connection_id
        )
    finally:
        agent_manager.connection_manager.disconnect(connection_id)


async def chat_stream(agent_manager: AgentManager, connection_id: str, message: str):
    async with agent_manager.lock:
        try:
            stream_agent = await agent_manager.get_agent(stream=True)
            if not stream_agent:
                await agent_manager.connection_manager.send_personal_message(
                    {"type": "error", "data": "Agent not available"}, connection_id
                )
                return
                
            async for event in stream_agent.run(message, stream=True):
                # logger.info(f"Sending event: {event}")
                await agent_manager.connection_manager.send_personal_message(event, connection_id)
            
            await agent_manager.connection_manager.send_personal_message(
                {"type": "done", "data": ""}, connection_id
            )

        except Exception as e:
            await agent_manager.connection_manager.send_personal_message(
                {"type": "error", "data": str(e)}, connection_id
            )
            logger.error(f"Stream error for {connection_id}: {e}")

        except asyncio.CancelledError:
            await agent_manager.connection_manager.send_personal_message(
                {"type":"done","data":""}, connection_id
            )
            return 


# 启动前，首先执行playwright install命令
import uvicorn
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)