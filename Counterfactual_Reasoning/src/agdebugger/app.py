import logging
import os
from typing import List

from autogen_agentchat.teams._group_chat._events import (
    GroupChatPause,
    GroupChatReset,
    GroupChatResume,
    GroupChatStart,
)
from autogen_core import EVENT_LOGGER_NAME
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .backend import BackendRuntimeManager
from .intervention_utils import write_file_async
from .serialization import deserialize
from .types import (
    EditHistoryMessage,
    EditQueueMessage,
    PublishMessage,
    SendMessage,
)
from .utils import load_app, message_to_json

# alt would be TRACE_LOGGER_NAME
logger = logging.getLogger(EVENT_LOGGER_NAME)
logger.setLevel(logging.DEBUG)

_RPC_ONLY_GROUP_CHAT_MESSAGES = (
    GroupChatStart,
    GroupChatReset,
    GroupChatPause,
    GroupChatResume,
)


async def get_server(module_str: str, message_history=None, state_cache=None) -> FastAPI:
    origins = [
        "http://localhost",
        "http://localhost:5173",
        "http://localhost:*",
    ]
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    api = FastAPI(root_path="/api")
    app.mount("/api", api)
    package_dir = os.path.dirname(os.path.abspath(__file__))
    ui_folder_candidates = [
        os.environ.get("AGDEBUGGER_UI_DIR"),
        os.path.join(package_dir, "web", "dist"),
        os.path.abspath(os.path.join(package_dir, os.pardir, os.pardir, "frontend", "dist")),
    ]
    ui_folder_path = next((path for path in ui_folder_candidates if path and os.path.isdir(path)), None)
    if os.environ.get("AGDEBUGGER_BACKEND_SERVE_UI", "TRUE") == "TRUE":
        if ui_folder_path is None:
            print(
                "[WARN] AGDebugger UI build directory not found. "
                "Set AGDEBUGGER_UI_DIR or build the frontend if you want the web UI."
            )
        else:
            app.mount("/", StaticFiles(directory=ui_folder_path, html=True), name="ui")

    # load app and make backend
    loaded_gc = await load_app(module_str)
    backend = BackendRuntimeManager(loaded_gc, logger, message_history, state_cache)
    await backend.async_initialize()

    @api.get("/agents")
    async def get_agent_list() -> List[str]:
        if not backend.ready:
            print("Agents not ready yet...")
            return []
        return backend.agent_names

    @api.get("/getMessageQueue")
    async def get_messages():
        message_queue = [message_to_json(msg) for msg in backend.message_queue_list]
        return message_queue

    @api.get("/getSessionHistory")
    async def getSessionHistory():
        saved_sessions = backend.read_current_session_history()

        return {
            "current_session": backend.session_counter,
            "message_history": saved_sessions,
        }

    @api.get("/num_tasks")
    async def get_outstanding_tasks() -> int:
        return backend.unprocessed_messages_count

    @api.post("/drop")
    async def drop():
        if backend.unprocessed_messages_count == 0:
            return {"status": "ok"}

        backend.intervention_handler.drop = True
        await backend.process_next()
        return {"status": "ok"}

    @api.post("/step")
    async def step():
        if backend.unprocessed_messages_count == 0:
            return {"status": "ok"}
        await backend.process_next()
        return {"status": "ok"}

    @api.post("/start_loop")
    async def start_loop():
        backend.start_processing()
        return {"status": "ok"}

    @api.post("/stop_loop")
    async def stop_loop(force: bool = False):
        await backend.stop_processing(force=force)
        return {"status": "ok"}

    @api.get("/loop_status")
    async def loop_status() -> bool:
        return backend.is_processing

    @api.get("/message_types")
    async def message_types():
        return backend.message_info

    @api.get("/topics")
    async def topics() -> List[str]:
        return backend.all_topics

    @api.get("/state/{name}/get")
    async def get_config(name: str):
        try:
            config = await backend.get_agent_config(name)
            return config
        except Exception as e:
            print("Error getting state: ", e)
            return {"status": "error", "message": str(e)}

    @api.post("/publish")
    async def publish_message(message: PublishMessage):
        if message.body is None:
            return {"status": "error", "message": "Message body cannot be None"}

        new_message = deserialize(message.body)
        if new_message is None:
            return {"status": "error", "message": "Failed to deserialize message body"}

        if isinstance(new_message, _RPC_ONLY_GROUP_CHAT_MESSAGES):
            manager_topic = backend.groupchat._group_chat_manager_topic_type
            if message.topic != manager_topic:
                return {
                    "status": "error",
                    "message": (
                        f"{type(new_message).__name__} must be sent directly to the group chat manager "
                        f"'{manager_topic}', not published to topic '{message.topic}'."
                    ),
                }
            await backend.send_message(new_message, manager_topic)
            return {"status": "ok"}

        await backend.publish_message(new_message, message.topic)
        return {"status": "ok"}

    @api.post("/send")
    async def send_message(message: SendMessage):
        if message.body is None:
            return {"status": "error", "message": "Message body cannot be None"}
        try:
            new_message = deserialize(message.body)
            if new_message is None:
                return {"status": "error", "message": "Failed to deserialize message body"}
            await backend.send_message(new_message, message.recipient)
        except Exception as e:
            return {"status": "error", "message": str(e)}

        return {"status": "ok"}

    @api.post("/team_reset")
    async def team_reset():
        """Full team-level reset: clears model_context on all participants,
        resets the manager, and drains the output queue."""
        try:
            await backend.team_reset()
        except Exception as e:
            return {"status": "error", "message": str(e)}
        return {"status": "ok"}

    @api.post("/editQueue")
    async def edit_message_queue(edit_message: EditQueueMessage):
        print("Editing message at index ", edit_message.idx, "with new content: ", edit_message.body)

        if edit_message.body is None:
            return {"status": "error", "message": "Messgage body cannot be None"}

        try:
            new_message = deserialize(edit_message.body)
            if new_message is None:
                return {"status": "error", "message": "Failed to deserialize message body"}
            await backend.edit_message_queue(new_message, edit_message.idx)
        except Exception as e:
            return {"status": "error", "message": str(e)}

        return {"status": "ok"}

    @api.post("/editAndRevertHistoryMessage")
    async def edit_and_revert_message(edit_message: EditHistoryMessage):
        try:
            if edit_message.body is not None:
                new_message = deserialize(edit_message.body)
                if new_message is None:
                    return {"status": "error", "message": "Failed to deserialize message body"}
            else:
                new_message = None
            await backend.edit_and_revert_message(new_message, edit_message.timestamp)
        except Exception as e:
            return {"status": "error", "message": str(e)}

        return {"status": "ok"}

    @api.get("/logs")
    async def get_logs():
        return backend.log_handler.get_log_messages()

    @api.post("/save_to_file")
    async def save_to_file():
        await write_file_async("history.pickle", backend.intervention_handler.history)
        await write_file_async("cache.pickle", backend.agent_checkpoints)

        return {"status": "ok"}

    return app
