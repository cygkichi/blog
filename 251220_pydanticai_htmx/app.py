import html
import uuid

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic_ai import (
    Agent,
    ModelMessage,
    RunUsage,
    UsageLimits,
)
import logfire

logfire.configure()

app = FastAPI()
usage: RunUsage = RunUsage()
usage_limits = UsageLimits(request_limit=10)

agent = Agent("google-gla:gemini-2.5-flash")
message_history: list[ModelMessage] = []

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def chat_page(request: Request) -> HTMLResponse:
    """Serve the main chat interface"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/chat")
async def chat(message: str = Form(...)) -> StreamingResponse:
    """Handle chat messages and return SSE stream"""

    async def event_generator():
        global message_history

        escaped_message = html.escape(message)
        user_html = f'<div class="message user">{escaped_message}</div>'
        yield f"data: {user_html}\n\n"

        logfire.info(
            "User message received", message=message, message_length=len(message)
        )

        result = await agent.run(
            message,
            message_history=message_history,
            usage=usage,
            usage_limits=usage_limits,
        )
        message_history = result.all_messages()
        response_text = str(result.output) if hasattr(result, "output") else str(result)

        escaped_response = html.escape(response_text)
        message_id = str(uuid.uuid4())

        logfire.info(
            "Assistant response generated",
            message_id=message_id,
            response_length=len(response_text),
            user_message=message,
            message_history=message_history,
        )

        # 3. AI応答を送信
        assistant_html = f'<div class="message assistant">{escaped_response}</div>'
        yield f"data: {assistant_html}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
