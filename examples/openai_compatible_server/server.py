import asyncio
import json
import time
from typing import List, Optional
import sys

from fastapi import FastAPI
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from moa_llm import create_moa_from_config

app = FastAPI(title="MoA-powered Chat API")


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "moa-model"
    messages: List[Message]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7  # This is set up in the moa_config.yaml file
    stream: Optional[bool] = False

if len(sys.argv) > 1:
    config_path = sys.argv[1]
else:
    config_path = "examples/openai_compatible_server/moa_config.yaml"

moa = create_moa_from_config(config_path, is_file_path=True)

async def stream_response(content: str):
    tokens = content.split()
    for token in tokens:
        chunk = {
            "choices": [{"delta": {"content": token + " "}}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0.1)
    yield "data: [DONE]\n\n"


@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]

    result = await moa.process(messages)

    if request.stream:
        return StreamingResponse(stream_response(result["content"]), media_type="text/event-stream")

    return {
        "id": "moa-1",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{"message": {"role": "assistant", "content": result["content"]}}],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)