import asyncio
import json
import time
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn

app = FastAPI()

@app.post("/v1/chat/completions")
async def mock_chat_completions(request: Request):
    body = await request.json()
    stream = body.get("stream", False)

    if stream:
        async def fake_token_stream():
            # Simulate initial Time to First Token (TTFT) delay
            await asyncio.sleep(0.05)
            
            # Emit 10 simulated tokens
            for i in range(10):
                chunk = {
                    "id": "chatcmpl-mock",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": body.get("model", "mock-model"),
                    "choices": [{
                        "index": 0,
                        "delta": {"content": f" token_{i}"},
                        "finish_reason": None if i < 9 else "stop"
                    }]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0.01)  # Simulate inter-token latency
            yield "data: [DONE]\n\n"

        return StreamingResponse(fake_token_stream(), media_type="text/event-stream")

    # Non-streaming response fallback
    await asyncio.sleep(0.1)
    return {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model", "mock-model"),
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "This is a mocked response."},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 15, "completion_tokens": 10, "total_tokens": 25}
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)