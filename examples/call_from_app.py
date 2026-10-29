"""Call the middleware from any Python application."""

import json
import os
import urllib.request


MIDDLEWARE_URL = os.environ.get("MIDDLEWARE_URL", "http://127.0.0.1:8080")


def ask(app: str, provider_type: str, query: str, context: str = "", model: str = "auto") -> dict:
    body = json.dumps(
        {
            "app": app,
            "model": model,
            "type": provider_type,
            "query": query,
            "context": context,
        }
    ).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("MIDDLEWARE_API_KEY", "")
    if api_key:
        headers["X-API-Key"] = api_key
    request = urllib.request.Request(
        MIDDLEWARE_URL.rstrip("/") + "/api/v1/chat",
        data=body,
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        return json.load(response)


if __name__ == "__main__":
    claude = ask(
        app="application-one",
        provider_type="claude",
        query="Say hello from application one",
        context="You are a helpful assistant.",
    )
    chatgpt = ask(
        app="chatgpt-app",
        provider_type="chatgpt",
        query="Say hello from the ChatGPT application",
        context="You are a helpful assistant.",
    )
    print(claude["answer"])
    print(chatgpt["answer"])
