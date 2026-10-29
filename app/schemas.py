from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    role: str
    content: str

    model_config = ConfigDict(str_strip_whitespace=True)

    def normalized_role(self) -> str:
        return self.role.strip().lower()


class ChatRequest(BaseModel):
    """Body every application sends to the middleware."""

    model: str = Field(default="auto", description='Use "auto" or a model id such as claude-sonnet-4-5.', examples=["auto"])
    type: str = Field(
        default="auto",
        description='Provider. Use "claude", "chatgpt", "grok", "groq", "nvidia", "ollama", "gemini", or "auto".',
        examples=["claude"],
    )
    query: str = Field(default="", description="The question to ask the model.", examples=["Say hello from application one"])
    context: str = Field(
        default="",
        description="Extra instructions or facts the model should use.",
        examples=["You are a helpful assistant."],
    )
    messages: list[ChatMessage] = Field(default_factory=list, description="Earlier conversation turns.")
    app: str = Field(default="default", description="Name of the application calling the middleware.", examples=["application-one"])
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=1024, ge=1, le=32000)
    top_p: float | None = Field(default=None, ge=0, le=1)
    stream: bool = Field(
        default=False,
        description='Set true to return a text/event-stream. Events are start, delta, done, then data: [DONE].',
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "examples": [
                {
                    "app": "application-one",
                    "model": "auto",
                    "type": "claude",
                    "query": "Say hello from application one",
                    "context": "You are a helpful assistant.",
                },
                {
                    "app": "chatgpt-app",
                    "model": "auto",
                    "type": "chatgpt",
                    "query": "Say hello from the ChatGPT application",
                    "context": "You are a helpful assistant.",
                },
            ]
        },
    )


class Usage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class ChatResponse(BaseModel):
    id: str
    created: int
    app: str
    type: str
    provider: str
    model: str
    requested_type: str
    requested_model: str
    query: str
    context: str
    answer: str
    finish_reason: str
    mock: bool
    usage: Usage
    metadata: dict[str, Any] = Field(default_factory=dict)


class ErrorDetail(BaseModel):
    message: str
    type: str
    provider: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


class OpenAIChoiceMessage(BaseModel):
    role: str
    content: str


class OpenAIChoice(BaseModel):
    index: int
    message: OpenAIChoiceMessage
    finish_reason: str


class OpenAIUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OpenAIChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[OpenAIChoice]
    usage: OpenAIUsage


class GeneratedImage(BaseModel):
    url: str | None = None
    b64_json: str | None = None


class ImageRequest(BaseModel):
    """Body every application sends to generate an image."""

    prompt: str = Field(default="", description="What the image should show.")
    model: str = Field(default="auto", description='Use "auto" or an image model id.')
    type: str = Field(
        default="auto",
        description='Provider. Use "chatgpt", "grok", "nvidia", "ollama", "gemini", "horde", or "auto".',
        examples=["grok"],
    )
    size: str = Field(default="1024x1024", description='Pixel size, such as "1024x1024".')
    n: int = Field(default=1, ge=1, le=4, description="How many images to create. NVIDIA accepts 1.")
    app: str = Field(default="image-app", description="Name of the application calling the middleware.")
    metadata: dict[str, Any] = Field(default_factory=dict)


class ImageResponse(BaseModel):
    id: str
    created: int
    app: str
    type: str
    provider: str
    model: str
    requested_type: str
    requested_model: str
    prompt: str
    size: str
    mock: bool
    images: list[GeneratedImage]
    metadata: dict[str, Any] = Field(default_factory=dict)


class VideoRequest(BaseModel):
    """Body every application sends to start a video."""

    prompt: str = Field(default="", description="What the video should show.")
    model: str = Field(default="auto", description='Use "auto" or a video model id.')
    type: str = Field(
        default="free",
        description='Provider. Use "free", "nvidia", "higgsfield", or "auto".',
        examples=["free"],
    )
    duration: int = Field(default=5, ge=4, le=30, description="Video length in seconds.")
    resolution: str = Field(default="720p", description='One of "480p", "720p", or "1080p".')
    aspect_ratio: str = Field(default="16:9", description='One of "16:9", "4:3", "1:1", "3:4", "9:16", or "21:9".')
    output_format: str = Field(default="mp4", description='Use "mp4" or "mov".')
    generate_audio: bool = Field(default=True, description="Ask Seedance to include audio.")
    app: str = Field(default="video-app", description="Name of the application calling the middleware.")
    metadata: dict[str, Any] = Field(default_factory=dict)


class VideoJobResponse(BaseModel):
    id: str
    created: int
    app: str
    type: str
    provider: str
    model: str
    prompt: str
    status: str
    video_url: str | None = None
    error: str | None = None
    mock: bool
    metadata: dict[str, Any] = Field(default_factory=dict)


class OpenAIChatRequest(BaseModel):
    """OpenAI-style body, so a ChatGPT application can point at this middleware."""

    model: str = "auto"
    messages: list[ChatMessage]
    type: str | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, ge=1, le=32000)
    max_completion_tokens: int | None = Field(default=None, ge=1, le=32000)
    top_p: float | None = Field(default=None, ge=0, le=1)
    stream: bool = Field(
        default=False,
        description="Set true to return OpenAI chat completion chunks as a text/event-stream.",
    )
    user: str | None = None

    model_config = ConfigDict(extra="ignore")
