from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    middleware_api_key: str = ""
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    xai_api_key: str = ""
    nvidia_api_key: str = ""
    groq_api_key: str = ""
    gemini_api_key: str = ""
    ollama_api_key: str = ""
    anthropic_base_url: str = "https://api.anthropic.com"
    openai_base_url: str = "https://api.openai.com/v1"
    xai_base_url: str = "https://api.x.ai/v1"
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    ollama_base_url: str = "https://ollama.com/v1"
    claude_default_model: str = "claude-sonnet-4-5"
    openai_default_model: str = "gpt-4o-mini"
    grok_default_model: str = "grok-4.7"
    nvidia_default_model: str = "nvidia/nemotron-3.5-lightning-30b-a3b"
    groq_default_model: str = "llama-3.3-70b-versatile"
    gemini_default_model: str = "gemini-3.5-flash-lite"
    gemini_image_model: str = "gemini-nano-banana-2.1"
    aihorde_api_key: str = ""
    aihorde_base_url: str = "https://aihorde.net/api/v2"
    aihorde_image_model: str = "stable_diffusion"
    aihorde_wait_seconds: float = 180
    ollama_default_model: str = "gemma4:31b"
    default_provider: str = "claude"
    image_default_provider: str = "grok"
    openai_image_model: str = "gpt-image-2.5-flare"
    grok_image_model: str = "grok-imagine-image-2.0"
    nvidia_image_model: str = "black-forest-labs/flux.1-dev"
    nvidia_image_url: str = "https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-dev"
    nvidia_video_model: str = "nvidia/cosmos3-nano"
    nvidia_video_url: str = "https://ai.api.nvidia.com/v1/cosmos/nvidia/cosmos3-nano"
    nvidia_video_status_url: str = "https://api.nvcf.nvidia.com/v2/nvcf/pexec/status"
    nvidia_video_wait_seconds: float = 900
    free_video_model: str = "Lightricks/ltx-video-distilled"
    free_video_url: str = "https://lightricks-ltx-video-distilled.hf.space"
    huggingface_token: str = ""
    hf_credentials: str = ""
    hf_base_url: str = "https://api.higgsfield.ai"
    hf_video_model: str = "bytedance/seedance-2.5/text-to-video"
    ollama_image_model: str = "x/z-image-turbo"
    llm_mode: str = "live"
    host: str = "127.0.0.1"
    port: int = 8080
    cors_origins: str = "*"
    request_timeout_seconds: float = 120

    @property
    def mock_mode(self) -> bool:
        return self.llm_mode.strip().lower() == "mock"

    @property
    def auth_required(self) -> bool:
        return bool(self.middleware_api_key.strip())


def get_settings() -> Settings:
    return Settings()
