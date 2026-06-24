from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "sqlite+aiosqlite:///./cv_flow.db"
    storage_path: Path = Path("./storage")
    engine_ws_port: int = 8765
    engine_python: str = "python"
    cors_origins: str = "http://localhost:5173"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]

    @property
    def models_dir(self) -> Path:
        return self.storage_path / "models"

    @property
    def compiled_dir(self) -> Path:
        return self.storage_path / "compiled"

    @property
    def pipelines_tmp_dir(self) -> Path:
        return self.storage_path / "tmp"


settings = Settings()
