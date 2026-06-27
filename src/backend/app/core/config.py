from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import URL


class Settings(BaseSettings):
    app_name: str = "ReviewScope"
    environment: str = "development"
    database_url: str | None = Field(default=None, alias="DATABASE_URL")
    sync_database_url: str | None = Field(default=None, alias="SYNC_DATABASE_URL")
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    secret_key: str = Field(alias="SECRET_KEY")
    postgres_user: str | None = Field(default=None, alias="POSTGRES_USER")
    postgres_password: str | None = Field(default=None, alias="POSTGRES_PASSWORD")
    postgres_db: str | None = Field(default=None, alias="POSTGRES_DB")
    postgres_host: str = Field(default="localhost", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5533, alias="POSTGRES_PORT")
    access_token_expire_minutes: int = 60 * 24
    upload_dir: Path = Path("/workspace/data/uploads")
    frontend_origin: str = "http://localhost:5173"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def async_database_url(self) -> str:
        if self._has_postgres_parts:
            return URL.create(
                "postgresql+asyncpg",
                username=self.postgres_user,
                password=self.postgres_password,
                host=self.postgres_host,
                port=self.postgres_port,
                database=self.postgres_db,
            ).render_as_string(hide_password=False)
        if not self.database_url:
            raise ValueError("DATABASE_URL or POSTGRES_HOST must be configured")
        return self.database_url

    @property
    def sync_sqlalchemy_database_url(self) -> str:
        if self._has_postgres_parts:
            return URL.create(
                "postgresql+psycopg",
                username=self.postgres_user,
                password=self.postgres_password,
                host=self.postgres_host,
                port=self.postgres_port,
                database=self.postgres_db,
            ).render_as_string(hide_password=False)
        if self.sync_database_url:
            return self.sync_database_url
        if not self.database_url:
            raise ValueError("SYNC_DATABASE_URL, DATABASE_URL, or POSTGRES_HOST must be configured")
        return self.database_url.replace("postgresql+asyncpg://", "postgresql+psycopg://")

    @property
    def _has_postgres_parts(self) -> bool:
        return bool(self.postgres_user and self.postgres_password is not None and self.postgres_db)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    return settings
