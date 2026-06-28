from collections.abc import AsyncGenerator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings


settings = get_settings()
engine = create_async_engine(settings.async_database_url, pool_pre_ping=True, poolclass=NullPool)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)

# Synchronous engine/session for the Celery worker. The real ML run is a long,
# blocking call that emits progress between stages, so the worker uses a plain
# sync session (psycopg) rather than asyncio — the progress sink can commit
# mid-run and the polling API sees each step advance.
sync_engine = create_engine(settings.sync_sqlalchemy_database_url, pool_pre_ping=True, poolclass=NullPool)
SyncSessionLocal = sessionmaker(sync_engine, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session
