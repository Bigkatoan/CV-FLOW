from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase
from app.config import settings

engine = create_async_engine(settings.database_url, echo=False)

AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session


async def create_tables():
    # Import all model modules so Base.metadata is fully populated before create_all.
    import app.models.model_registry  # noqa: F401
    import app.models.datahub         # noqa: F401
    import app.models.compiled_node   # noqa: F401
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

