from datetime import datetime, timezone
from sqlalchemy import String, Text, DateTime
from sqlalchemy.orm import Mapped, mapped_column
from app.database import Base


def _utcnow():
    return datetime.now(timezone.utc)


class CompiledNode(Base):
    __tablename__ = "compiled_nodes"

    source_hash: Mapped[str] = mapped_column(String(64), primary_key=True)  # SHA256 hex
    so_path: Mapped[str] = mapped_column(Text, nullable=False)
    compile_flags: Mapped[str | None] = mapped_column(Text)                 # JSON array string
    stderr_output: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), nullable=False)          # ok | error
    compiled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
