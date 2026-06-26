from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ModelEntry(Base):
    __tablename__ = "model_registry"
    __table_args__ = (UniqueConstraint("name", "version", name="uq_name_version"),)

    # ── Existing columns (unchanged) ──────────────────────────────────────────
    id:          Mapped[str]      = mapped_column(String(36),  primary_key=True)
    name:        Mapped[str]      = mapped_column(String(128), nullable=False)
    version:     Mapped[str]      = mapped_column(String(32),  nullable=False)
    task:        Mapped[str]      = mapped_column(String(32),  nullable=False)
    file_path:   Mapped[str]      = mapped_column(Text,        nullable=False)
    config_json: Mapped[str]      = mapped_column(Text,        nullable=False)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    # ── New columns (all nullable — existing rows get NULL, never breaks) ─────
    slug: Mapped[str | None] = mapped_column(
        String(128), nullable=True, index=True,
        comment="URL-friendly identifier, e.g. 'yolov8n-detection-1.0.0'"
    )
    tag: Mapped[str | None] = mapped_column(
        String(16), nullable=True, default="stable",
        comment="Lifecycle tag: stable | experimental | deprecated"
    )
    is_latest: Mapped[bool | None] = mapped_column(
        Boolean, nullable=True, default=True,
        comment="True if this is the latest version in its model family"
    )
    parent_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("model_registry.id", name="fk_model_parent", ondelete="SET NULL"),
        nullable=True,
        comment="Parent model ID for fork/finetune lineage"
    )
    changelog:   Mapped[str | None] = mapped_column(Text,        nullable=True)
    description: Mapped[str | None] = mapped_column(Text,        nullable=True)
    ports_json:  Mapped[str | None] = mapped_column(
        Text, nullable=True,
        comment="JSON: {inputs:[{name,tensor_name,type,shape,dtype,dynamic_axes,desc}], outputs:[...]}"
    )
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    size_bytes:   Mapped[int | None]      = mapped_column(Integer, nullable=True)
    author:       Mapped[str | None]      = mapped_column(String(128), nullable=True)
    license:      Mapped[str | None]      = mapped_column(String(64),  nullable=True)
    extra_meta:   Mapped[str | None]      = mapped_column(
        Text, nullable=True,
        comment="JSON for benchmark results, training info, dataset details, etc."
    )
