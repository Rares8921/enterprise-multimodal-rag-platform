"""
Database models
"""
from enum import Enum
from sqlalchemy import Index, Boolean, Column, DateTime, Enum as SQLEnum, Integer, JSON, String, Text, UniqueConstraint, text
from sqlalchemy.orm import declarative_base
from datetime import datetime
import uuid

Base = declarative_base()


class DocumentStatus(str, Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    OCR_COMPLETE = "ocr_complete"
    LAYOUT_PARSED = "layout_parsed"
    EMBEDDED = "embedded"
    INDEXED = "indexed"
    COMPLETED = "completed"
    FAILED = "failed"


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("tenant_id", "content_hash", name="uq_tenant_content"),
        Index("idx_documents_not_deleted", "is_deleted"),
        Index("idx_documents_status_tenant", "status", "tenant_id"),
    )

    doc_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(String, nullable=False, index=True)
    doc_type = Column(String, nullable=False)
    filename = Column(String, nullable=False)
    bucket_name = Column(String, nullable=False)
    object_name = Column(String, nullable=False)
    content_hash = Column(String, nullable=False, index=True)
    file_size = Column(Integer)
    status = Column(
        SQLEnum(
            DocumentStatus,
            name="document_status",
            native_enum=True,
            validate_strings=True,
        ),
        nullable=False,
        default=DocumentStatus.UPLOADED,
        server_default=text("'UPLOADED'"),
        index=True,
    )

    retry_count = Column(Integer, nullable=False, default=0)
    last_error = Column(Text, nullable=True)

    version = Column(Integer, nullable=False, default=1)
    embedding_model = Column(String, nullable=True)
    ocr_engine = Column(String, nullable=True)

    is_deleted = Column(Boolean, nullable=False, default=False)
    deleted_at = Column(DateTime(timezone=True), nullable=True)

    uploaded_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    ocr_completed_at = Column(DateTime(timezone=True), nullable=True)
    embedding_completed_at = Column(DateTime(timezone=True), nullable=True)
    indexed_at = Column(DateTime(timezone=True), nullable=True)
    processed_at = Column(DateTime(timezone=True), nullable=True)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Strictly structured metadata - do not dump arbitrary data
    document_metadata = Column(JSON, nullable=True)
    processing_metadata = Column(JSON, nullable=True)
    ocr_metadata = Column(JSON, nullable=True)
    layout_metadata = Column(JSON, nullable=True)
