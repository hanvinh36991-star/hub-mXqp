from pydantic import BaseModel, ConfigDict
from datetime import datetime
from typing import Optional


class BaseSchema(BaseModel):
    """Base schema with common fields"""
    model_config = ConfigDict(from_attributes=True)


class TimestampMixin(BaseModel):
    """Mixin for timestamp fields"""
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class IDMixin(BaseModel):
    """Mixin for ID field"""
    id: Optional[int] = None