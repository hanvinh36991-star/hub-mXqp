from typing import TypeVar, Generic, Type, Optional, List, Dict, Any
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.models.database import Base

ModelType = TypeVar("ModelType", bound=Base)
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)


class BaseRepository(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    """Base repository with CRUD operations"""
    
    def __init__(self, model: Type[ModelType], db: Session):
        self.model = model
        self.db = db
    
    def get(self, id: int) -> Optional[ModelType]:
        return self.db.query(self.model).filter(self.model.id == id).first()
    
    def get_multi(
        self, 
        skip: int = 0, 
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[ModelType]:
        query = self.db.query(self.model)
        
        if filters:
            for key, value in filters.items():
                if hasattr(self.model, key):
                    if isinstance(value, list):
                        query = query.filter(getattr(self.model, key).in_(value))
                    else:
                        query = query.filter(getattr(self.model, key) == value)
        
        return query.offset(skip).limit(limit).all()
    
    def create(self, obj_in: CreateSchemaType) -> ModelType:
        obj_in_data = obj_in.dict() if hasattr(obj_in, 'dict') else obj_in
        db_obj = self.model(**obj_in_data)
        self.db.add(db_obj)
        self.db.commit()
        self.db.refresh(db_obj)
        return db_obj
    
    def update(self, db_obj: ModelType, obj_in: UpdateSchemaType) -> ModelType:
        obj_data = obj_in.dict(exclude_unset=True) if hasattr(obj_in, 'dict') else obj_in
        
        for field, value in obj_data.items():
            if hasattr(db_obj, field):
                setattr(db_obj, field, value)
        
        self.db.add(db_obj)
        self.db.commit()
        self.db.refresh(db_obj)
        return db_obj
    
    def delete(self, id: int) -> Optional[ModelType]:
        obj = self.get(id)
        if obj:
            self.db.delete(obj)
            self.db.commit()
        return obj
    
    def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        query = self.db.query(self.model)
        
        if filters:
            for key, value in filters.items():
                if hasattr(self.model, key):
                    if isinstance(value, list):
                        query = query.filter(getattr(self.model, key).in_(value))
                    else:
                        query = query.filter(getattr(self.model, key) == value)
        
        return query.count()