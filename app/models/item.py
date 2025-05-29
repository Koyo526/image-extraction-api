# アイテムモデル
from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship
from app.db import Base
class Item(Base):
    __tablename__ = "items"

    id = Column(Integer, primary_key=True, index=True)
    user = Column(Integer, ForeignKey("users.id"), nullable=False)
    img_path = Column(String, nullable=False)
    filename = Column(String, nullable=True)
