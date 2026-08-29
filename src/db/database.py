import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from src.config import DATABASE_URL

# --- SYNCHRONOUS DATABASE SETUP (for backward compatibility) ---
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False}  # Cần cho SQLite trong FastAPI/Streamlit
    )
else:
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        pool_size=30,       # Giảm từ 50 xuống 30 để an toàn với Supabase Free Tier (max 60)
        max_overflow=10,    # Giảm từ 30 xuống 10
        pool_timeout=30     # Đợi tối đa 30s trước khi báo lỗi
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# --- ASYNCHRONOUS DATABASE SETUP (for high performance API) ---
ASYNC_DATABASE_URL = DATABASE_URL
if DATABASE_URL.startswith("sqlite:///"):
    ASYNC_DATABASE_URL = DATABASE_URL.replace("sqlite:///", "sqlite+aiosqlite:///")
elif DATABASE_URL.startswith("postgresql://"):
    ASYNC_DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
elif DATABASE_URL.startswith("postgres://"):
    ASYNC_DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://")

if ASYNC_DATABASE_URL.startswith("sqlite"):
    async_engine = create_async_engine(
        ASYNC_DATABASE_URL,
        connect_args={"check_same_thread": False}
    )
else:
    async_engine = create_async_engine(
        ASYNC_DATABASE_URL,
        pool_pre_ping=True,
        pool_size=30,       # Cấu hình an toàn tương tự để tránh tràn connection Supabase
        max_overflow=10,
        pool_timeout=30,
        connect_args={
            "statement_cache_size": 0,
            "prepared_statement_cache_size": 0
        }
    )

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def get_db_async():
    async with AsyncSessionLocal() as db:
        yield db

