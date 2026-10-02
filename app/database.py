"""
Week 4: Database design & implementation.

Uses SQLite for local dev (zero setup). To move to Postgres for deployment
(Week 9-10), just change DATABASE_URL in .env to something like:
    postgresql://user:password@localhost:5432/eval_dashboard
No other code changes needed — SQLAlchemy handles both.
"""
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./eval_dashboard.db")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI dependency — gives each request its own DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
