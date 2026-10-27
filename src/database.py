"""
SQLite database for prediction history storage.
"""
import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict
import json

from sqlalchemy import create_engine, Column, String, Float, Boolean, DateTime, Integer, Text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.config import DATABASE_URL

logger = logging.getLogger(__name__)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class PredictionRecord(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    model_name = Column(String)
    model_version = Column(String)

    # Sensor inputs (JSON)
    sensor_inputs = Column(Text)

    # Prediction output
    predicted_failure = Column(Boolean)
    failure_probability = Column(Float)
    decision_threshold = Column(Float)
    risk_level = Column(String)

    # Anomaly
    is_anomaly = Column(Boolean, nullable=True)
    anomaly_score = Column(Float, nullable=True)

    # Metadata
    input_warnings = Column(Text, nullable=True)
    inference_latency_ms = Column(Float, nullable=True)


def init_db():
    """Create database tables."""
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialised")


def get_db():
    """FastAPI dependency: yield a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def save_prediction(db: Session, record: Dict) -> PredictionRecord:
    """Save a prediction result to the database."""
    db_record = PredictionRecord(
        request_id=record.get("request_id", ""),
        model_name=record.get("model_name", ""),
        model_version=record.get("model_version", ""),
        sensor_inputs=json.dumps(record.get("sensor_inputs", {})),
        predicted_failure=record.get("predicted_failure", False),
        failure_probability=record.get("failure_probability", 0.0),
        decision_threshold=record.get("decision_threshold", 0.5),
        risk_level=record.get("risk_level", "UNKNOWN"),
        is_anomaly=record.get("is_anomaly"),
        anomaly_score=record.get("anomaly_score"),
        input_warnings=json.dumps(record.get("input_warnings", [])),
        inference_latency_ms=record.get("inference_latency_ms"),
    )
    db.add(db_record)
    db.commit()
    db.refresh(db_record)
    return db_record


def get_history(db: Session, limit: int = 50) -> List[Dict]:
    """Get recent prediction history."""
    records = (
        db.query(PredictionRecord)
        .order_by(PredictionRecord.timestamp.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": r.id,
            "request_id": r.request_id,
            "timestamp": r.timestamp.isoformat() if r.timestamp else None,
            "model_name": r.model_name,
            "model_version": r.model_version,
            "sensor_inputs": json.loads(r.sensor_inputs or "{}"),
            "predicted_failure": r.predicted_failure,
            "failure_probability": r.failure_probability,
            "risk_level": r.risk_level,
            "is_anomaly": r.is_anomaly,
            "anomaly_score": r.anomaly_score,
            "input_warnings": json.loads(r.input_warnings or "[]"),
        }
        for r in records
    ]
