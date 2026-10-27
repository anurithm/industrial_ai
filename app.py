"""
Application entry point.
Run directly for local development or import for deployment.
"""
import logging
import subprocess
import sys
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def check_models():
    """Check if trained model artifacts exist."""
    from src.config import MODELS_DIR
    return (MODELS_DIR / "best_model.joblib").exists()


if __name__ == "__main__":
    import uvicorn
    from src.config import API_HOST, API_PORT
    from src.api import app

    if not check_models():
        logger.warning(
            "Model not trained. Run: python -m src.train_pipeline\n"
            "Starting API anyway — endpoints will return 503 until training is complete."
        )

    logger.info(f"Starting IndustrialAI API at http://{API_HOST}:{API_PORT}")
    uvicorn.run(app, host=API_HOST, port=API_PORT, log_level="info")
