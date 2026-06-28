from celery import Celery

from app.core.config import get_settings


settings = get_settings()

celery_app = Celery("reviewscope", broker=settings.redis_url, backend=settings.redis_url)

# Import the task modules so their @celery_app.task registrations run in the
# worker. Done after celery_app is defined so the app.* -> app.worker import in
# those modules resolves. ml_pipeline keeps reviewscope_ml as a lazy in-task
# import, so this stays light (no torch at worker startup).
from app import ml_pipeline as _ml_pipeline  # noqa: E402,F401
from app import tasks as _tasks  # noqa: E402,F401
