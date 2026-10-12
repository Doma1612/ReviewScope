from celery import Celery

from app.core.config import get_settings


settings = get_settings()

celery_app = Celery("reviewscope", broker=settings.redis_url, backend=settings.redis_url)

# One task per worker process: a pipeline run pins its GPU through
# CUDA_VISIBLE_DEVICES, which CUDA reads only once per process. A recycled
# process would keep the previous run's cards (or CUDA state) for the next run.
celery_app.conf.worker_max_tasks_per_child = 1

# Register tasks after defining celery_app to avoid circular imports.
from app import ml_pipeline as _ml_pipeline  # noqa: E402,F401
from app import tasks as _tasks  # noqa: E402,F401
