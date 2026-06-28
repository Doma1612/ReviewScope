from fastapi import APIRouter

from app.core.config import get_settings
from app.schemas import ModelsRead


router = APIRouter()


@router.get("/models", response_model=ModelsRead)
def models() -> ModelsRead:
    """Available embedding and LLM models (app-spec System endpoint).

    In simulated mode the names are static; otherwise they come from the frozen
    pipeline spec the seam runs (``reviewscope_ml.app.app_default_spec``).
    """
    settings = get_settings()
    if settings.simulate_ml:
        return ModelsRead(
            embedding_model="simulated",
            label_model="simulated",
            variant="simulated",
            simulated=True,
        )
    from reviewscope_ml.app import APP_DEFAULT_VARIANT, app_default_spec

    spec = app_default_spec()
    return ModelsRead(
        embedding_model=spec.embedding_model,
        label_model=spec.label_model,
        variant=APP_DEFAULT_VARIANT,
        simulated=False,
    )
