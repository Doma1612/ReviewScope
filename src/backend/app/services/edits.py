import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import EDIT_ACTIONS, ClusterEdit


def record_edit(
    db: AsyncSession,
    *,
    project_id: uuid.UUID,
    actor_id: uuid.UUID,
    action: str,
    **fields,
) -> ClusterEdit:
    # Stage the audit row in the mutation transaction; the caller commits.
    if action not in EDIT_ACTIONS:
        raise ValueError(f"Unknown edit action {action!r}; known: {EDIT_ACTIONS}")
    edit = ClusterEdit(project_id=project_id, actor_id=actor_id, action=action, **fields)
    db.add(edit)
    return edit
