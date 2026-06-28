"""Helper for writing the cluster-edit audit log (WP B1).

Other WPs call ``record_edit`` inside their mutation transactions; it only stages
the row (``db.add``) — the caller is responsible for committing so the edit shares
the transaction with the change it describes.
"""
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
    """Stage a ``ClusterEdit`` row. Accepts the optional subject columns
    (``cluster_id``, ``target_cluster_id``, ``document_id``, ``new_label``,
    ``note``, ``payload``) as keyword arguments."""
    if action not in EDIT_ACTIONS:
        raise ValueError(f"Unknown edit action {action!r}; known: {EDIT_ACTIONS}")
    edit = ClusterEdit(project_id=project_id, actor_id=actor_id, action=action, **fields)
    db.add(edit)
    return edit
