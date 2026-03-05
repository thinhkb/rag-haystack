from __future__ import annotations
from typing import Any, Callable

CONF_ORDER = {"public": 0, "internal": 1, "restricted": 2}

def _conf_leq(user_level: str, doc_level: str) -> bool:
    ul = CONF_ORDER.get((user_level or "internal").lower(), 1)
    dl = CONF_ORDER.get((doc_level or "restricted").lower(), 2)
    return dl <= ul


def build_access_predicate(
    *,
    user_context: dict[str, Any],
    filters: dict[str, Any] | None = None,
) -> Callable[[dict[str, Any]], bool]:

    filters = filters or {}

    user_department = user_context.get("department")
    user_conf_level = user_context.get("confidentiality_level")
    user_role = user_context.get("role")

    doc_ids = filters.get("doc_ids")

    if isinstance(doc_ids, list):
        doc_ids = [str(x) for x in doc_ids]
    else:
        doc_ids = None

    def predicate(meta: dict[str, Any]) -> bool:

        # filter by doc_id (precision improvement)
        if doc_ids is not None:
            if str(meta.get("doc_id")) not in doc_ids:
                return False

        # department check
        doc_department = meta.get("department")

        if doc_department and user_department:
            if doc_department != user_department:
                return False

        # confidentiality check
        doc_conf = meta.get("confidentiality_level")

        if doc_conf and user_conf_level:
            if doc_conf == "restricted" and user_conf_level != "restricted":
                return False

        # role-based control
        allowed_roles = meta.get("allowed_roles")

        if allowed_roles:
            if user_role not in allowed_roles:
                return False

        return True

    return predicate