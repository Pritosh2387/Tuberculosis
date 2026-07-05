"""
Small runtime compatibility shims for the ``torch`` C-extension surface.

These helpers keep LungCare AI importable across the supported ``torch``
range (``>=2.1``) even when newer transitive dependencies (notably
``transformers``, pulled in by ``torchvision`` / ``torchmetrics``) expect
public APIs that only appeared in later ``torch`` releases.

Call :func:`ensure_pytree_compat` once, as early as possible, before any
import that may load ``transformers``.  The function is idempotent and a
no-op on ``torch`` builds that already expose the modern API.
"""

from __future__ import annotations


def ensure_pytree_compat() -> bool:
    """
    Provide ``torch.utils._pytree.register_pytree_node`` on older torch builds.

    ``transformers>=4.4x`` calls the public ``register_pytree_node`` at import
    time, passing keyword-only arguments such as ``serialized_type_name`` that
    only exist on ``torch>=2.2``.  ``torch<2.2`` exposes the private
    ``_register_pytree_node`` with a narrower signature.

    We install a thin wrapper that forwards the three core callables
    (type, flatten_fn, unflatten_fn) and silently drops the serialization-only
    keyword arguments the older private function cannot accept.  Runtime
    flatten/unflatten behaviour is unaffected; only ``torch.export`` string
    serialization (unused by this project and its tests) is skipped.

    Returns:
        ``True`` if a shim was installed, ``False`` if none was needed
        (either the public API already exists or torch is unavailable).
    """
    try:
        import torch.utils._pytree as _pytree
    except Exception:  # pragma: no cover - torch always present in practice
        return False

    if hasattr(_pytree, "register_pytree_node"):
        return False

    private = getattr(_pytree, "_register_pytree_node", None)
    if private is None:  # pragma: no cover - unexpected torch layout
        return False

    def register_pytree_node(cls, flatten_fn, unflatten_fn, *args, **kwargs):  # type: ignore[no-untyped-def]
        """Compat wrapper: forward core callables, drop newer-only kwargs."""
        return private(cls, flatten_fn, unflatten_fn)

    _pytree.register_pytree_node = register_pytree_node  # type: ignore[attr-defined]
    return True
