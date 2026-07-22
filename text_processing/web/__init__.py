"""Web layer for text_processing: schemas, pipeline cache, async jobs, router.

The HTTP concerns (Pydantic schemas, the bounded pipeline cache, the async
job store, and the FastAPI endpoints) each live in their own focused module.

The ``router`` object is intentionally NOT re-exported here: doing so would
shadow the ``text_processing.web.router`` submodule with the ``APIRouter``
instance. Import it from the submodule instead::

    from text_processing.web.router import router
"""

from __future__ import annotations
