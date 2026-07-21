"""Web layer for text_processing: schemas, pipeline cache, async jobs, router.

Split out of the former root-level ``text_processing_routes.py`` so the
HTTP concerns (Pydantic schemas, the bounded pipeline cache, the async job
store, and the FastAPI endpoints) each live in their own focused module.
``text_processing_routes`` remains as a thin shim re-exporting this layer.

The ``router`` object is intentionally NOT re-exported here: doing so would
shadow the ``text_processing.web.router`` submodule with the ``APIRouter``
instance. Import it from the submodule instead::

    from text_processing.web.router import router
"""

from __future__ import annotations
