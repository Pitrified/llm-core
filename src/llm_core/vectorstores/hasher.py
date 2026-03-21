"""SHA-256 document ID generation for deduplication.

Produces deterministic, reproducible IDs from document content and metadata.
Same content + metadata always produces the same ID.

Example:
    ::

        doc_id = document_id("Hello world", {"entity_type": "greeting"})
"""

import hashlib
import json


def document_id(content: str, metadata: dict) -> str:
    """SHA-256 hash of content + sorted metadata as a hex document ID.

    Args:
        content: The document's page content.
        metadata: The document's metadata dictionary.

    Returns:
        64-character lowercase hex string.
    """
    h = hashlib.sha256()
    h.update(content.encode("utf-8"))
    h.update(json.dumps(metadata, sort_keys=True).encode("utf-8"))
    return h.hexdigest()
