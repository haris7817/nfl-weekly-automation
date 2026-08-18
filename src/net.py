"""TLS trust-store handling for local Windows machines.

Consumer antivirus products (AVG, Avast, Kaspersky, ESET) and corporate
proxies terminate HTTPS and re-sign it with a locally generated root CA.
That CA lives in the Windows certificate store, so browsers and ``curl``
work, but Python's ``requests`` uses the bundled ``certifi`` list and fails
with ``CERTIFICATE_VERIFY_FAILED``.

``truststore`` routes verification through the operating system's own
verifier, which fixes this without disabling certificate checking. It is a
no-op on the Linux GitHub Actions runner, where certifi already works.

This never turns verification off. If the trust store cannot be used the
run continues with the default behaviour and the real error surfaces.
"""

from __future__ import annotations

import os
import sys

from .logging_utils import get_logger

log = get_logger("net")

_INJECTED = False


def _enabled() -> bool:
    raw = os.getenv("USE_SYSTEM_CERT_STORE")
    if raw is not None and raw.strip() != "":
        return raw.strip().lower() in {"1", "true", "yes", "y", "on"}
    # Default on for Windows (where TLS-inspecting AV is common), off elsewhere.
    return sys.platform == "win32"


def enable_system_trust_store() -> bool:
    """Route TLS verification through the OS trust store. Returns success."""
    global _INJECTED

    if _INJECTED:
        return True
    if not _enabled():
        return False

    try:
        import truststore
    except ImportError:
        log.debug("truststore is not installed; using the certifi bundle.")
        return False

    try:
        truststore.inject_into_ssl()
    except Exception as exc:  # pragma: no cover - platform specific
        log.warning(
            "Could not enable the system certificate store (%s: %s). "
            "Continuing with the default certifi bundle.",
            type(exc).__name__,
            exc,
        )
        return False

    _INJECTED = True
    log.debug("TLS verification is using the operating system trust store.")
    return True
