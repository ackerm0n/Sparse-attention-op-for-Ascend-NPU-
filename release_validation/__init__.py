"""Remote-executable release validation for the TriangleMix wheel.

The modules in this directory are deliberately outside the wheel.  They
validate an already installed artifact and never add source directories to
``sys.path``.
"""

from __future__ import annotations

__all__: list[str] = []
