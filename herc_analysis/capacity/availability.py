"""L4 -- availability providers.

Small provider classes (``FixedAvailability``, ``FromRunAvailability``,
``BatteryAvailability``) sharing one ``attach(df, name)`` convention that adds a
``{name}__availability`` channel. Populated in Phase 4 of the refactor.
"""

from __future__ import annotations
