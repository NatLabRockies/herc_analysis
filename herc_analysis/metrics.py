"""L3 -- the long-format metrics table.

Will hold ``MetricSet`` and ``MetricSpec``: the canonical long-format metric
output keyed by ``(entity, metric, resolution, period)`` shared by ``Scenario``
and ``Comparison``, with ``to_nested()`` for back-compat.

Populated in Phase 3 of the refactor (see the refactoring plan, layer L3).
"""

from __future__ import annotations
