"""L5 -- the single cross-scenario comparison engine.

Will hold ``Comparison``, built either from live ``Scenario`` objects
(``from_scenarios``) or collected metric files (``from_cases``), consuming only
the long-format metrics table.

Populated in Phase 3 of the refactor (see the refactoring plan, layer L5).
"""

from __future__ import annotations
