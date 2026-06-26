"""L2 -- the single-scenario analysis object.

Will hold ``Scenario``, replacing the ``OutputAnalysis`` + ``TotalMetrics`` pair:
one Hercules run, a derived ``channels`` frame composed from the L1 functions,
and metrics computed on demand.

Populated in Phase 2 of the refactor (see the refactoring plan, layer L2).
"""

from __future__ import annotations
