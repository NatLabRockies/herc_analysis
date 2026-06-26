"""L0 -- Hercules boundary.

Thin adapter over :class:`hercules.hercules_output.HerculesOutput`. Loads a run
and exposes its metadata via a read-through scalar record; copies no bulk data.

Populated in Phase 2 of the refactor (see the refactoring plan, layer L0).
"""

from __future__ import annotations
