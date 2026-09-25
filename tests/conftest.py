"""Shared test isolation for the FastAPI smoke suite."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


_TEST_DATA_ROOT = Path(tempfile.mkdtemp(prefix="shorts-studio-tests-"))
os.environ["LOCAL_OUTPUT_DIR"] = str(_TEST_DATA_ROOT / "output")
os.environ["SHORTS_STUDIO_DATA_DIR"] = str(_TEST_DATA_ROOT)
os.environ["SHORTS_AUTO_RESUME"] = "false"
# The render budget is one process-global per-minute quota, so a full run's worth
# of enqueues would decide which late test can still create a project.  Every
# test that asserts the quota monkeypatches its own limiter and limit; a wide
# default keeps the rest of the suite order-independent.
os.environ["SHORTS_JOB_RATE_LIMIT_PER_MINUTE"] = "100000"
