from datetime import datetime

import pytest

from home.bus import InMemoryBus
from home.devices import SimClock


@pytest.fixture
def bus():
    return InMemoryBus()


@pytest.fixture
def clock():
    # A Monday at noon: the sun is up, so the battery rule stays quiet unless a test wants it.
    return SimClock(datetime(2026, 10, 12, 12, 0))
