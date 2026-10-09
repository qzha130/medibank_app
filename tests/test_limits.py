"""Anonymous sessions share a request budget and cannot bypass it by refreshing."""

from concurrent.futures import ThreadPoolExecutor

from medibank.limits import RequestLimiter


def test_global_budget_counts_distinct_sessions_and_recovers_after_window():
    limiter = RequestLimiter(limit=2, window_seconds=60)
    assert limiter.allow("first", now=0)
    assert limiter.allow("second", now=1)
    assert not limiter.allow("third", now=2)
    assert limiter.allow("third", now=60)
    assert not limiter.allow("fourth", now=60)


def test_rejected_cooldown_does_not_consume_shared_budget():
    limiter = RequestLimiter(limit=2, window_seconds=60, cooldown_seconds=5)
    assert limiter.allow("first", now=0)
    assert not limiter.allow("first", now=4)
    assert limiter.allow("first", now=5)


def test_concurrent_requests_cannot_exceed_global_budget():
    limiter = RequestLimiter(limit=3)
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda number: limiter.allow(str(number), now=0), range(8)))
    assert sum(results) == 3
