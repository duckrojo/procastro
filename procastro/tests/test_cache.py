
import pytest
import numpy as np
from procastro.cache.cache import _AstroCache
from procastro.cache.cachev2 import _AstroCachev2

@pytest.fixture
def setup_caches():
    old_cache = _AstroCache(max_cache=10,lifetime=10)
    new_cache = _AstroCachev2(max_cache=10,lifetime=10)
    return old_cache, new_cache

def test_cache_results(setup_caches):
    old_cache, new_cache = setup_caches

    @old_cache
    def old_cached_function(x):
        return x ** 2

    @new_cache
    def new_cached_function(x):
        return x ** 2

    test_data = [1, 2, 3, 4, 5]
    for x in test_data:
        assert old_cached_function(x) == new_cached_function(x)

def test_cache_invalidation(setup_caches,capsys):
    old_cache, new_cache = setup_caches

    execution_count_old = {"count": 0}
    execution_count_new = {"count": 0}

    @old_cache
    def old_cached_function(x, force=False):
        execution_count_old["count"] += 1
        return x ** 2

    @new_cache
    def new_cached_function(x, force=False):
        execution_count_new["count"] += 1
        return x ** 2

    old_result_1 = old_cached_function(6)
    new_result_1 = new_cached_function(6)

    assert execution_count_old["count"] == 1
    assert execution_count_new["count"] == 1

    old_result_2 = old_cached_function(6, force=True)
    new_result_2 = new_cached_function(6, force=True)

    assert execution_count_old["count"] == 2
    assert execution_count_new["count"] == 2

    assert old_result_2 == new_result_2

def test_non_hashable_objects(setup_caches):

    old_cache, new_cache = setup_caches

    @old_cache
    def old_cached_function(x):
        return x ** 2

    @new_cache
    def new_cached_function(x):
        return x ** 2

    non_hashable_input = np.array([1, 2, 3])

    old_result = old_cached_function(non_hashable_input)
    new_result = new_cached_function(non_hashable_input)

    assert np.array_equal(old_result, non_hashable_input ** 2)
    assert np.array_equal(new_result, non_hashable_input ** 2)

    try:
        hash(non_hashable_input)
        is_hashable = True
    except TypeError:
        is_hashable = False

    assert not is_hashable, "The input should not be hashable"


def test_max_cache_limit(setup_caches):
    old_cache, new_cache = setup_caches

    @old_cache
    def old_cached_function(x):
        return x ** 2

    @new_cache
    def new_cached_function(x):
        return x ** 2

    for i in range(15):
        old_cached_function(i)
        new_cached_function(i)

    assert len(old_cache._cache) <= 10, "Old cache exceeded max size"
    assert len(new_cache._cache) <= 10, "New cache exceeded max size"


    
    






