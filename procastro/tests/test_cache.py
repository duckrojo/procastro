
from time import sleep
import pytest
import numpy as np
from procastro.cache.cache import _AstroCache
from procastro.cache.cachev2 import _AstroCachev2
from procastro.cache.utils import compare_caches
import diskcache as dc

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
    
    assert compare_caches(old_cache._cache, new_cache._cache), "Caches do not match!"




def test_cache_invalidation(setup_caches):
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


    old_result_1 = old_cached_function(6, force = False)
    new_result_1 = new_cached_function(6, force = False)

    assert execution_count_old["count"] == 1
    assert execution_count_new["count"] == 1

    old_result_2 = old_cached_function(6, force=True)
    new_result_2 = new_cached_function(6, force=True)

    assert execution_count_old["count"] == 2
    assert execution_count_new["count"] == 2

    assert old_result_2 == new_result_2

    assert compare_caches(old_cache._cache, new_cache._cache), "Caches do not match!"

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

    assert compare_caches(old_cache._cache, new_cache._cache), "Caches do not match!"


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

    #verify that the oldest entries are removed
    old_cache_removed_keys = [(0,),(1,), (2,), (3,), (4,)]
    new_cache_removed_keys = [(0,),(1,), (2,), (3,), (4,)]

    for key in old_cache_removed_keys:
        assert key not in old_cache._cache, f"Old cache should not contain {key}"
    
    for key in new_cache_removed_keys:
        assert key not in new_cache._cache, f"New cache should not contain {key}"


    assert compare_caches(old_cache._cache, new_cache._cache), "Caches do not match!"
    
    



def test_cache_memoize():

    cache= dc.Cache(max_cache=10,lifetime=10)
    

    execution_count = {"count": 0}
    @cache.memoize(expire= 10)
    def expensive_function(x):
        """A function that simulates an expensive computation."""
        execution_count["count"] += 1
        sleep(1)
        return x ** 2

    assert expensive_function(2) == 4
    assert expensive_function(2) == 4
    assert execution_count["count"] == 1, "Function should only be executed once for the same input."
    sleep(11)  # Wait for the cache to expire
    assert expensive_function(2) == 4, "Function should be executed again after cache expiration."
    assert execution_count["count"] == 2, "Function should be executed again after cache expiration."


def test_memoize_with_disk_usage():
    cache = dc.Cache(directory="cachedir")
    cache.clear()  # Clear the cache before starting
    execution_count = {"count": 0}
    @cache.memoize(expire= 10)
    def expensive_function(x):
        """A function that simulates an expensive computation."""
        execution_count["count"] += 1
        sleep(1)
        return x ** 2

    assert expensive_function(2) == 4
    assert expensive_function(2) == 4
    assert execution_count["count"] == 1, "Function should only be executed once for the same input."
    sleep(11)  # Wait for the cache to expire
    assert expensive_function(2) == 4, "Function should be executed again after cache expiration."
    assert execution_count["count"] == 2, "Function should be executed again after cache expiration."


    
def test_memoize_with_disk_and_expiration_time():
    cache = dc.Cache(directory="cachedir")
    cache.clear()  # Clear the cache before starting

    expiration={"count": 0}
    @cache.memoize(expire= 10) #cache which expires in 10 seconds
    def expensive_function(x):
        """A function that simulates an expensive computation."""
        expiration["count"] += 1
        sleep(1)
        return expiration["count"]
    
    expensive_function(2)  # Call the function to populate the cache
    sleep(5)  # Wait for 5 seconds
    assert expensive_function(2) == 1, "Cache should still contain the value."
    sleep(6)  # Wait for the cache to expire
    assert expensive_function(2) == 2,  "Cache should have been executed, because it expired "
    cache.close()  # Close the cache to release resources



def test_memoize_with_cache_dict():
    cache = dc.Cache(directory="cachedir")
    cache.clear()  # Clear the cache before starting


    @cache.memoize()
    def expensive_function(x):
        """A function that simulates an expensive computation."""
        sleep(1)
        return x ** 2
    
    ##filling the cache with 10 elements
    for i in range(10):
        expensive_function(i)
    expected_results = [i ** 2 for i in range(10)]
    for key in cache.iterkeys():
        assert cache[key] in expected_results, f"Cache should contain {key}"


def test_cache_memoize_with_files():
    cache = dc.Cache(directory="cachedir",size_limit=0.5)
    cache.clear()


    @cache.memoize()
    def read_file(filename):
        with open(filename, 'r') as file:
            return file.read()
        
    