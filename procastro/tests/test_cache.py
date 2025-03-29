
import pytest
import numpy as np
from procastro.cache.cache import _AstroCache
from procastro.cache.cachev2 import _AstroCachev2

@pytest.fixture
def setup_caches():
    old_cache = _AstroCache(max_cache=10, lifetime=1)
    new_cache = _AstroCachev2(max_cache=10, lifetime=1)
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

    # Ejecutar ambas funciones con el mismo valor
    old_result_1 = old_cached_function(6)
    new_result_1 = new_cached_function(6)

    # Verificar que las funciones se ejecutaron una vez
    assert execution_count_old["count"] == 1
    assert execution_count_new["count"] == 1

    # Forzar la invalidación del caché
    old_result_2 = old_cached_function(6, force=True)
    new_result_2 = new_cached_function(6, force=True)

    # Verificar que las funciones se ejecutaron nuevamente
    assert execution_count_old["count"] == 2
    assert execution_count_new["count"] == 2

    # Verificar que los resultados después de la invalidación son iguales
    assert old_result_2 == new_result_2

def test_non_hashable_objects(setup_caches):
    """
    Test the behavior of caching functions when provided with non-hashable inputs.
    This test ensures that caching mechanisms can handle non-hashable objects
    (e.g., numpy arrays) without raising errors and that the cached functions
    return the correct results.
    Args:
        setup_caches (tuple): A fixture that provides two caching mechanisms,
                              `old_cache` and `new_cache`, for testing.
    Steps:
        1. Define two cached functions using the provided caching mechanisms.
        2. Use a non-hashable input (numpy array) to call the cached functions.
        3. Verify that the results of the cached functions match the expected
           computation (element-wise square of the input).
        4. Check that the input is indeed non-hashable by attempting to hash it
           and catching a `TypeError`.
    Assertions:
        - The results of the cached functions match the expected computation.
        - The input is confirmed to be non-hashable.
    """
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
    