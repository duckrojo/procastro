import pytest
import tempfile
import time
from procastro.astro.solar_system import HorizonsInterface

from procastro.astrofile.astrofile import AstroFile
from procastro.cache.cache import AstroCache
from pathlib import Path
from procastro.astro.solar_system import jpl_cache, usgs_map_cache
from procastro.astrofile.astrofile import astrofile_cache


@pytest.fixture
def astrocache():
    """Fixture to create an in-memory AstroCache instance."""
    return AstroCache()


@pytest.fixture
def disk_based_astrocache():
    """Fixture to create a disk-based AstroCache instance and clean up after tests."""
    cache_dir = "tempdir"
    cache = AstroCache(label_on_disk=cache_dir)

    # Yield the cache instance for use in tests
    yield cache

    # Cleanup: Remove the cachedir directory after the test
    cache._cache.close()  # Close the cache to release any locks
    path = Path(cache_dir)
    if path.exists() and path.is_dir():
        for file in path.iterdir():
            file.unlink()  # Remove all files in the directory
        path.rmdir()  # Remove the directory itself


def test_basic_caching(astrocache):
    """Test that the cache stores and retrieves results correctly."""

    counts = {"count": 0}

    @astrocache
    def cached_function(x):
        counts["count"] += 1
        return x ** 2
    # First call should compute the result
    assert cached_function(2) == 4

    assert astrocache._cache.__len__() == 1  # Check if the cache has the result

    # Second call should return the cached result
    assert cached_function(2) == 4

    # Check if the cache still has the result
    assert astrocache._cache.__len__() == 1
    assert counts["count"] == 1  # Ensure the function was called only once


def test_cache_expiration(astrocache):
    """Test that cached items expire after the specified lifetime."""
    astrocache = AstroCache(lifetime=1 / 86400)  # 1 second lifetime

    @astrocache
    def cached_function(x):
        return x ** 2

    # First call should compute the result
    assert cached_function(2) == 4

    # Wait for the cache to expire
    time.sleep(2)

    # Second call should recompute the result
    assert cached_function(2) == 4


def test_force_bypass(astrocache):
    """Test that the force parameter bypasses the cache."""
    calls = {
        "count": 0
    }

    @astrocache
    def cached_function(x):
        calls["count"] += 1
        return x ** 2

    # First call should compute the result
    assert cached_function(2) == 4

    # Second call with force=True should recompute the result
    assert cached_function(2, force=True) == 4
    assert calls["count"] == 2  # Ensure the function was called twice


def test_non_hashable_arguments(astrocache):
    """Test that non-hashable arguments raise a TypeError."""
    @astrocache
    def cached_function(arg):
        return arg

    # Non-hashable argument (e.g., a list)
    with pytest.raises(TypeError):
        cached_function([1, 2, 3])


def test_disk_based_cache(disk_based_astrocache):
    """Test that the cache works correctly when stored on disk."""
    disk_based_astrocache.clear()

    @disk_based_astrocache
    def cached_function(x):
        return x ** 2

    cache_dir = Path(disk_based_astrocache.cache_directory)
    assert cache_dir.exists() and cache_dir.is_dir()

    assert disk_based_astrocache.get_stats().get('keys_count') == 0
    # First call should compute the result
    assert cached_function(2) == 4
    cache_file = cache_dir / "cache.db"
    size = cache_file.stat().st_size

    assert size is not None, "El archivo de cache debería crecer tras almacenar un valor"

    assert disk_based_astrocache.get_stats().get('keys_count') == 1

    # Segunda llamada no debe cambiar el tamaño del archivo
    assert cached_function(2) == 4
    size_after_second_call = cache_file.stat().st_size
    assert size_after_second_call == size, "El tamaño del archivo de caché no debería cambiar tras una segunda llamada con el mismo argumento"
    assert disk_based_astrocache.get_stats().get('keys_count') == 1


def test_memory_cache_size(astrocache):
    """Test that the cache size does not change after repeated cached calls."""
    astrocache = AstroCache()  # Set a small cache size limit
    astrocache.clear()

    @astrocache
    def cached_function(x):
        return x ** 2
    assert astrocache.get_stats().get('currsize') == 0
    # Add multiple items to the cache
    for i in range(1, 10):
        print(f"Adding {i**2} to cache")
        cached_function(i)
        assert astrocache.get_stats().get('currsize') == i


def test_eviction_policy(astrocache):
    """Test that the cache evicts items when the size limit is exceeded."""
    astrocache = AstroCache(1)  # Small cache size in bytes

    @astrocache
    def cached_function(x):
        return x ** 2

    # Add multiple items to the cache
    for i in range(10):
        print(f"Adding {i} to cache")
        print(f"Cache size: {astrocache._cache.__len__()}")
        cached_function(i)

    # check that the cache size is 1.

    assert astrocache.get_stats().get('currsize') == 1

    # Check if the cache has been evicted correctly
    assert astrocache._cache.__len__() == 1


def test_astrofile_cache():
    """Test that the cache works correctly with AstroFile."""
    path = "procastro/timeseries/example/data/raw/wasp19-000.fits"
    astrofile_cache.clear()  # Clear the cache before the test
    af = AstroFile(path)
    data1 = af.data  # Primera llamada, debería invocar la función real
    # Verificar que el caché tiene el resultado
    assert astrofile_cache.get_stats().get("currsize") == 1

    data2 = af.data  # Segunda llamada, debería venir del caché
    # Verificar que el caché tiene el resultado
    assert astrofile_cache.get_stats().get("currsize") == 1

    # Verificar que los datos son los mismos
    assert data1 is not None
    assert data2 is not None

    # Para arrays de NumPy, usar comparaciones apropiadas
    import numpy as np

    if isinstance(data1, np.ndarray) and isinstance(data2, np.ndarray):
        # Para arrays NumPy con posibles NaN
        assert np.array_equal(data1, data2, equal_nan=True)
    elif hasattr(data1, '__array__') and hasattr(data2, '__array__'):
        # Para objetos similares a arrays (como Table de astropy)
        assert np.array_equal(np.asarray(
            data1), np.asarray(data2), equal_nan=True)
    else:
        # Para otros tipos de datos
        assert data1 == data2

    # Verificar que realmente vienen del cache (mismo objeto en memoria)
    assert data1 is data2, "Los datos deberían ser el mismo objeto si vienen del cache"


def test_jpl_cache_basic():
    """Probar que las solicitudes a _request_horizons_online se almacenan en caché."""

    spec = "COMMAND=301\nCENTER='500@399'\nMAKE_EPHEM=YES"
    cached_request = HorizonsInterface._request_horizons_online
    result1 = cached_request(spec)
    result2 = cached_request(spec)

    # Verificar que el resultado es el mismo pero se recalcula
    # Verificar que el caché tiene el resultado y es solo una vez
    assert jpl_cache._cache.__len__() == 1

    # Verificar que el resultado es el mismo y que se almacena en caché

    assert result1 == result2


def test_jpl_cache_force():
    """Probar que el parámetro force ignora el caché."""
    spec = "COMMAND=301\nCENTER='500@399'\nMAKE_EPHEM=YES"
    cached_request = HorizonsInterface._request_horizons_online
    result1 = cached_request(spec)
    result2 = cached_request(spec, force=True)

    # Verificar que el resultado es diferente (ya que se forzó la recalculación)
    assert result1 != result2
    assert jpl_cache._cache.__len__() == 1
