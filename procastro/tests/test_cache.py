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

astrofile_cache = AstroCache()
jpl_cache = AstroCache(max_cache=1e12, lifetime=30,)
usgs_map_cache = AstroCache(max_cache=30, lifetime=30,
                            hashable_kw=['detail'], label_on_disk='USGSmap',
                            force_kwd="no_cache")

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

    counts = {"count":0}
    @astrocache
    def cached_function(x):
        counts["count"] += 1
        return x ** 2
    # First call should compute the result
    assert cached_function(2) == 4

    assert astrocache._cache.__len__() == 1  # Check if the cache has the result

    # Second call should return the cached result
    assert cached_function(2) == 4
    

    assert astrocache._cache.__len__() == 1  # Check if the cache still has the result
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
    @astrocache
    def cached_function(x):
        return x ** 2

    # First call should compute the result
    assert cached_function(2) == 4

    # Second call with force=True should recompute the result
    assert cached_function(2, force=True) == 4

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
    @disk_based_astrocache
    def cached_function(x):
        return x ** 2

    # First call should compute the result
    assert cached_function(2) == 4

    # Second call should return the cached result
    assert cached_function(2) == 4

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

    #check if the key for 0 is not in the cache
    assert astrocache._cache.__len__() == 0  # Check if the cache has been evicted correctly



def test_astrofile_cache():
    """Test that the cache works correctly with AstroFile."""
    path = "demo_data/ob01_5_spec.fits"
    af = AstroFile(path)
    data1 = af.data  # Primera llamada, debería invocar la función real
    assert astrofile_cache._cache.__len__() == 1  # Verificar que el caché tiene el resultado
    data2 = af.data  # Segunda llamada, debería venir del caché
    assert astrofile_cache._cache.__len__() == 1  # Verificar que el caché sigue teniendo el resultado
    # Verificar que los datos son los mismos
    assert data1 is not None
    assert data2 is not None
    



def test_jpl_cache_basic():
    """Probar que las solicitudes a _request_horizons_online se almacenan en caché."""
    
    spec = "COMMAND=301\nCENTER='500@399'\nMAKE_EPHEM=YES"
    cached_request = HorizonsInterface._request_horizons_online
    result1 = cached_request(spec)
    result2 = cached_request(spec)

    # Verificar que el resultado es el mismo pero se recalcula
    assert jpl_cache._cache.__len__() == 1  # Verificar que el caché tiene el resultado y es solo una vez




    # Verificar que el resultado es el mismo y que se almacena en caché
    
    assert result1 == result2

def test_jpl_cache_force():
    """Probar que el parámetro force ignora el caché."""
    spec = "COMMAND=301\nCENTER='500@399'\nMAKE_EPHEM=YES"
    cached_request = HorizonsInterface._request_horizons_online
    result1 = cached_request(spec)
    result2 = cached_request(spec, force=True)
    
    assert result1 != result2  # Verificar que el resultado es diferente (ya que se forzó la recalculación)
    assert jpl_cache._cache.__len__() == 1



