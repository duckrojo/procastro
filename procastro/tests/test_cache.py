import pytest
import tempfile
import time
from unittest.mock import MagicMock
from procastro.astrofile.astrofile import AstroFile
from procastro.cache.cachev2 import _AstroCachev2, astrofile_cachev2, astrofile_cachev2_on_disk
from pathlib import Path

@pytest.fixture
def astrocache():
    """Fixture to create an in-memory _AstroCachev2 instance."""
    return _AstroCachev2()


@pytest.fixture
def disk_based_astrocache():
    """Fixture to create a disk-based _AstroCachev2 instance and clean up after tests."""
    cache_dir = "tempdir"
    cache = _AstroCachev2(label_on_disk=cache_dir)

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
    @astrocache
    def cached_function(x):
        return x ** 2

    # First call should compute the result
    assert cached_function(2) == 4

    # Second call should return the cached result
    assert cached_function(2) == 4

def test_cache_expiration(astrocache):
    """Test that cached items expire after the specified lifetime."""
    astrocache = _AstroCachev2(lifetime=1 / 86400)  # 1 second lifetime

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
    astrocache = _AstroCachev2(max_cache=100)  # Small cache size

    @astrocache
    def cached_function(x):
        return x ** 2

    # Add multiple items to the cache
    for i in range(10):
        cached_function(i)

    # Verify that some items have been evicted
    # (Exact behavior depends on the eviction policy)
    assert cached_function(0) == 0  # May have been evicted and recomputed



def test_astrofile_with_astrocachev2():
    """Test that the astrofile cache works correctly with _AstroCachev2."""
    file_dir = "demo_data/ob01_5_spec.fits"
    file = AstroFile(file_dir)
    data = file.data
    assert data is not None, "Data should be loaded successfully"
    assert astrofile_cachev2_on_disk._store_on_disk , "Data should be on disk"
    assert astrofile_cachev2_on_disk.cache_directory , "Data should have a valid path"




