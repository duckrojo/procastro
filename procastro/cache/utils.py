def compare_caches(cache1, cache2):
    """
    Compare the contents of two caches based on their keys and values,
    ignoring the date stored in the (date, value) tuples.

    Args:
        cache1 (dict): The first cache to compare.
        cache2 (dict): The second cache to compare.

    Returns:
        bool: True if the caches are equivalent in keys and values, False otherwise.
    """
    # Extract keys and values, ignoring the date part of the tuple
    cache1_filtered = {key: value[1] for key, value in cache1.items()}
    cache2_filtered = {key: value[1] for key, value in cache2.items()}

    # Compare the filtered caches
    return cache1_filtered == cache2_filtered