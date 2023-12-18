from typing import Tuple


def format_query(query: str, entity: Tuple[str], context: str, prefix=""):
    """
    Number of elements in entity must match the number of format {} things in query.
    This is to handle for multiple-entity entities (e.g. friend enemy pairs)
    """
    return prefix + context + query.format(*entity)
