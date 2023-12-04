def format_query(query: str, entity: str, context: str, prefix=""):
    return prefix + context + query.format(entity)
