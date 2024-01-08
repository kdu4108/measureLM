import re
from typing import Tuple


def format_query(query: str, entity: Tuple[str], context: str, prefix="", answer=None):
    """
    Number of elements in entity must match the number of format {} things in query.
    This is to handle for multiple-entity entities (e.g. friend enemy pairs)
    """
    if "{entity}" in query:
        if "{answer}" in query:
            concrete_query = query.format(entity=entity[0], answer=answer)
        else:
            concrete_query = query.format(entity=entity[0])
    else:
        if answer is not None:
            concrete_query = query.format(*entity, answer=answer)
        else:
            concrete_query = query.format(*entity)
    return prefix + context + concrete_query


def extract_name_from_yago_uri(uri: str):
    is_reversed = False
    if uri.startswith("reverse-"):
        uri = uri.split("reverse-")[1]
        is_reversed = True
    pattern = r"http://(?:www\.)?([^\/]+)\.org\/(.+)$"
    matches = re.match(pattern, uri)

    if matches:
        kb_domain = matches.group(1)
        relation = matches.group(2)
    else:
        raise ValueError(f"Could not find match containing kb_domain and relation for uri {uri}.")

    domain_to_name = {"schema": "schema", "yago-knowledge": "yago", "w3": "w3"}
    kb_name = domain_to_name[kb_domain]
    kb_name = ("reverse-" if is_reversed else "") + kb_name

    return kb_name, relation
