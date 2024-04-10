import re
import random
from typing import Tuple, List


def format_query(query: str, entity: Tuple[str], context: str, prefix="", answer=None):
    """
    Number of elements in entity must match the number of format {} things in query.
    This is to handle for multiple-entity entities (e.g. friend enemy pairs)
    """
    if not isinstance(entity, tuple):
        raise ValueError("entity must be of type tuple.")
    if "{entity}" in query:
        if "{answer}" in query:
            if answer is None:
                raise ValueError("Expected answer to be provided because query contains {answer} but none was given.")
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
    if uri.startswith("http://schema.org"):
        pattern = r"http://(?:www\.)?([^\/]+)\.org\/(.+)$"
    elif uri.startswith("http://yago-knowledge.org"):
        pattern = r"http:\/\/([^\.]+)\.org\/resource\/([^\/]+)"
    else:
        raise ValueError(
            f"Expected uri to only start with http://schema.org or http://yago-knowledge.org, instead received {uri}"
        )
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


def top_entity_uri_degree(entities_and_answers: List[Tuple[str]], yago_ec: dict, n: int):
    """
    Returns the top n entities by the entity_uri degree in the YAGO graph.
    Args:
        entities_and_answers: List of (entity, answer) tuples
        yago_ec: YAGO dict for a fixed query. contains keys "entity_uri_to_degree" and "entity_namesake_to_degree", along with others.
        n: Number of entities to return

    Returns:
        Length n list of (entity, answer) tuples
    """
    entities = set(yago_ec["entities"])

    # Need to convert from entity to the entity_uri. We assume that there is only one entity_uri per entity (entity namesake).
    entity_to_entity_uri = {e: e_uri for (e, e_uri) in zip(yago_ec["entities"], yago_ec["entity_uris"])}
    entity_uri_to_degree = {
        e_uri: degree for (e_uri, degree) in zip(yago_ec["entity_uris"], yago_ec["entity_uri_to_degree"])
    }

    # Fake entities iff entity havs degree 0
    entities_and_answers_and_degs = [
        (e, a, entity_uri_to_degree[entity_to_entity_uri[e]] if e in entities else 0) for (e, a) in entities_and_answers
    ]
    entities_and_answers_and_degs = sorted(entities_and_answers_and_degs, key=lambda x: x[2], reverse=True)

    # We want to return n entities, but we want to return half real and half fake.
    # TODO: generalize for >2 entity_types
    num_real = n // 2
    num_fake = n - num_real

    # We want the top num_real real entities, but we sample from the fake entities to avoid alphabetically sorted fake names
    real_e_a = [(e, a) for (e, a, _) in entities_and_answers_and_degs[:num_real]]
    fake_e_a = random.sample([(e, a) for (e, a, d) in entities_and_answers_and_degs if d == 0], num_fake)
    return real_e_a + fake_e_a


def top_entity_namesake_degree(entities_and_answers: List[Tuple[str]], yago_ec: dict, n: int):
    """
    Returns the top n entities by the namesake degree in the YAGO graph.
    Args:
        entities_and_answers: List of (entity_uri, answer) tuples
        yago_ec: YAGO dict for a fixed query. contains keys "entity_uri_to_degree" and "entity_namesake_to_degree", along with others.
        n: Number of entities to return

    Returns:
        Length n list of (entity, answer) tuples
    """
    # Some name mixing here: entity refers to the entity namesake (label in YAGO), except when specified otherwise.
    entities = set(yago_ec["entities"])
    entity_namesake_to_degree = {
        e: degree for (e, degree) in zip(yago_ec["entities"], yago_ec["entity_namesake_to_degree"])
    }

    # Fake entities iff entity havs degree 0
    entities_and_answers_and_degs = [
        (e, a, entity_namesake_to_degree[e] if e in entities else 0) for (e, a) in entities_and_answers
    ]
    entities_and_answers_and_degs = sorted(entities_and_answers_and_degs, key=lambda x: x[2], reverse=True)

    # We want to return n entities, but we want to return half real and half fake.
    num_real = n // 2
    num_fake = n - num_real

    # We want the top num_real real entities, but we sample from the fake entities to avoid alphabetically sorted fake names
    real_e_a = [(e, a) for (e, a, _) in entities_and_answers_and_degs[:num_real]]
    fake_e_a = random.sample([(e, a) for (e, a, d) in entities_and_answers_and_degs if d == 0], num_fake)
    return real_e_a + fake_e_a


def random_sample(entities_and_answers: List[Tuple[str]], yago_ec: dict, n: int):
    return random.sample(entities_and_answers, n)
