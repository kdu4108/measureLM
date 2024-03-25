import re
from negate import Negator


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


def negate_template(template: str) -> str:
    negator = Negator()

    if "'{entity}'" in template:
        template = template.replace("'{entity}'", "MyEntityBlahblahblahEtta")
        entity_sub = "apostr"
    elif "{entity}" in template:
        template = template.replace("{entity}", "MyEntityBlahblahblahEtta")
        entity_sub = "no_apostr"

    if "'{answer}'" in template:
        template = template.replace("'{answer}'", "MyAnswerBlahblahblahsAlice")
        answer_sub = "apostr"
    elif "{answer}" in template:
        template = template.replace("{answer}", "MyAnswerBlahblahblahsAlice")
        answer_sub = "no_apostr"

    negated_template = negator.negate_sentence(template, prefer_contractions=False)

    if entity_sub == "apostr":
        negated_template = negated_template.replace("MyEntityBlahblahblahEtta", "'{entity}'")
    elif entity_sub == "no_apostr":
        negated_template = negated_template.replace("MyEntityBlahblahblahEtta", "{entity}")

    if answer_sub == "apostr":
        negated_template = negated_template.replace("MyAnswerBlahblahblahsAlice", "'{answer}'")
    elif answer_sub == "no_apostr":
        negated_template = negated_template.replace("MyAnswerBlahblahblahsAlice", "{answer}")
    return negated_template


def lowercase_first_letter(s):
    return s[:1].lower() + s[1:]
