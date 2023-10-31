from measureLM import helpers

import pandas as pd
import random


def format_prompt(country, city, contextCity=None):
    in_context_prefix = "The capital of {country} is {contextCity}."
    template = "Q: What is the capital of {country}? A:"
    prompt = template.format(country=country, city=city)

    if contextCity is not None:
        in_context_prefix = in_context_prefix.format(country=country, contextCity=contextCity)
        prompt = f"{in_context_prefix} {prompt}"
    return prompt


def load_country_capitals(n_pairs=10, n_wrong_contexts=0, seed=0):
    df = pd.read_csv(helpers.ROOT_DIR / "data" / "CountryCapital" / "country-capital.csv")
    countries, capitals = df["country"].to_list(), df["capital"].to_list()

    random.seed(seed)
    idcs = random.sample(range(len(countries)), n_pairs)

    prompts = []
    for idx in idcs:
        country, capital = countries[idx], capitals[idx]
        memory_prompt = format_prompt(country, capital, contextCity=None)
        right_context_prompt = format_prompt(country, capital, contextCity=capital)
        country_capital_prompts = [capital, memory_prompt, right_context_prompt]

        wrong_context_idcs = random.sample(list(set(range(len(countries))) - set([idx])), n_wrong_contexts)
        for wrong_context_idx in wrong_context_idcs:
            wrong_context_prompt = format_prompt(country, capital, contextCity=capitals[wrong_context_idx])
            country_capital_prompts.append(wrong_context_prompt)
        prompts.append(country_capital_prompts)
    return prompts


if __name__ == "__main__":
    prompts = load_country_capitals(n_pairs=10, n_wrong_contexts=0, seed=0)
    print(prompts)