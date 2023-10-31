import pandas as pd
import tqdm, itertools
from measureLM import helpers, measuring
tqdm.tqdm.pandas()

def load_synth_data(n=None, seed=0):
    df = pd.read_excel(helpers.ROOT_DIR / "data" / "friend_enemy_list.xlsx")
    if isinstance(n, int):
        df = df.sample(n=n, random_state=seed)
    return df


def construct_bias_context_pairs(df, prompt=None, pair_type="bias", from_no_context=True, pos_prefix=None, neg_prefix=None):

    if prompt is None:
        prompt = "The relationship between {ent1} and {ent2} is"
    if pos_prefix is None:
        pos_prefix = "{ent1} loves {ent2}."
    if neg_prefix is None:
        neg_prefix = "{ent1} hates {ent2}."

    ent1_ent2 = list(zip(df["ent1"].to_list(), df["ent2"].to_list()))
    if pair_type == "bias":
        ent1_ent2_pairs = []
        entPair1_entPair2 = list(itertools.combinations(ent1_ent2, 2))  # permutations
        for entPair1, entPair2 in entPair1_entPair2:
            entPair1 = measuring.form_prompt(prompt, {"ent1": entPair1[0], "ent2": entPair1[1]})
            entPair2 = measuring.form_prompt(prompt, {"ent1": entPair2[0], "ent2": entPair2[1]})
            ent1_ent2_pairs.append((entPair1, entPair2))

    elif pair_type == "context":
        ent1_ent2_pairs = []
        for ent1, ent2 in ent1_ent2:
            no_context_prompt = measuring.form_prompt(f"{prompt}", {"ent1": ent1, "ent2": ent2})
            pos_context = measuring.form_prompt(f"{pos_prefix} {prompt}", {"ent1": ent1, "ent2": ent2})
            neg_context = measuring.form_prompt(f"{neg_prefix} {prompt}", {"ent1": ent1, "ent2": ent2})

            if from_no_context:
                ent1_ent2_pairs.append((no_context_prompt, pos_context))
                ent1_ent2_pairs.append((no_context_prompt, neg_context))
            else:
                ent1_ent2_pairs.append((pos_context, no_context_prompt))
                ent1_ent2_pairs.append((neg_context, no_context_prompt))

    print(f"pair_type: {pair_type} --> {len(ent1_ent2_pairs)} data points")
    return ent1_ent2_pairs


if __name__ == "__main__":
    df = load_synth_data(n=5)
    model = measuring.load_model(model_name="gpt2-small", device="cpu")

    scales = [["good", "bad"], ["friendly", "hostile"], ["positive", "negative"]]
    prompt = "The relationship between {ent1} and {ent2} is"
    df = measure_scale(df, prompt, model, scales)
    context_pairs = construct_bias_context_pairs(df, pair_type="context")


