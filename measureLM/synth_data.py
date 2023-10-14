import pandas as pd
import tqdm
from measureLM import helpers, measuring
tqdm.tqdm.pandas()

def load_synth_data(n=None, seed=0):
    df = pd.read_excel(helpers.ROOT_DIR / "data" / "friend_enemy_list.xlsx")
    if isinstance(n, int):
        df = df.sample(n=n, random_state=seed)
    return df


def measure_scale(df, prompt, model, scales, prefix=""):
    scale_names = [f"{prefix}" + ("-".join(scale)) for scale in scales]
    df[scale_names] = df.progress_apply(lambda row: measuring.get_scale_vals({"ent1": row["ent1"], "ent2": row["ent2"]},
                                                    prompt, model, scales),axis=1,result_type="expand")
    return df


if __name__ == "__main__":
    df = load_synth_data(n=5)
    model = measuring.load_model(model_name="gpt2-small", device="cpu")

    scales = [["good", "bad"], ["friendly", "hostile"], ["positive", "negative"]]
    prompt = "The relationship between {ent1} and {ent2} is"
    df = measure_scale(df, prompt, model, scales)
