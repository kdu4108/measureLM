import torch, transformer_lens, itertools
from functools import partial
from IPython.display import clear_output

from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)

from tqdm import tqdm
from measureLM import measuring, helpers


def construct_bias_context_pairs(df, prompt=None, pos_prefix=None, neg_prefix=None, pair_type="bias"):

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
            pos_context = measuring.form_prompt(f"{pos_prefix} {prompt}", {"ent1": ent1, "ent2": ent2})
            neg_context = measuring.form_prompt(f"{neg_prefix} {prompt}", {"ent1": ent1, "ent2": ent2})
            ent1_ent2_pairs.append((pos_context, neg_context))

    print(f"pair_type: {pair_type} --> {len(ent1_ent2_pairs)} data points")
    return ent1_ent2_pairs


def patch_hook_point(patched_activs, hook: HookPoint, new_activs, hook_layer_name, extract_tok_idx=-1,
                     insert_tok_idx=None):
    # print(f'patching {hook.name} <-- {hook_layer_name}')
    if extract_tok_idx is None or extract_tok_idx == -1:
        extract_tok_idx = (0, -1)
    if insert_tok_idx is None:
        insert_tok_idx = extract_tok_idx
    new_activs_hook = new_activs[hook_layer_name]
    vector_direction.append(torch.stack([new_activs_hook[extract_tok_idx], patched_activs[insert_tok_idx]]))
    patched_activs[insert_tok_idx] = new_activs_hook[extract_tok_idx]


def patch_activs(model, old_logits, new_logits, new_activs, prompt, logit_idx):
    n_layers = model.cfg.n_layers
    hook_names = ["attn_out", "mlp_out"]

    effect_strength = torch.zeros(n_layers, len(hook_names), device=model.cfg.device)
    global vector_direction
    vector_direction = []

    #for layer in tqdm(range(n_layers), position=1, leave=True):
    for layer in range(n_layers):
        for hook_i, hook_name in enumerate(hook_names):
            hook_layer_name = transformer_lens.utils.get_act_name(hook_name, layer)
            patch_layers_fn = [
                (hook_layer_name, partial(patch_hook_point, new_activs=new_activs, hook_layer_name=hook_layer_name))]
            patched_logits = model.run_with_hooks(prompt, fwd_hooks=patch_layers_fn, reset_hooks_end=True)

            ## get measurement change
            patched_logits = measuring.select_logits(patched_logits, logit_idx)
            patched_logit_diff = (patched_logits[..., 0] - patched_logits[..., 1])

            ## store effect strength
            old_logit_diff = (old_logits[..., 0] - old_logits[..., 1])
            new_logit_diff = (new_logits[..., 0] - new_logits[..., 1])
            effect_strength[layer, hook_i] = torch.abs((patched_logit_diff - old_logit_diff) / (new_logit_diff - old_logit_diff))
            # torch.abs(patched_logits_v-old_logits_v)
    clear_output()

    vector_direction = torch.stack(vector_direction)
    vector_direction = torch.movedim(vector_direction, 0, 1)
    vector_direction = vector_direction.view(2, model.cfg.n_layers, -1, model.cfg.d_model)
    return effect_strength.detach(), vector_direction.detach()


def run_patching_loop(model, prompt_pairs, scale_idx):
    all_vector_scale, all_vector_dir = [], []
    for (prompt_1, prompt_2) in tqdm(prompt_pairs, position=0, leave=False):
        old_logits, old_activs = measuring.prompt_with_cache(model, prompt_1, logit_idx=scale_idx)
        new_logits, new_activs = measuring.prompt_with_cache(model, prompt_2, logit_idx=scale_idx)
        vector_scale, vector_dir = patch_activs(model, old_logits, new_logits, new_activs, prompt_1, scale_idx)

        all_vector_scale.append(vector_scale)
        all_vector_dir.append(vector_dir)

    vector_scale = torch.stack(all_vector_scale).detach()  ## shape: prompt, layers, att vs mlp
    vector_scale = vector_scale.mean(0)
    vector_dir = torch.stack(all_vector_dir).detach()  ## shape: prompt, new vs old, layers, att vs mlp, emb dim
    return vector_scale, vector_dir

if __name__ == "__main__":

    model = measuring.load_model(model_name="gpt2-medium", device="mps")

    scales = ["good", "bad"]
    scale_idx = measuring.get_logit_indices(scales, model)

    old_prompt = ["The relationship between Harry Potter and Ronald Weasley is"]
    # new_prompt = ["Harry absolutely hates Ron. The relationship between Harry Potter and Ronald Weasley is"]
    new_prompt = ["The relationship between Jack and Mary is"]

    old_logits, old_activs = measuring.prompt_with_cache(model, old_prompt, logit_idx=scale_idx)
    new_logits, new_activs = measuring.prompt_with_cache(model, new_prompt, logit_idx=scale_idx)

    vector_scale, vector_dir = patch_activs(model, old_logits, new_logits, new_activs, old_prompt, scale_idx)