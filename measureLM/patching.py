import torch, transformer_lens, itertools
from functools import partial

from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)

from tqdm import tqdm
from measureLM import measuring, helpers

COS = torch.nn.CosineSimilarity(dim=-1, eps=1e-08)

## Get Patching____________________________________

def patch_hook_point(old_activs, hook: HookPoint, new_activs, hook_layer_name, extract_tok_idx=-1,
                     insert_tok_idx=None):
    # print(f'patching {hook.name} <-- {hook_layer_name}')
    if extract_tok_idx is None or extract_tok_idx == -1:
        extract_tok_idx = (0, -1)
    if insert_tok_idx is None:
        insert_tok_idx = extract_tok_idx
    new_activs_hook = new_activs[hook_layer_name]
    vector_direction.append(torch.stack([old_activs[insert_tok_idx].detach(), new_activs_hook[extract_tok_idx]]))
    old_activs[insert_tok_idx] = new_activs_hook[extract_tok_idx]


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
            patch_layers_fn = [(hook_layer_name, partial(patch_hook_point, new_activs=new_activs, hook_layer_name=hook_layer_name))]
            patched_logits = model.run_with_hooks(prompt, fwd_hooks=patch_layers_fn, reset_hooks_end=True)

            ## get measurement change
            patched_logits = measuring.select_logits(patched_logits, logit_idx, norm=False)
            patched_logit_diff = (patched_logits[..., 0] - patched_logits[..., 1])

            ## store effect strength
            old_logit_diff = measuring.compute_scale_val(old_logits, scale_val_type="diff")
            new_logit_diff = measuring.compute_scale_val(new_logits, scale_val_type="diff")
            #effect_strength[layer, hook_i] = (patched_logit_diff - old_logit_diff) / (new_logit_diff - old_logit_diff)
            effect = torch.abs((patched_logit_diff - old_logit_diff) / (new_logit_diff - old_logit_diff))
            #effect = torch.abs(patched_logits[..., 0]-old_logits[..., 0])
            effect_strength[layer, hook_i] = effect

    vector_direction = torch.stack(vector_direction)
    vector_direction = torch.movedim(vector_direction, 0, 1)
    vector_direction = vector_direction.view(2, model.cfg.n_layers, -1, model.cfg.d_model)
    return effect_strength.detach(), vector_direction.detach()


def run_patching_loop(model, prompt_pairs, scale_idx):
    all_vector_scale, all_vector_dir = [], []
    for (prompt_1, prompt_2) in tqdm(prompt_pairs, position=0, leave=False):
        old_logits, old_activs = measuring.prompt_with_cache(model, prompt_1, logit_idx=scale_idx, norm=False)
        new_logits, new_activs = measuring.prompt_with_cache(model, prompt_2, logit_idx=scale_idx, norm=False)
        vector_scale, vector_dir = patch_activs(model, old_logits, new_logits, new_activs, prompt_1, scale_idx)

        all_vector_scale.append(vector_scale)
        all_vector_dir.append(vector_dir)

    vector_scale = torch.stack(all_vector_scale).detach()  ## shape: prompt, layers, att vs mlp
    vector_dir = torch.stack(all_vector_dir).detach()  ## shape: prompt, new vs old, layers, att vs mlp, emb dim
    return vector_scale, vector_dir


### Apply Patching____________________________________________________________

def prepare_dir_scale_patch(dir_vec, scale_vec):
    a, b = dir_vec[:, 0], dir_vec[:, 1]

    # a / |a| * |b| * (a * b / (|a| * |b|))
    # --> a * (a * b / |a|**2)
    dir_vec = torch.einsum("...nlcd,...nlcd->...nlc", a, b) / (torch.norm(a, dim=-1) ** 2)
    dir_vec = torch.einsum("...nlcd,...nlc->...nlcd", a, dir_vec)

    dir_scale = torch.einsum("...nlcd,...nlc->...nlcd", dir_vec, scale_vec)
    dir_scale = dir_scale.mean(0)
    return dir_scale


def control_hook_point(activs, hook: HookPoint, dir_scale, alpha):
    i = int(hook.layer())
    j = {"hook_attn_out": 0, "hook_mlp_out": 1}[hook.name.split(".")[2]]
    patched_activs = activs[..., -1, :] + (alpha * (dir_scale[i, j, :]))
    activs[..., -1, :] = patched_activs


def control_bias_context(model, prompt, dir_scale, alpha=1.0):
    patch_hook_fn = partial(control_hook_point, dir_scale=dir_scale, alpha=alpha)
    patch_layers_fn = [(lambda name: name.endswith("attn_out") or name.endswith("mlp_out"), patch_hook_fn)]

    patch_logits = model.run_with_hooks(prompt, fwd_hooks=patch_layers_fn, return_type="logits", reset_hooks_end=True)
    return patch_logits


if __name__ == "__main__":

    model = measuring.load_model(model_name="gpt2-small", device="mps")

    scales = ["good", "bad"]
    scale_idx = measuring.get_logit_indices(scales, model)

    old_prompt = ["The relationship between Harry Potter and Ronald Weasley is"]
    # new_prompt = ["Harry absolutely hates Ron. The relationship between Harry Potter and Ronald Weasley is"]
    new_prompt = ["The relationship between Jack and Mary is"]

    old_logits, old_activs = measuring.prompt_with_cache(model, old_prompt, logit_idx=scale_idx)
    new_logits, new_activs = measuring.prompt_with_cache(model, new_prompt, logit_idx=scale_idx)

    vector_scale, vector_dir = patch_activs(model, old_logits, new_logits, new_activs, old_prompt, scale_idx)

