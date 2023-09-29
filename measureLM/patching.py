import torch
import transformer_lens
from functools import partial
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)


def create_hook_mapping(model, patch_l=(0, 40), extract_l=None, hook_type="mlp_out"):
    if isinstance(patch_l, tuple):
        patch_l = list(range(*patch_l))
    if extract_l is None:
        extract_l = patch_l
    if isinstance(extract_l, tuple):
        extract_l = list(range(*extract_l))

    if len(extract_l) < len(patch_l):
        extract_l = extract_l + ([extract_l[-1]] * (len(patch_l) - len(extract_l)))

    extract_l[:] = [x for x in extract_l if x < model.cfg.n_layers]
    patch_l[:] = [x for x in patch_l if x < model.cfg.n_layers]

    extract_l = [transformer_lens.utils.get_act_name(hook_type, l) for l in extract_l]
    patch_l = [transformer_lens.utils.get_act_name(hook_type, l) for l in patch_l]
    patch_extract_map = dict(zip(patch_l, extract_l))
    return patch_extract_map


def patch_point(patched_activs, hook: HookPoint, old_activs, patch_map, extract_tok_idx, insert_tok_idx=None):
    print(f'patching {hook.name} <-- {patch_map[hook.name]}')
    old_activs = old_activs[patch_map[hook.name]]
    if extract_tok_idx is None or extract_tok_idx == -1:
        extract_tok_idx = (0, -1)
    if insert_tok_idx is None:
        insert_tok_idx = extract_tok_idx
    patched_activs[insert_tok_idx] = old_activs[extract_tok_idx]
    return patched_activs


def extract_resid_post(resid_post_layer, hook: HookPoint):
    #print(f'extracting {hook.name}')
    resid_post[..., hook.layer(), :] = resid_post_layer


def intervene(new_tokens, old_activs, model, patch_map, extract_tok_idx=-1, insert_tok_idx=None):
    ## patching mlp output_________________
    patch_hook_fn = partial(patch_point, old_activs=old_activs, patch_map=patch_map, extract_tok_idx=extract_tok_idx, insert_tok_idx=insert_tok_idx)
    patch_layers_fn = [(hook_point, patch_hook_fn) for hook_point in patch_map.keys()]

    ## extracting residual stream out___________________
    global resid_post
    resid_post = torch.zeros(*new_tokens.shape, model.cfg.n_layers, model.cfg.d_model, device=model.cfg.device)
    extract_hook_fn = partial(extract_resid_post)
    extract_layers_fn = [(transformer_lens.utils.get_act_name("resid_post", layer), extract_hook_fn) for layer in range(0, model.cfg.n_layers)]

    reset_hooks_end = True
    patch_logits = model.run_with_hooks(new_tokens, fwd_hooks=patch_layers_fn + extract_layers_fn, return_type="logits",reset_hooks_end=reset_hooks_end)
    return patch_logits, resid_post



if __name__ == "__main__":

    from measureLM import visualizing, decoding, scoring

    model = transformer_lens.HookedTransformer.from_pretrained("gpt2-medium").to("cpu")
    model.cfg.spacing = "Ġ"
    model.tokenizer.pad_token = model.tokenizer.eos_token

    ## 1 get activations_______________
    prompts = ["Q: What is the capital of France? A: Paris Q: What is the capital of Poland? A:"]
    logits, activs, tokens = decoding.encode(prompts, model)
    layer_scores = decoding.early_decoding(activs, model)

    ## 2 use activations to intervene_______________
    new_prompts = ["Germany, mug, table, Germany, mug, table,"]
    new_logits, new_activs, new_tokens = decoding.encode(new_prompts, model)
    pred = model.tokenizer.convert_ids_to_tokens(torch.topk(new_logits[:, -1, :], k=3).indices.tolist()[0])
    print(pred)

    patch_map = create_hook_mapping(model, extract_l=(15, 40), patch_l=(15, 40))
    patch_logits, resid_post = intervene(new_tokens, activs, model, patch_map, extract_tok_idx=-1)
    resid_layer_scores = decoding.early_decoding(resid_post, model)

    pred = model.tokenizer.convert_ids_to_tokens(torch.topk(patch_logits[:, -1, :], k=3).indices.tolist()[0])
    print(pred)

    new_token_candidates = ["Germany", "Berlin", "mug", "table"]

    ## scoring
    tok_idx = scoring.token_select(new_tokens, model)
    scored_tokens = scoring.scores_to_tokens(resid_layer_scores, tok_idx, model, mode=new_token_candidates)
    visualizing.visualize_token_ranks(scored_tokens, new_token_candidates, new_prompts)
