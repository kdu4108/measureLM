import torch
import transformer_lens
from functools import partial
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)


def patch_mlp_out(mlp_out_new, hook: HookPoint, old_activs: torch.Tensor, patch_tok_idx: int):
    print(f'patching {hook.name}')
    mlp_out_old = old_activs[hook.name]
    mlp_out_new[:, patch_tok_idx, :] = mlp_out_old[:, patch_tok_idx, :]
    return mlp_out_new


def extract_resid_post(resid_post_layer, hook: HookPoint):
    print(f'extracting {hook.name}')
    resid_post[..., hook.layer(), :] = resid_post_layer


def intervene(new_tokens, old_activs, model, patch_tok_idx=-1, l_start_end=[0, 99]):
    ## patching mlp output_________________
    patch_layers = list(range(0, model.cfg.n_layers))[l_start_end[0]:l_start_end[1]]
    patch_hook_fn = partial(patch_mlp_out, old_activs=old_activs, patch_tok_idx=patch_tok_idx)
    patch_layers_fn = [(transformer_lens.utils.get_act_name("mlp_out", layer), patch_hook_fn) for layer in patch_layers]

    ## extracting residual stream out___________________
    global resid_post
    resid_post = torch.zeros(*new_tokens.shape, model.cfg.n_layers, model.cfg.d_model)
    extract_hook_fn = partial(extract_resid_post)
    extract_layers_fn = [(transformer_lens.utils.get_act_name("resid_post", layer), extract_hook_fn) for layer in
                         range(0, model.cfg.n_layers)]

    patch_logits = model.run_with_hooks(new_tokens, fwd_hooks=patch_layers_fn + extract_layers_fn, return_type="logits",
                                        reset_hooks_end=True)
    return patch_logits, resid_post


if __name__ == "__main__":
    from measureLM import visualizing, decoding

    model = transformer_lens.HookedTransformer.from_pretrained("gpt2-medium").to("cpu")
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

    patch_logits, resid_post = intervene(new_tokens, activs, model, patch_tok_idx=-1, l_start_end=[15, 30])
    resid_layer_scores = decoding.early_decoding(resid_post, model)

    pred = model.tokenizer.convert_ids_to_tokens(torch.topk(patch_logits[:, -1, :], k=3).indices.tolist()[0])
    print(pred)

    new_token_candidates = ["Germany", "Berlin", "mug", "table"]

    ## scoring
    tok_idx = decoding.token_select(new_tokens, model)
    scored_tokens = decoding.scores_to_tokens(resid_layer_scores[tok_idx], model, mode=new_token_candidates)
    visualizing.visualize_token_ranks(scored_tokens, new_token_candidates, new_prompts)
