from measureLM import visualizing

import torch
import transformer_lens

## ENCODING and DECODING____________________________

def encode(texts, model, tokens_only=False):
    if model.tokenizer.pad_token is None:  ## some tokenizers do not have pad tokens
        model.tokenizer.pad_token = model.tokenizer.eos_token
    if hasattr(model, 'to_tokens'):
        tokens = model.to_tokens(texts, prepend_bos=False)
    elif hasattr(model, 'tokenizer'):
        tokens = model.tokenizer(texts, padding=True, return_tensors='pt').input_ids
    else:
        raise Exception(f"no tokenizer given")
    if tokens_only:
        return tokens
    logits, activs = model.run_with_cache(tokens)
    return logits, activs, tokens


def decode(h, model):
    if hasattr(model, 'mlm_head'):
        scores = model.unembed(model.mlm_head(h))
    elif hasattr(model, 'ln_final'):
        scores = model.unembed(model.ln_final(h))
    else:
        raise Exception(f"no decoding module")
    return scores


def early_decoding(activs, model, l_start_end=[0, 99]):
    layer_scores = []
    for layer in range(l_start_end[0], l_start_end[1]):
        if layer < model.cfg.n_layers:
            if isinstance(activs, transformer_lens.ActivationCache):
                hidden_name = transformer_lens.utils.get_act_name("resid_post", layer)
                h = activs[hidden_name]
            elif isinstance(activs, torch.Tensor):
                h = activs[...,layer,:]
            scores = decode(h, model)  ## decode the hidden state through last layer
            layer_scores.append(scores)

    layer_scores = torch.stack(layer_scores)
    layer_scores = torch.swapaxes(layer_scores, 0, 1)
    layer_scores = torch.swapaxes(layer_scores, 1, 2)
    return layer_scores  ## dims: (prompts, tokens, layers, token_scores)


if __name__ == "__main__":

    ## decoder-only model____________
    model = transformer_lens.HookedTransformer.from_pretrained("gpt2-medium").to("cpu")
    model.tokenizer.pad_token = model.tokenizer.eos_token
    model.cfg.spacing = "Ġ"

    token_candidates = ["Paris", "France", "Poland", "Warsaw"]
    prompts = ["Q: What is the capital of France? A: Paris Q: What is the capital of Poland? A:"]

    ## encoding
    logits, activs, tokens = encode(prompts, model)
    layer_scores = early_decoding(activs, model)


