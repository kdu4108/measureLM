import torch, transformer_lens, tqdm

def load_model(model_name="gpt2-small", device="cpu"):
    model = transformer_lens.HookedTransformer.from_pretrained(model_name).to(device)
    model.cfg.spacing = "Ġ"
    model.tokenizer.pad_token = model.tokenizer.eos_token
    return model

def get_token_indices(toks_str, model):
    assert isinstance(toks_str, list), f"tokens_str must be a list of strings"
    # token = model.cfg.spacing + token
    # token_id = model.tokenizer.convert_tokens_to_ids(token)
    #token_idx = torch.LongTensor([model.tokenizer.encode(f" {token}")[0] for token in tokens_str])
    toks_idx = torch.LongTensor([model.tokenizer.convert_tokens_to_ids(model.cfg.spacing + tok_str) for tok_str in toks_str])
    return toks_idx

def select_logits(logits, tok_idx, norm=True):
    logits = torch.index_select(logits[..., -1, :], -1, tok_idx)
    if norm:
        logits = logits / logits.sum()
    return logits

class Default(dict):
    def __missing__(self, key):
        return '{' + key + '}'

def get_scale_vals(kwargs, prompt, model, scales, reversed=False):

    scale_vals = []
    for scale in scales:
        toks_idx = get_token_indices(scale, model)

        prompt = prompt.format_map(Default(kwargs))
        logits, activs = model.run_with_cache(prompt)
        toks_logits = select_logits(logits, toks_idx)
        scale_val = toks_logits[..., 0].item()

        if reversed:
            kwargs['ent1'], kwargs['ent2'] = kwargs['ent2'], kwargs['ent1']
            prompt = prompt.format_map(Default(kwargs))
            logits, activs = model.run_with_cache(prompt)
            toks_logits = select_logits(logits, toks_idx)
            scale_val_reverse = toks_logits[..., 0].item()
            scale_val = (scale_val + scale_val_reverse) / 2

        scale_vals.append(scale_val)
    return scale_vals


if __name__ == "__main__":

    scales = [["good", "bad"], ["friendly", "hostile"], ["positive", "negative"]]
    prompt = "The relationship between {ent1} and {ent2} is"
    model = load_model()
    scale_vals = get_scale_vals({"ent1": "Trump", "ent2": "Biden"}, prompt, model, scales)
    print(scale_vals)
