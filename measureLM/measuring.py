import torch, transformer_lens, tqdm

def load_model(model_name="gpt2-small", device="cpu"):
    model = transformer_lens.HookedTransformer.from_pretrained(model_name).to(device)
    model.cfg.spacing = "Ġ"
    model.tokenizer.pad_token = model.tokenizer.eos_token
    return model

def get_logit_indices(toks_str, model):
    assert isinstance(toks_str, list), f"tokens_str must be a list of strings"
    # token = model.cfg.spacing + token
    # token_id = model.tokenizer.convert_tokens_to_ids(token)
    #token_idx = torch.LongTensor([model.tokenizer.encode(f" {token}")[0] for token in tokens_str])
    toks_idx = torch.LongTensor([model.tokenizer.convert_tokens_to_ids(model.cfg.spacing + tok_str) for tok_str in toks_str])
    return toks_idx.to(model.cfg.device)

def select_logits(logits, logit_idx, norm=False):
    logits = torch.index_select(logits[..., -1, :], -1, logit_idx)
    if norm:
        logits = logits / logits.sum()
    return logits


def form_prompt(prompt, kwargs):
    class Default(dict):
        def __missing__(self, key):
            return '{' + key + '}'
    prompt = prompt.format_map(Default(kwargs))
    return prompt


def prompt_with_cache(model, prompt, logit_idx=None, norm=False):
    logits, activs = model.run_with_cache(prompt)
    if isinstance(logit_idx,torch.LongTensor) or isinstance(logit_idx,torch.Tensor):
        logits = select_logits(logits, logit_idx, norm)
    return logits, activs


def compute_scale_val(logits, scale_val_type="diff"):
    if scale_val_type=="diff":
        scale_val = (logits[..., 0] - logits[..., 1])
    elif scale_val_type == "pos":
        scale_val = logits[..., 0].item()
    return scale_val


def get_scale_vals(kwargs, prompt, model, scales, scale_val_type="diff", norm=False, reversed=False):
    scale_vals = []
    for scale in scales:
        toks_idx = get_logit_indices(scale, model)

        logits, activs = prompt_with_cache(model, prompt, kwargs)
        toks_logits = select_logits(logits, toks_idx, norm)
        scale_val = compute_scale_val(toks_logits, scale_val_type)

        if reversed:
            kwargs['ent1'], kwargs['ent2'] = kwargs['ent2'], kwargs['ent1']
            logits, activs = prompt_with_cache(model, prompt, kwargs)

            toks_logits = select_logits(logits, toks_idx, norm)
            scale_val_reverse = compute_scale_val(toks_logits, scale_val_type)
            scale_val = (scale_val + scale_val_reverse) / 2

        scale_vals.append(scale_val)
    return scale_vals


if __name__ == "__main__":

    scales = [["good", "bad"], ["friendly", "hostile"], ["positive", "negative"]]
    prompt = "The relationship between {ent1} and {ent2} is"
    model = load_model()
    scale_vals = get_scale_vals({"ent1": "Trump", "ent2": "Biden"}, prompt, model, scales)
    print(scale_vals)
