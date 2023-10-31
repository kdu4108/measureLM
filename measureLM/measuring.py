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
        logits = logits / logits.sum(-1)
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


def get_scale_vals(prompt, kwargs, model, scales, scale_val_type="pos", norm=True, reversed=False):
    scale_vals = []
    for scale in scales:
        if isinstance(scale, list):
            logit_idx = get_logit_indices(scale, model)
        elif isinstance(scale, torch.Tensor):
            logit_idx = scale ## do nothing, scale is already provided as logit indeces
        toks_logits, _ = prompt_with_cache(model, form_prompt(prompt, kwargs), logit_idx, norm)
        scale_val = compute_scale_val(toks_logits, scale_val_type)

        if reversed:
            kwargs['ent1'], kwargs['ent2'] = kwargs['ent2'], kwargs['ent1']
            toks_logits, _ = prompt_with_cache(model, form_prompt(prompt, kwargs), logit_idx, norm)

            scale_val_reverse = compute_scale_val(toks_logits, scale_val_type)
            scale_val = (scale_val + scale_val_reverse) / 2

        scale_vals.append(scale_val)
    return scale_vals


def measure_scale(df, prompt, model, scales, prefix=""):
    if isinstance(scales[0], list):
        scale_names = [f"{prefix}" + ("-".join(scale)) for scale in scales]
        df[scale_names] = df.progress_apply(lambda row: get_scale_vals({"ent1": row["ent1"], "ent2": row["ent2"]},prompt, model, scales),axis=1,result_type="expand")
    else:
        df[f"{prefix}"] = df.progress_apply(lambda row: get_scale_vals(prompt, {"ent1": row["ent1"],"ent2": row["ent2"]},model, scales,scale_val_type="pos",norm=True, reversed=True)[0],axis=1, result_type="expand")
    return df


if __name__ == "__main__":

    model = load_model(model_name="gpt2-small", device="mps")
    scales = [["good", "bad"], ["friendly", "hostile"], ["positive", "negative"]]
    logit_idx = get_logit_indices(scales[0], model)

    prompt = "The relationship between {ent1} and {ent2} is"
    scale_vals = get_scale_vals(prompt, {"ent1": "Trump", "ent2": "Biden"}, model, [logit_idx])
    print(scale_vals)
