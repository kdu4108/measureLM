import torch
from measureLM import visualizing

## ENCODING and DECODING____________________________

def encode(texts, tokenizer, model):
    if tokenizer.pad_token is None:  ## some tokenizers do not have pad tokens
        tokenizer.pad_token = tokenizer.eos_token
    tokens = tokenizer(texts, padding=True, return_tensors='pt')
    output = model(**tokens, output_hidden_states=True)
    return output, tokens


def decode(h, model):
    if model.can_generate():  ## decoder-only
        scores = model.lm_head(h)
    else:  ## encoder-only
        scores = model.cls(h)
    return scores


def early_decoding(hidden_states, model, l_start_end=[0, 99]):
    layer_scores = []
    for i, h in enumerate(hidden_states[l_start_end[0]:l_start_end[1]]):
        scores = decode(h, model)  ## decode the hidden state through last layer
        layer_scores.append(scores)

    layer_scores = torch.stack(layer_scores)
    layer_scores = torch.swapaxes(layer_scores, 0, 1)
    layer_scores = torch.swapaxes(layer_scores, 1, 2)
    return layer_scores  ## dims: (prompts, tokens, layers, token_scores)


## SCORING____________________________

def token_select(tokens, tokenizer, select_token="[MASK]"):  # "Ġhi"
    if select_token is not None:
        select_token_id = tokenizer.convert_tokens_to_ids(select_token)  ## retrieve index of [MASK]
        batch_idx, seq_idx = (tokens.input_ids == select_token_id).nonzero(as_tuple=True)

    if select_token is None or select_token not in tokenizer.vocab:  ## get last token before padding
        batch_idx, seq_idx = (tokens.input_ids != tokenizer.pad_token_id).nonzero(as_tuple=True)
        batch_idx, unique_batch_counts = torch.unique_consecutive(batch_idx, return_counts=True)
        unique_batch_cumsum = torch.cumsum(unique_batch_counts, dim=0) - 1
        seq_idx = seq_idx[unique_batch_cumsum]

    assert batch_idx.shape[0] > 0, f"mlm-type model and {select_token} token not in prompt text"
    batch_seq_idx = (batch_idx, seq_idx)
    return batch_seq_idx


def topK_scores(scores, tokenizer, topk=5):
    pred_scores, pred_tokens = [], []
    topK_preds = torch.topk(scores, k=topk)

    scores = topK_preds.values.tolist()
    indices = topK_preds.indices.tolist()
    # for scores, indices in zip(topK_preds.values.tolist(), topK_preds.indices.tolist()):
    scores = list(map(lambda score: round(score, 2), scores))
    pred_scores.append(scores)
    tokens = list(map(lambda idx: tokenizer.convert_ids_to_tokens(idx), indices))
    pred_tokens.append(tokens)
    return pred_tokens, pred_scores


def get_token_rank(scores, tokenizer, token, space="Ġ"):
    if space is not None:
        token = "Ġ" + token
    token_id = tokenizer.convert_tokens_to_ids(token)
    token_ranks = torch.argsort(scores, descending=True)

    token_scores = scores[token_ranks]
    token_rank = torch.where(token_ranks == token_id)[0].item()
    token_score = token_scores[token_rank]
    token_rank = round(1 / (token_rank + 1), 4)  # round(1-(token_rank / len(scores)), 4)
    return token_rank, token_score


def scores_to_tokens(layer_scores, tokenizer, mode=2, print_res=True):
    prompt_layer_res = {}
    for idx, prompt in enumerate(layer_scores):
        print(f"\nprompt {idx}")
        layer_res = {}
        for l, scores in enumerate(prompt):
            if isinstance(mode, int):  ## get top tokens
                tokens, scores = topK_scores(scores, tokenizer, topk=mode)
                layer_res[l] = list(zip(scores, tokens))
                if print_res:
                    print(f"layer {l}: {layer_res[l]}")

            elif isinstance(mode, list):  ## search specific tokens
                if isinstance(mode[0], list) and len(mode) == len(layer_scores):
                    pass  ## per prompt mode
                elif isinstance(mode[0], str):
                    token_ranks, token_scores = [], []
                    for token in mode:
                        token_rank, token_score = get_token_rank(scores, tokenizer, token)
                        token_ranks.append(token_rank)
                    layer_res[l] = token_ranks
                    if print_res:
                        print(f"layer {l}: {list(zip(layer_res[l], mode))}")
        prompt_layer_res[idx] = layer_res
    return prompt_layer_res

if __name__ == "__main__":

    from transformers import AutoModelForCausalLM, DebertaForMaskedLM, AutoTokenizer

    ## decoder-only model____________
    tokenizer = AutoTokenizer.from_pretrained('gpt2-medium')
    model = AutoModelForCausalLM.from_pretrained('gpt2-medium')

    token_candidates = ["Paris", "France", "Germany"]
    prompts = ["Paris is the capital of", "The capital of France is"]

    ## encoding
    output, tokens = encode(prompts, tokenizer, model)
    layer_scores = early_decoding(output.hidden_states, model)

    ## scoring
    tok_idx = token_select(tokens, tokenizer)
    scored_tokens = scores_to_tokens(layer_scores[tok_idx], tokenizer, mode=token_candidates)
    visualizing.visualize_token_ranks(scored_tokens, token_candidates, prompts)


    ## encoder-only model____________
    tokenizer = AutoTokenizer.from_pretrained("lsanochkin/deberta-large-feedback")
    model = DebertaForMaskedLM.from_pretrained("lsanochkin/deberta-large-feedback")

    token_candidates = ["Paris", "France", "Germany"]
    prompts = ["Paris is the capital of [MASK]", "The capital of France is [MASK]"]

    ## encoding
    output, tokens = encode(prompts, tokenizer, model)
    layer_scores = early_decoding(output.hidden_states, model)

    ## scoring
    tok_idx = token_select(tokens, tokenizer)
    scored_tokens = scores_to_tokens(layer_scores[tok_idx], tokenizer, mode=token_candidates)
    visualizing.visualize_token_ranks(scored_tokens, token_candidates, prompts)