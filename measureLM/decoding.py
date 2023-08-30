import torch

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


def encode(texts, tokenizer, model):
    if tokenizer.pad_token is None:  ## some tokenizers do not have pad tokens
        tokenizer.pad_token = tokenizer.eos_token
    tokens = tokenizer(texts, padding=True, return_tensors='pt')
    output = model(**tokens, output_hidden_states=True)
    return output, tokens


def ids_to_tokens(scores, topk=5):
    pred_scores, pred_tokens = [], []
    topK_preds = torch.topk(scores, k=topk)
    for scores, indices in zip(topK_preds.values.tolist(), topK_preds.indices.tolist()):
        scores = list(map(lambda score: round(score, 2), scores))
        pred_scores.append(scores)
        tokens = list(map(lambda idx: tokenizer.convert_ids_to_tokens(idx), indices))
        pred_tokens.append(tokens)
    return pred_tokens, pred_scores


def decode(h, model):
    if model.can_generate(): ## decoder-only
        scores = model.lm_head(h)
    else:  ## encoder-only
        scores = model.cls(h)
    return scores


def early_decoding(hidden_states, tok_idx, model, l_start_end=[0, 99]):
    for i, h in enumerate(hidden_states[l_start_end[0]:l_start_end[1]]):
        h = h[tok_idx]  ## get hidden states per token
        scores = decode(h, model)  ## decode the hidden state through last layer
        tokens, scores = ids_to_tokens(scores, topk=2)
        print(f"layer {i}: {list(zip(scores, tokens))}")



if __name__ == "__main__":

    from transformers import AutoModelForCausalLM, DebertaForMaskedLM, AutoTokenizer

    ## decoder-only model____________
    tokenizer = AutoTokenizer.from_pretrained('gpt2-medium')
    model = AutoModelForCausalLM.from_pretrained('gpt2-medium')

    texts = ["Hi, how are", "The capital of France is"]
    output, tokens = encode(texts, tokenizer, model)
    tok_idx = token_select(tokens, tokenizer, select_token=None)
    early_decoding(output.hidden_states, tok_idx, model)

    ## encoder-only model____________
    tokenizer = AutoTokenizer.from_pretrained("lsanochkin/deberta-large-feedback")
    model = DebertaForMaskedLM.from_pretrained("lsanochkin/deberta-large-feedback")

    texts = ["Hi, how are [MASK]", "The capital of France is [MASK]"]
    output, tokens = encode(texts, tokenizer, model)
    tok_idx = token_select(tokens, tokenizer, select_token="[MASK]")
    early_decoding(output.hidden_states, tok_idx, model)