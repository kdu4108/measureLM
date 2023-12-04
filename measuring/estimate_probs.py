from typing import Callable, List, Dict, Optional
from collections import Counter
import math
import numpy as np
import torch
import scipy.stats as sst
from transformers import GPTNeoXForCausalLM, AutoTokenizer

from preprocessing.utils import format_query

# 1. Approximate x ∈ Σ∗ with a set of contexts from a dataset
# 2. Approximate p(y|x, q[e]) with monte carlo samples of y given x and q[e].
# 3. Approximate p(y|q[e]) with monte carlo samples of x.
# 4. Approximate p(x) with samples from a corpus (empirical distribution), but meaning/interpretation is complicated.
# Run a model on each of these sentences and get a score


def create_position_ids_from_input_ids(input_ids, padding_idx):
    """
    Replace non-padding symbols with their position numbers. This is modified from huggingface's
    `transformers.modeling_utils.create_position_ids_from_input_ids`.

    :param torch.Tensor x:
    :return torch.Tensor:
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) - 1) * mask
    return (
        incremental_indices.long()
    )  # + padding_idx (for some reason this is here in the OG code, but I can't make sense of why)


def estimate_prob_x_given_e(entity: str, contexts: List[str]):
    """
    Returns a (len(contexts),) nparray containing the probability of each context.
    """
    return np.ones(len(contexts)) / len(contexts)


def get_prob_next_word(model: GPTNeoXForCausalLM, tokens: Dict[str, torch.LongTensor]):
    """
    Args:
        model
        tokens - dict of
            {
                "input_ids": torch tensor of token IDs with shape (bs, context_width),
                "attention_mask: torch tensor of attention mask with shape (bs, context_width)
            }

    Returns:
        (bs, vocab_sz) tensor containing the probability distribution over the vocab of the next token for each sequence in the batch `tokens`.
    """
    position_ids = create_position_ids_from_input_ids(tokens["input_ids"], model.config.pad_token_id)
    logits = model(**tokens, position_ids=position_ids)["logits"]  # shape: (bs, mcw, vocab_sz)
    return logits[:, -1, :]  # shape: (bs, vocab_sz)


def check_answer_map(model, answer_map):
    special_model_tokens = set([v for (k, v) in vars(model.config).items() if k.endswith("token_id")])
    counter = Counter()
    for k, idxs in answer_map.items():
        for idx in idxs:
            idx = idx.item()
            if idx in special_model_tokens:
                counter[idx] += 1
    if counter:
        raise ValueError(
            f"WARNING: some of the tokens in your answer map correspond to special tokens of the model you may not have intended. This could occur if one of your tokens to the model is unknown and therefore was given an ID of an UNK, PAD, EOS, etc. token. Here are the counts of each special token in your answer map: {counter}."
        )
        print(
            f"WARNING: some of the tokens in your answer map correspond to special tokens of the model you may not have intended. This could occur if one of your tokens to the model is unknown and therefore was given an ID of an UNK, PAD, EOS, etc. token. Here are the counts of each special token in your answer map: {counter}."
        )


def score_model_for_next_word_prob(
    prompts: List[str],
    model,
    tokenizer,
    start: int = 0,
    end: Optional[int] = None,
    answer_map: Dict[int, List[str]] = None,
) -> torch.FloatTensor:
    """
    Args:
        prompts - list of prompts on which to score the model and get probability distribution for next word
        model
        tokenizer
        start, end - optional indices at which to slice the prompts dataset for scoring. By default, slice the whole dataset.
        answer_map - dict from the answer support (as an int) to the tokens which qualify into the respective answer.

    Returns:
        (end-start, answer_vocab_sz)-shaped torch float tensor representing the logit distribution (over the answer vocab) for the next token for all prompts in range start:end.
    """
    tokens = tokenizer(prompts[start:end], padding=True, return_tensors="pt").to(
        model.device
    )  # shape: (len(contexts), max_context_width)
    last_word_logits = get_prob_next_word(model, tokens)  # shape: (len(contexts), vocab_sz)

    if answer_map is not None:
        check_answer_map(model, answer_map)
        last_word_logits_agg = torch.zeros(last_word_logits.shape[0], len(answer_map), device=model.device)
        for answer, option_ids in answer_map.items():
            logit_vals = torch.index_select(
                input=last_word_logits, dim=1, index=option_ids
            )  # shape; (bs, len(option_ids))
            last_word_logits_agg[:, answer] = torch.sum(logit_vals, dim=1)
        last_word_logits = last_word_logits_agg

    return last_word_logits


def determine_bs(f, model, tokenizer, prompts, **kwargs):
    raise NotImplementedError("TODO for when I want to optimize for efficiently maxing out GPU usage in batch scoring")


def sharded_score_model(
    f: Callable,
    model: GPTNeoXForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    bs: Optional[int] = None,
    **kwargs,
):
    if bs is None:
        bs = determine_bs(f, model, tokenizer, prompts, **kwargs)

    num_batches = math.ceil(len(prompts) / bs)
    output = []
    for b in range(num_batches):
        start, end = b * bs, min((b + 1) * bs, len(prompts))
        output.append(f(model=model, tokenizer=tokenizer, prompts=prompts, start=start, end=end, **kwargs))

    return torch.cat(output, dim=0)


def estimate_prob_next_word_given_x_and_entity(
    query, entity: str, contexts: List[str], model: GPTNeoXForCausalLM, tokenizer: AutoTokenizer, bs=32, answer_map=None
):
    """
    Args:
        entity: str - the entity of interest
        contexts: List[str] - list of contexts appended to the query regarding entity

    Returns:
      samples - a list of torch longtensors of shape (num_samples, max_length) with length len(contexts)
      possible_outputs - a dict mapping from all observed outputs to its unique index.
    """
    complete_queries = [format_query(query, entity, context) for context in contexts]

    if tokenizer.padding_side != "left":
        raise ValueError(
            f"Expected tokenizer {tokenizer} to have padding side of `left` for batch generation, instead has padding side of `{tokenizer.padding_side}`. Please make sure you initialize the tokenizer to use left padding."
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model.config.pad_token_id != model.config.eos_token_id:
        print("Setting model.config.pad_token_id to model.config.eos_token_id")
        model.config.pad_token_id = model.config.eos_token_id

    last_word_logits = sharded_score_model(
        f=score_model_for_next_word_prob,
        model=model,
        tokenizer=tokenizer,
        prompts=complete_queries,
        bs=bs,
        answer_map=answer_map,
    )  # shape: (len(contexts), vocab_sz)

    last_word_probs = torch.nn.functional.softmax(last_word_logits, dim=1)  # shape: (len(contexts, vocab_sz))

    return last_word_probs.cpu().detach().numpy()


def estimate_prob_y_given_context_and_entity(
    query: str,
    entity: str,
    contexts: List[str],
    model: GPTNeoXForCausalLM,
    tokenizer: AutoTokenizer,
    num_samples=None,
    max_output_length=1,
    answer_map=None,
):
    """
    Args:
        output_samples - a list of sampled outputs from the model given the context and entity.
                         Outputs need not be unique.

    Returns:
        a (len(set(output_samples)),) nparray containing the probability of each output.
    """
    if max_output_length > 1 and num_samples is None:
        raise ValueError(
            "Estimating p(y | x, q[e]) for outputs y with length >1 requires sampling. Please specify a value for num_samples."
        )

    if num_samples is not None:
        return sample_y_given_x_and_entity(
            query=query,
            entity=entity,
            contexts=contexts,
            model=model,
            tokenizer=tokenizer,
            num_samples=num_samples,
            max_output_length=max_output_length,
        )

    return estimate_prob_next_word_given_x_and_entity(
        query=query,
        entity=entity,
        contexts=contexts,
        model=model,
        tokenizer=tokenizer,
        answer_map=answer_map,
    )


def sample_y_given_x_and_entity(
    query, entity: str, contexts: List[str], model: GPTNeoXForCausalLM, tokenizer: AutoTokenizer, num_samples=1000
):
    """
    Args:
        entity: str - the entity of interest
        contexts: List[str] - list of contexts appended to the query regarding entity
    """
    raise NotImplementedError()


def estimate_cmi(query, entity, contexts, model, tokenizer, answer_map=None):
    """
    Computes the conditional mutual information I(X; Y | q[e]) of answer Y and context X when conditioned on query regarding entity e.

    I(X; Y | q[e]) = \sum_{x \in X} \sum_{y \in Y} (p(x, y | q[e]) * log(p(y | x, q[e]) / p(y | q[e]))) # noqa: W605

    So we need to monte carlo estimate:
        (1) p(y | x, q[e])                                             , shape: (|X|, |Y|)
        (2) p(x | q[e])                                                , shape: (|X|,)
        (3) p(x, y | q[e]) = p(y | x, q[e]) * p(x | q[e])              , shape: (|X|, |Y|)
        (4) p(y | q[e]) = \sum_{x \in X} (p(y | x, q[e]) * p(x | q[e])), shape: (|Y|,) # noqa: W605
    """
    prob_x_given_e = estimate_prob_x_given_e(entity, contexts)  # shape: (|X|,)
    prob_y_given_context_and_entity = estimate_prob_y_given_context_and_entity(
        query, entity, contexts, model, tokenizer, answer_map=answer_map
    )  # shape: (|X|, |Y|)

    prob_x_y_given_e = np.einsum("ij, i -> ij", prob_y_given_context_and_entity, prob_x_given_e)  # shape: (|X|, |Y|)
    prob_y_given_e = np.einsum("ij, i -> j", prob_y_given_context_and_entity, prob_x_given_e)  # shape: (|Y|,)

    return np.sum(prob_x_y_given_e * np.nan_to_num(np.log(prob_y_given_context_and_entity / prob_y_given_e)))


def kl_div(p, q):
    return sst.entropy(p, q, axis=1)


def difference(p, q):
    p_prob = torch.nn.functional.softmax(torch.tensor(p), dim=1)
    q_prob = torch.nn.functional.softmax(torch.tensor(q), dim=1)
    return ((p_prob[:, 1] - p_prob[:, 0]) - (q_prob[:, 1] - q_prob[:, 0])).detach().cpu().numpy()


def difference_abs_val(p, q):
    p_prob = torch.nn.functional.softmax(torch.tensor(p), dim=1)
    q_prob = torch.nn.functional.softmax(torch.tensor(q), dim=1)
    return torch.abs((p_prob[:, 1] - p_prob[:, 0]) - (q_prob[:, 1] - q_prob[:, 0])).detach().cpu().numpy()


def difference_p_good_only(p, q):
    p_prob = torch.nn.functional.softmax(torch.tensor(p), dim=1)
    q_prob = torch.nn.functional.softmax(torch.tensor(q), dim=1)
    return (p_prob[:, 1] - q_prob[:, 1]).detach().cpu().numpy()


def estimate_entity_score(query, entity, contexts, model, tokenizer, distance_metric=kl_div, answer_map=None):
    """
    Computes the conditional mutual information I(X; Y | q[e]) of answer Y and context X when conditioned on query regarding entity e.

    I(X; Y | q[e]) = \sum_{x \in X} \sum_{y \in Y} (p(x, y | q[e]) * log(p(y | x, q[e]) / p(y | q[e]))) # noqa: W605

    So we need to monte carlo estimate:
        (1) p(y | x, q[e])                                             , shape: (|X|, |Y|)
        (2) p(x | q[e])                                                , shape: (|X|,)
        (3) p(x, y | q[e]) = p(y | x, q[e]) * p(x | q[e])              , shape: (|X|, |Y|)
        (4) p(y | q[e]) = \sum_{x \in X} (p(y | x, q[e]) * p(x | q[e])), shape: (|Y|,) # noqa: W605
    """
    if tokenizer.padding_side != "left":
        raise ValueError(
            f"Expected tokenizer {tokenizer} to have padding side of `left` for batch generation, instead has padding side of `{tokenizer.padding_side}`. Please make sure you initialize the tokenizer to use left padding."
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model.config.pad_token_id != model.config.eos_token_id:
        print("Setting model.config.pad_token_id to model.config.eos_token_id")
        model.config.pad_token_id = model.config.eos_token_id

    prob_x_given_e = estimate_prob_x_given_e(entity, contexts)  # shape: (|X|,)
    prompt_sans_context = format_query(query=query, entity=entity, context="")
    pred_sans_context = score_model_for_next_word_prob(
        [prompt_sans_context], model, tokenizer, answer_map=answer_map
    )  # shape: (1, len(answer_map))

    prompts_with_context = [format_query(query=query, entity=entity, context=context) for context in contexts]
    preds_with_context = sharded_score_model(
        f=score_model_for_next_word_prob,
        model=model,
        tokenizer=tokenizer,
        prompts=prompts_with_context,
        bs=32,
        answer_map=answer_map,
    )  # shape: (len(contexts), len(answer_map))

    distance_with_context = distance_metric(
        preds_with_context.detach().cpu().numpy(), pred_sans_context.detach().cpu().numpy()
    )  # shape: (len(contexts),)

    return np.dot(distance_with_context, prob_x_given_e)


if __name__ == "__main__":
    query = "On a scale from 1 to 5 stars, the quality of this movie, '{}', is rated "
    entity = "The Dark Knight"
    contexts = [
        "Here's a movie review: 'The movie was terrific and I loved it'.",
        "Here's a movie review: 'The movie was awful and I hated it'.",
    ]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "EleutherAI/pythia-70m-deduped"

    model = GPTNeoXForCausalLM.from_pretrained(
        model_name,
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left",
    )

    query = "On a scale from 1 to 5 stars, the quality of this movie, '{}', is rated "
    print(estimate_cmi(query, entity, contexts, model, tokenizer))
    print(estimate_entity_score(query, entity, contexts, model, tokenizer, distance_metric=kl_div, answer_map=None))
