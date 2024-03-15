from typing import Callable, List, Dict, Optional, Set, Tuple
from collections import Counter
import math
import numpy as np
import re
import pandas as pd
import torch
import scipy.stats as sst
from transformers import GPTNeoXForCausalLM, AutoTokenizer

from preprocessing.utils import format_query
from measuring.utils import AnswerGroup

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


def estimate_prob_x_given_e(entity: str, contexts: Set[str], contexts_counter: Optional[Counter] = None):
    """
    Returns a (len(contexts),) nparray containing the probability of each context.

    Args:
        contexts - a set of unique contexts
        contexts_counter - a counter mapping the counts of each context in the list of contexts
    """
    if contexts_counter is not None:
        return np.array([contexts_counter[c] / contexts_counter.total() for c in contexts])

    # Otherwise, assume uniform distribution over contexts
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
    try:
        position_ids = create_position_ids_from_input_ids(tokens["input_ids"], model.config.pad_token_id)
        logits = model(**tokens, position_ids=position_ids)["logits"]  # shape: (bs, mcw, vocab_sz)
    except TypeError as e:  # noqa: F841
        # print(
        #     f"Failed to make forward pass with position_ids; do you have a sufficient transformers library version? (e.g. >=4.30.0 ish?)\nFull error: {e}"
        # )
        logits = model(**tokens)["logits"]  # shape: (bs, mcw, vocab_sz)
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


def generate(
    prompts: List[str],
    model,
    tokenizer,
    max_output_length,
    start: int = 0,
    end: Optional[int] = None,
) -> torch.FloatTensor:
    """
    Args:
        prompts - list of prompts on which to score the model and get probability distribution for next word
        model
        tokenizer
        start, end - optional indices at which to slice the prompts dataset for scoring. By default, slice the whole dataset.
        max_output_length - how many tokens to generate

    Returns:
        (end-start, max_output_length)-shaped torch long tensor representing the generated outputfor all prompts in range start:end.
    """
    tokens = tokenizer(prompts[start:end], padding=True, return_tensors="pt").to(
        model.device
    )  # shape: (len(contexts), max_context_width)
    output_tokens = model.generate(**tokens, max_length=len(tokens["input_ids"][0]) + max_output_length)[
        :, -max_output_length:
    ]

    return output_tokens


def determine_bs(f, model, tokenizer, prompts, **kwargs):
    raise NotImplementedError("TODO for when I want to optimize for efficiently maxing out GPU usage in batch scoring")


def sharded_score_model(
    f: Callable,
    model: GPTNeoXForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    bs: Optional[int] = None,
    **kwargs,
) -> torch.FloatTensor:
    if bs is None:
        bs = determine_bs(f, model, tokenizer, prompts, **kwargs)

    num_batches = math.ceil(len(prompts) / bs)
    output = []
    for b in range(num_batches):
        start, end = b * bs, min((b + 1) * bs, len(prompts))
        output.append(
            f(model=model, tokenizer=tokenizer, prompts=prompts, start=start, end=end, **kwargs).detach().cpu()
        )

    return torch.cat(output, dim=0).float()


def estimate_prob_next_word_given_x_and_entity(
    query,
    entity: str,
    contexts: Set[str],
    model: GPTNeoXForCausalLM,
    tokenizer: AutoTokenizer,
    bs=32,
    answer_map=None,
    answer_entity: Optional[str] = None,
):
    """
    Args:
        entity: str - the entity of interest
        contexts: List[str] - list of contexts appended to the query regarding entity

    Returns:
      samples - a list of torch longtensors of shape (num_samples, max_length) with length len(contexts)
      possible_outputs - a dict mapping from all observed outputs to its unique index.
    """
    complete_queries = [format_query(query, entity, context, answer=answer_entity) for context in contexts]

    if tokenizer.padding_side != "left":
        raise ValueError(
            f"Expected tokenizer {tokenizer} to have padding side of `left` for batch generation, instead has padding side of `{tokenizer.padding_side}`. Please make sure you initialize the tokenizer to use left padding."
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model.config.pad_token_id != model.config.eos_token_id:
        print("Setting model.config.pad_token_id to model.config.eos_token_id")
        model.config.pad_token_id = model.config.eos_token_id

    last_word_logits: torch.FloatTensor = sharded_score_model(
        f=score_model_for_next_word_prob,
        model=model,
        tokenizer=tokenizer,
        prompts=complete_queries,
        bs=bs,
        answer_map=answer_map,
    )  # shape: (len(contexts), vocab_sz)

    last_word_probs = torch.nn.functional.softmax(last_word_logits, dim=1)  # shape: (len(contexts, vocab_sz))

    return last_word_probs.detach().cpu().numpy()


def estimate_prob_y_given_context_and_entity(
    query: str,
    entity: str,
    contexts: Set[str],
    model: GPTNeoXForCausalLM,
    tokenizer: AutoTokenizer,
    num_samples=None,
    max_output_length=1,
    answer_map=None,
    bs=32,
    answer_entity: Optional[str] = None,
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
            answer_entity=answer_entity,
        )  # TODO: implement this to work for answer_entity

    return estimate_prob_next_word_given_x_and_entity(
        query=query,
        entity=entity,
        contexts=contexts,
        model=model,
        tokenizer=tokenizer,
        answer_map=answer_map,
        bs=bs,
        answer_entity=answer_entity,
    )


def sample_y_given_x_and_entity(
    query,
    entity: str,
    contexts: List[str],
    model: GPTNeoXForCausalLM,
    tokenizer: AutoTokenizer,
    num_samples=1,
    max_output_length=10,
    bs=32,
    answer_entity: Optional[str] = None,
) -> torch.LongTensor:
    """
    Args:
        entity: str - the entity of interest
        contexts: List[str] - list of contexts appended to the query regarding entity
        max_output_length: int - max number of tokens to output. Default to 10 because most entities are 1-5 words long

    Returns:
        a (len(contexts), max_output_length)-shaped tensor of the model's tokens for each complete query prefixed with a context in contexts.
    """
    complete_queries = [format_query(query, entity, context, answer=answer_entity) for context in contexts]

    if tokenizer.padding_side != "left":
        raise ValueError(
            f"Expected tokenizer {tokenizer} to have padding side of `left` for batch generation, instead has padding side of `{tokenizer.padding_side}`. Please make sure you initialize the tokenizer to use left padding."
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model.config.pad_token_id != model.config.eos_token_id:
        print("Setting model.config.pad_token_id to model.config.eos_token_id")
        model.config.pad_token_id = model.config.eos_token_id

    output_tokens = sharded_score_model(
        f=generate,
        model=model,
        tokenizer=tokenizer,
        prompts=complete_queries,
        bs=bs,
        max_output_length=max_output_length,
    )

    return output_tokens  # shape: (len(contexts), max_output_length)


def is_answer_original_or_context(output: str, original_answer: str, context_answer: str) -> AnswerGroup:
    if output.strip().startswith(original_answer):
        return AnswerGroup.ORIGINAL
    elif output.strip().startswith(context_answer):
        return AnswerGroup.CONTEXT
    else:
        return AnswerGroup.OTHER


def construct_regex_for_answer_from_context_template(template):
    """
    Given a context template like "{entity} is the highest point of", constructs a regex which can return the answer from a sentence completing that template (until the end of a period and new line).
    So, we want a regex that can return "Asia" for the sentence "Jamaica is the highest point of Asia.\n"
    """
    # Patterns for entity and answer placeholders
    entity_pattern = r"(?:.+)"  # non matching group
    answer_pattern = r"(.*?)"  # matching group

    # Escape special characters in the template, then replace placeholders
    template_escaped = re.escape(template)
    template_with_patterns = template_escaped.replace("\\{entity\\}", entity_pattern).replace(
        "\\{answer\\}", answer_pattern
    )

    # The final regex pattern captures the answer
    regex_pattern = template_with_patterns + r"(?=\.\n|$)"
    return regex_pattern


def extract_answer(context_template: str, sentence: str):
    """
    Given a context_template (e.g., "{entity} is the highest point of")
    and a sentence (e.g. "Jamaica is the highest point of Asia.\n"),
    returns the answer in the context (e.g., "Asia").
    """
    regex = construct_regex_for_answer_from_context_template(context_template)
    match = re.search(regex, sentence)
    if match:
        return match.group(1)

    print("No match found")
    return None


def p_score_ent_diff(prob_y_given_e, prob_y_given_context_and_entity):
    """
    Defining p_score as:
      H(Y|E=e) - H(Y|X=x, E=e)
    Return shape: (|X|,)
    """
    H_y_given_e: np.float16 = -np.sum(prob_y_given_e * np.nan_to_num(np.log(prob_y_given_e)))  # shape: ()
    H_y_given_x_e: np.float16 = -np.sum(
        prob_y_given_context_and_entity * np.nan_to_num(np.log(prob_y_given_context_and_entity)), axis=1
    )  # shape: (|X|,)
    persuasion_scores: np.ndarray = H_y_given_e - H_y_given_x_e  # shape: (|X|,)
    return persuasion_scores


def p_score_kl(prob_y_given_e, prob_y_given_context_and_entity):
    """
    Defining p_score as:
      \sum_{y \in Y} p(y | X=x, E=e) * log(p(y | X=x, E=e) / p(y | E=e) # noqa
    Return shape: (|X|,)
    """
    log_prob_ratio = np.nan_to_num(np.log(prob_y_given_context_and_entity / prob_y_given_e))  # shape: (|X|, |Y|)
    persuasion_scores: np.ndarray = np.sum(prob_y_given_context_and_entity * log_prob_ratio, axis=1)  # shape: (|X|,)
    return persuasion_scores


def estimate_cmi(
    query: str,
    entity: str,
    contexts: List[str],
    model,
    tokenizer,
    answer_map: Dict[int, List[str]] = None,
    bs: int = 32,
    answer_entity: Optional[str] = None,
) -> Tuple[float, float]:
    """
    (1) Computes the conditional mutual information I(X; Y | q[e]) of answer Y and context X when conditioned on query regarding entity e.

    I(X; Y | q[e]) = \sum_{x \in X} \sum_{y \in Y} (p(x, y | q[e]) * log(p(y | x, q[e]) / p(y | q[e]))) # noqa: W605

    So we need to monte carlo estimate:
        (1) p(y | x, q[e])                                             , shape: (|X|, |Y|)
        (2) p(x | q[e])                                                , shape: (|X|,)
        (3) p(x, y | q[e]) = p(y | x, q[e]) * p(x | q[e])              , shape: (|X|, |Y|)
        (4) p(y | q[e]) = \sum_{x \in X} (p(y | x, q[e]) * p(x | q[e])), shape: (|Y|,) # noqa: W605


    (2) Furthermore, computes the half pointwise conditional MI I(Y; X=x | q[e]).

    I(Y; X=x | q[e]) = H(Y | q[e]) - H(Y | X=x, q[e])
                     = - \sum_{y \in Y} (p(y | q[e]) * log(p(y | q[e]))) + \sum_{y \in Y} (p(y | x, q[e]) * log(p(y | x, q[e])))

    Args:
        entity: str - the entity of interest
        contexts: List[str] - list of contexts prepended to the query
        model - the model to use for scoring
        tokenizer - the tokenizer to use for tokenizing the contexts and query
        answer_map - dict from the answer support (as an int) to the tokens which qualify into the respective answer.
        bs - batch size to use for scoring the model
        answer_entity - the entity representing the "answer" to a question. Formatted into closed queries.

    Returns:
        sus_score - the susceptibility score for the given entity
        persuasion_scores - the persuasion scores each context for the given entity
    """
    contexts_counter = Counter(contexts)
    contexts_set = sorted(list(set(contexts)))

    prob_x_given_e = estimate_prob_x_given_e(entity, contexts_set, contexts_counter=contexts_counter)  # shape: (|X|,)
    prob_y_given_context_and_entity = estimate_prob_y_given_context_and_entity(
        query,
        entity,
        contexts_set,
        model,
        tokenizer,
        answer_map=answer_map,
        bs=bs,
        answer_entity=answer_entity,
    )  # shape: (|X|, |Y|)

    prob_x_y_given_e = np.einsum("ij, i -> ij", prob_y_given_context_and_entity, prob_x_given_e)  # shape: (|X|, |Y|)
    prob_y_given_e = np.einsum("ij, i -> j", prob_y_given_context_and_entity, prob_x_given_e)  # shape: (|Y|,)

    sus_score: np.float16 = np.sum(
        prob_x_y_given_e * np.nan_to_num(np.log(prob_y_given_context_and_entity / prob_y_given_e))
    )  # shape: ()

    # Two possible views of persuasion score: should it be H(Y | q[e]) - H(Y | X=x, q[e]) OR \sum_{y \in Y} p(y | X=x, q[e]) * log(p(y | X=x, q[e])/p(y | q[e]))?
    persuasion_scores_ent_diff = p_score_ent_diff(prob_y_given_e, prob_y_given_context_and_entity)  # shape: (|X|,)
    persuasion_scores_kl = p_score_kl(prob_y_given_e, prob_y_given_context_and_entity)  # shape: (|X|,)

    def p_scores_per_context(p_scores, dtype=np.float64):
        context_to_pscore = {context: score for context, score in zip(contexts_set, p_scores.astype(dtype))}
        persuasion_scores: List[float] = [context_to_pscore[context] for context in contexts]  # shape: (len(contexts),)
        return persuasion_scores

    persuasion_scores_ent_diff = p_scores_per_context(persuasion_scores_ent_diff)
    persuasion_scores_kl = p_scores_per_context(persuasion_scores_kl)

    return sus_score, persuasion_scores_ent_diff, persuasion_scores_kl


def compute_memorization_ratio(
    query: str,
    entity: str,
    contexts: List[str],
    model,
    tokenizer,
    context_template: str,
    bs: int = 32,
    answer_entity: Optional[str] = None,
) -> Tuple[float, List[int], List[str]]:
    """
    Computes the memorization ratio of the model for the given entity.
    The memorization ratio, as defined by Longpre et al 2022 (https://arxiv.org/pdf/2109.05052.pdf) is defined as:

    MR = # of original answers / (# of original answers + # of context answers)

    Args:
        entity: str - the entity of interest
        contexts: List[str] - list of contexts prepended to the query.
        model - the model to use for scoring
        tokenizer - the tokenizer to use for tokenizing the contexts and query
        answer_map - dict from the answer support (as an int) to the tokens which qualify into the respective answer.
        bs - batch size to use for scoring the model
        answer_entity - the entity representing the "answer" to a question. Formatted into closed queries.

    Returns:
        mr - the memorization ratio for the given entity
        og_or_ctx_answers - whether the model output matches the original/context/other answer for each context for the given entity
        outputs - the model's outputs for each context for the given entity
    """
    contexts_set = sorted(list(set(contexts)))

    output_tokens: torch.LongTensor = sample_y_given_x_and_entity(
        query=query,
        entity=entity,
        contexts=contexts_set,
        model=model,
        tokenizer=tokenizer,
        num_samples=1,
        bs=bs,
        answer_entity=answer_entity,
    ).long()

    outputs: List[str] = tokenizer.batch_decode(output_tokens)  # shape: (len(contexts),)
    context_answers: List[str] = [
        extract_answer(context_template=context_template, sentence=c) for c in contexts
    ]  # shape: (len(contexts),)

    og_or_ctx_answers: List[AnswerGroup] = [
        is_answer_original_or_context(
            output,
            "Yes" if "{answer}" in query else answer_entity,
            "No" if "{answer}" in query else context_answers[i],
        )
        for i, output in enumerate(outputs)
    ]  # TODO: fix this for when we add closed queries with the expected answer being No.

    answergroup_counts = pd.Series(og_or_ctx_answers).value_counts()
    og_counts = answergroup_counts.get(AnswerGroup.ORIGINAL, default=0)
    ctx_counts = answergroup_counts.get(AnswerGroup.CONTEXT, default=0)

    mr = og_counts / (og_counts + ctx_counts) if og_counts + ctx_counts != 0 else 0

    return mr, [x.value for x in og_or_ctx_answers], outputs


def kl_div(p, q):
    return sst.entropy(p, q, axis=1)


def difference(p, q):
    p_prob = torch.nn.functional.softmax(torch.tensor(p, dtype=torch.float), dim=1)
    q_prob = torch.nn.functional.softmax(torch.tensor(q, dtype=torch.float), dim=1)
    return ((p_prob[:, 1] - p_prob[:, 0]) - (q_prob[:, 1] - q_prob[:, 0])).detach().cpu().numpy()


def difference_abs_val(p, q):
    p_prob = torch.nn.functional.softmax(torch.tensor(p, dtype=torch.float), dim=1)
    q_prob = torch.nn.functional.softmax(torch.tensor(q, dtype=torch.float), dim=1)
    return torch.abs((p_prob[:, 1] - p_prob[:, 0]) - (q_prob[:, 1] - q_prob[:, 0])).detach().cpu().numpy()


def difference_p_good_only(p, q):
    p_prob = torch.nn.functional.softmax(torch.tensor(p, dtype=torch.float), dim=1)
    q_prob = torch.nn.functional.softmax(torch.tensor(q, dtype=torch.float), dim=1)
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


# if __name__ == "__main__":
#     query = "On a scale from 1 to 5 stars, the quality of this movie, '{}', is rated "
#     entity = "The Dark Knight"
#     contexts = [
#         "Here's a movie review: 'The movie was terrific and I loved it'.",
#         "Here's a movie review: 'The movie was awful and I hated it'.",
#     ]
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model_name = "EleutherAI/pythia-70m-deduped"

#     model = GPTNeoXForCausalLM.from_pretrained(
#         model_name,
#     ).to(device)

#     tokenizer = AutoTokenizer.from_pretrained(
#         model_name,
#         padding_side="left",
#     )

#     query = "On a scale from 1 to 5 stars, the quality of this movie, '{}', is rated "
#     print(estimate_cmi(query, entity, contexts, model, tokenizer))
#     print(estimate_entity_score(query, entity, contexts, model, tokenizer, distance_metric=kl_div, answer_map=None))
