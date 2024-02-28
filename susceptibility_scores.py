import argparse
from ast import literal_eval
import gc
import hashlib
import json
import random
import os
import sys
from tqdm import tqdm
from typing import List, Dict

import pandas as pd
from transformers import GPTNeoXForCausalLM, AutoTokenizer
import torch
import numpy as np
import wandb

from measuring.estimate_probs import compute_memorization_ratio, estimate_cmi
from preprocessing.datasets import CountryCapital, FriendEnemy, WorldLeaders, YagoECQ, EntityContextQueryDataset
from preprocessing.utils import extract_name_from_yago_uri, format_query


def load_model_and_tokenizer(model_id, load_in_8bit, device):
    try:
        model = GPTNeoXForCausalLM.from_pretrained(model_id, load_in_8bit=load_in_8bit, device_map="auto")
    except:  # noqa: E722
        print(f"Failed to load model {model_id} in 8-bit. Attempting to load normally.")
        model = GPTNeoXForCausalLM.from_pretrained(
            model_id,
            load_in_8bit=False,
        ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        padding_side="left",
    )

    torch.cuda.empty_cache()
    gc.collect()
    return model, tokenizer


def get_context_template(yago_qec: dict, query_id: str):
    # For now, we extract it from the query forms.
    # Probably we would want to extract it from its own key though.
    # TODO: refactor yago_qec to have its own key for context template
    return yago_qec[query_id]["query_forms"]["open"][1]


def get_args():
    parser = argparse.ArgumentParser(description="Arguments for computing susceptibility scores.")
    parser.add_argument("DATASET_NAME", type=str, help="Name of the dataset class")
    parser.add_argument(
        "-P", "--RAW_DATA_PATH", type=str, default="data/YagoECQ/yago_qec.json", help="Path to the raw data"
    )
    parser.add_argument("-S", "--SEED", type=int, default=0, help="Random seed")
    parser.add_argument(
        "-M", "--MODEL_ID", type=str, default="EleutherAI/pythia-6.9b-deduped", help="Name of the model to use"
    )
    parser.add_argument("-B", "--LOAD_IN_8BIT", action="store_true", help="Whether to load in 8 bit")
    parser.add_argument("-Q", "--QUERY_ID", type=str, help="Name of the query id, if using YagoECQ dataset")
    parser.add_argument("-MC", "--MAX_CONTEXTS", type=int, default=450, help="Max number of contexts in dataset")
    parser.add_argument("-ME", "--MAX_ENTITIES", type=int, default=90, help="Max number of entities in dataset")
    parser.add_argument("-T", "--CAP_PER_TYPE", action="store_true", help="Whether to cap per type")
    parser.add_argument(
        "-A", "--ABLATE_OUT_RELEVANT_CONTEXTS", action="store_true", help="Whether to ablate out relevant contexts"
    )
    parser.add_argument("-D", "--DEDUPLICATE_ENTITIES", action="store_true", help="Whether to deduplicate the entities")
    parser.add_argument(
        "-U",
        "--UNIFORM_CONTEXTS",
        action="store_true",
        help="Whether to enforce that each entity is uniformly represented across the contexts",
    )
    parser.add_argument(
        "-ES",
        "--ENTITY_SELECTION_FUNC_NAME",
        type=str,
        default="random_sample",
        help="Name of the entity selection function name. Must be one of the functions in preprocessing.utils",
    )
    parser.add_argument(
        "-O",
        "--OVERWRITE",
        action="store_true",
        help="Whether to overwrite existing results and recompute susceptibility scores",
    )
    parser.add_argument(
        "-ET", "--ENTITY_TYPES", type=json.loads, default=["entities", "gpt_fake_entities"], help="Entity types to use"
    )
    parser.add_argument("-QT", "--QUERY_TYPES", type=json.loads, default=["closed"], help="Query types to use")
    parser.add_argument(
        "-AM", "--ANSWER_MAP", type=json.loads, default=dict(), help="answer map from int to list of ints"
    )
    return parser.parse_args()


def construct_paths_and_dataset_kwargs(
    DATASET_NAME: str,
    RAW_DATA_PATH: str,
    SEED: int,
    MODEL_ID: str,
    LOAD_IN_8BIT: bool,
    QUERY_ID: str,
    MAX_CONTEXTS: int,
    MAX_ENTITIES: int,
    CAP_PER_TYPE: bool,
    ABLATE_OUT_RELEVANT_CONTEXTS: bool,
    DEDUPLICATE_ENTITIES: bool,
    UNIFORM_CONTEXTS: bool,
    ENTITY_SELECTION_FUNC_NAME: str,
    OVERWRITE: bool,
    ENTITY_TYPES: List[str],
    QUERY_TYPES: List[str],
    ANSWER_MAP: Dict[int, List[str]],
    verbose=False,
):
    DATASET_KWARGS_IDENTIFIABLE = dict(
        max_contexts=MAX_CONTEXTS,
        max_entities=MAX_ENTITIES,
        cap_per_type=CAP_PER_TYPE,
        raw_data_path=RAW_DATA_PATH,
        ablate_out_relevant_contexts=ABLATE_OUT_RELEVANT_CONTEXTS,
        uniform_contexts=UNIFORM_CONTEXTS,
        deduplicate_entities=DEDUPLICATE_ENTITIES,
        entity_selection_func_name=ENTITY_SELECTION_FUNC_NAME,
        overwrite=OVERWRITE,
    )
    if DATASET_NAME == "YagoECQ":
        SUBNAME = f"{extract_name_from_yago_uri(QUERY_ID)[0]}_{extract_name_from_yago_uri(QUERY_ID)[1]}"  # TODO: probably need to fix this
        DATASET_KWARGS_IDENTIFIABLE = {
            **DATASET_KWARGS_IDENTIFIABLE,
            **{"query_id": QUERY_ID, "subname": SUBNAME, "entity_types": ENTITY_TYPES, "query_types": QUERY_TYPES},
        }

    # Paths
    # Construct dataset and data ids
    data_id = (
        DATASET_NAME if "subname" not in DATASET_KWARGS_IDENTIFIABLE else f"{DATASET_KWARGS_IDENTIFIABLE['subname']}"
    )
    data_id += (
        f"-mc{DATASET_KWARGS_IDENTIFIABLE['max_contexts']}"
        if "max_contexts" in DATASET_KWARGS_IDENTIFIABLE and DATASET_KWARGS_IDENTIFIABLE["max_contexts"] is not None
        else ""
    )
    data_id += (
        f"-me{DATASET_KWARGS_IDENTIFIABLE['max_entities']}"
        if "max_entities" in DATASET_KWARGS_IDENTIFIABLE and DATASET_KWARGS_IDENTIFIABLE["max_entities"] is not None
        else ""
    )
    data_id += (
        "-capperet"
        if "cap_per_type" in DATASET_KWARGS_IDENTIFIABLE and DATASET_KWARGS_IDENTIFIABLE["cap_per_type"]
        else ""
    )
    data_id += (
        "-ablate"
        if "ablate_out_relevant_contexts" in DATASET_KWARGS_IDENTIFIABLE
        and DATASET_KWARGS_IDENTIFIABLE["ablate_out_relevant_contexts"]
        else ""
    )
    data_id += (
        "-uctxs"
        if "uniform_contexts" in DATASET_KWARGS_IDENTIFIABLE and DATASET_KWARGS_IDENTIFIABLE["uniform_contexts"]
        else ""
    )
    data_id += (
        "-ddpents"
        if "deduplicate_entities" in DATASET_KWARGS_IDENTIFIABLE and DATASET_KWARGS_IDENTIFIABLE["deduplicate_entities"]
        else ""
    )

    data_id += (
        "-ET_" + "_".join(DATASET_KWARGS_IDENTIFIABLE["entity_types"])
        if "entity_types" in DATASET_KWARGS_IDENTIFIABLE and DATASET_KWARGS_IDENTIFIABLE["entity_types"]
        else ""
    )

    data_id += (
        "-QT_" + "_".join(DATASET_KWARGS_IDENTIFIABLE["query_types"])
        if "query_types" in DATASET_KWARGS_IDENTIFIABLE and DATASET_KWARGS_IDENTIFIABLE["query_types"]
        else ""
    )

    data_id += (
        f"-ES_{DATASET_KWARGS_IDENTIFIABLE['entity_selection_func_name']}"
        if "entity_selection_func_name" in DATASET_KWARGS_IDENTIFIABLE
        and DATASET_KWARGS_IDENTIFIABLE["entity_selection_func_name"] is not None
        else ""
    )

    data_id += (
        "-AM_" + hashlib.sha256(json.dumps(ANSWER_MAP, sort_keys=True).encode()).hexdigest()[:8]
        if ANSWER_MAP is not None
        else ""
    )

    data_dir = os.path.join(
        "data",
        DATASET_NAME,
        f"{DATASET_KWARGS_IDENTIFIABLE['subname']}" if "subname" in DATASET_KWARGS_IDENTIFIABLE else "",
        data_id,
        f"{SEED}",
    )
    input_dir = os.path.join(data_dir, "inputs")
    entities_path = os.path.join(input_dir, "entities.json")
    contexts_path = os.path.join(input_dir, "contexts.json")
    queries_path = os.path.join(input_dir, "queries.json")
    answers_path = os.path.join(input_dir, "answers.json")
    val_data_path = os.path.join(input_dir, "val.csv")

    DATASET_KWARGS_IDENTIFIABLE = {
        **DATASET_KWARGS_IDENTIFIABLE,
        **dict(
            entities_path=entities_path,
            contexts_path=contexts_path,
            queries_path=queries_path,
            answers_path=answers_path,
        ),
    }

    # Construct model id
    model_id = f"{MODEL_ID}"
    model_id += "-8bit" if LOAD_IN_8BIT else ""
    model_dir = os.path.join(data_dir, "models", model_id)

    # Results path
    results_dir = os.path.join(model_dir, "results")
    val_results_path = os.path.join(results_dir, "val.csv")

    if verbose:
        print(f"Data dir: {data_dir}")
        print(f"Model dir: {model_dir}")

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    return (
        data_dir,
        input_dir,
        entities_path,
        contexts_path,
        queries_path,
        answers_path,
        val_data_path,
        model_dir,
        results_dir,
        val_results_path,
        data_id,
        model_id,
        DATASET_KWARGS_IDENTIFIABLE,
    )


def main():
    args = get_args()
    DATASET_NAME = args.DATASET_NAME
    RAW_DATA_PATH = args.RAW_DATA_PATH
    SEED = args.SEED
    MODEL_ID = args.MODEL_ID
    LOAD_IN_8BIT = args.LOAD_IN_8BIT
    QUERY_ID = args.QUERY_ID
    MAX_CONTEXTS = args.MAX_CONTEXTS
    MAX_ENTITIES = args.MAX_ENTITIES
    CAP_PER_TYPE = args.CAP_PER_TYPE
    ABLATE_OUT_RELEVANT_CONTEXTS = args.ABLATE_OUT_RELEVANT_CONTEXTS
    UNIFORM_CONTEXTS = args.UNIFORM_CONTEXTS
    DEDUPLICATE_ENTITIES = args.DEDUPLICATE_ENTITIES
    ENTITY_SELECTION_FUNC_NAME = args.ENTITY_SELECTION_FUNC_NAME
    OVERWRITE = args.OVERWRITE
    ENTITY_TYPES = args.ENTITY_TYPES
    QUERY_TYPES = args.QUERY_TYPES
    ANSWER_MAP = {int(k): v for k, v in args.ANSWER_MAP.items()} if args.ANSWER_MAP else None
    COMPUTE_MR = False

    # Model parameters
    BATCH_SZ = 32

    # wandb stuff
    PROJECT_NAME = "context-vs-bias"
    GROUP_NAME = None
    TAGS = ["yago"]
    LOG_DATASETS = True

    # Set random seeds
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # Construct paths from run parameters and construct DATASET_KWARGS_IDENTIFIABLE
    (
        data_dir,
        input_dir,
        entities_path,
        contexts_path,
        queries_path,
        answers_path,
        val_data_path,
        model_dir,
        results_dir,
        val_results_path,
        data_id,
        model_id,
        DATASET_KWARGS_IDENTIFIABLE,
    ) = construct_paths_and_dataset_kwargs(
        DATASET_NAME=DATASET_NAME,
        RAW_DATA_PATH=RAW_DATA_PATH,
        SEED=SEED,
        MODEL_ID=MODEL_ID,
        LOAD_IN_8BIT=LOAD_IN_8BIT,
        QUERY_ID=QUERY_ID,
        MAX_CONTEXTS=MAX_CONTEXTS,
        MAX_ENTITIES=MAX_ENTITIES,
        CAP_PER_TYPE=CAP_PER_TYPE,
        ABLATE_OUT_RELEVANT_CONTEXTS=ABLATE_OUT_RELEVANT_CONTEXTS,
        UNIFORM_CONTEXTS=UNIFORM_CONTEXTS,
        DEDUPLICATE_ENTITIES=DEDUPLICATE_ENTITIES,
        ENTITY_SELECTION_FUNC_NAME=ENTITY_SELECTION_FUNC_NAME,
        OVERWRITE=OVERWRITE,
        ENTITY_TYPES=ENTITY_TYPES,
        QUERY_TYPES=QUERY_TYPES,
        ANSWER_MAP=ANSWER_MAP,
        verbose=True,
    )
    # GPU stuff
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # wandb stuff
    os.environ["WANDB_NOTEBOOK_NAME"] = os.path.join(os.getcwd(), "main.ipynb")

    params_to_log = {k: v for k, v in locals().items() if k.isupper()}

    run = wandb.init(
        project=PROJECT_NAME,
        group=GROUP_NAME,
        config=params_to_log,
        tags=TAGS,
        mode="online",
    )
    print(dict(wandb.config))

    dataset: EntityContextQueryDataset = getattr(sys.modules[__name__], DATASET_NAME)(**DATASET_KWARGS_IDENTIFIABLE)
    val_df_contexts_per_qe = dataset.get_contexts_per_query_entity_df()

    # After loading/preprocessing your dataset, log it as an artifact to W&B
    if LOG_DATASETS:
        print(f"Saving datasets to {input_dir}.")
        os.makedirs(input_dir, exist_ok=True)
        val_df_contexts_per_qe.to_csv(val_data_path)

        print(f"Logging datasets to w&b run {wandb.run}.")
        artifact = wandb.Artifact(name=data_id, type="dataset")
        artifact.add_dir(local_path=data_dir)
        run.log_artifact(artifact)

    model, tokenizer = None, None
    if not os.path.exists(val_results_path) or OVERWRITE:
        print("Computing susceptibility scores.")
        model, tokenizer = load_model_and_tokenizer(MODEL_ID, LOAD_IN_8BIT, device)
        answer_map_tensor = (
            {k: torch.tensor(v, device=model.device) for k, v in ANSWER_MAP.items()} if ANSWER_MAP is not None else None
        )

        tqdm.pandas()
        val_df_contexts_per_qe["sus_score_and_persuasion_scores"] = val_df_contexts_per_qe.progress_apply(
            lambda row: estimate_cmi(
                query=row["query_form"],
                entity=row["entity"],
                contexts=row["contexts"],
                model=model,
                tokenizer=tokenizer,
                answer_map=answer_map_tensor,
                bs=BATCH_SZ,
                answer_entity=row["answer"],
            ),
            axis=1,
        )
        val_df_contexts_per_qe["susceptibility_score"] = val_df_contexts_per_qe[
            "sus_score_and_persuasion_scores"
        ].apply(lambda x: x[0])
        val_df_contexts_per_qe["persuasion_scores"] = val_df_contexts_per_qe["sus_score_and_persuasion_scores"].apply(
            lambda x: x[1]
        )
        val_df_contexts_per_qe["full_query_example"] = val_df_contexts_per_qe.progress_apply(
            lambda row: format_query(
                query=row["query_form"], entity=row["entity"], context=row["contexts"][0], answer=row["answer"]
            ),
            axis=1,
        )
        val_df_contexts_per_qe.drop(columns=["sus_score_and_persuasion_scores"], inplace=True)
        val_df_contexts_per_qe.to_csv(val_results_path)
    else:
        print("Loading cached sus score results from disk.")
        val_df_contexts_per_qe = pd.read_csv(
            val_results_path,
            index_col=0,
            converters={"contexts": literal_eval, "entity": literal_eval, "persuasion_scores": literal_eval},
        )

    if COMPUTE_MR and ("sampled_mr" not in val_df_contexts_per_qe.columns or OVERWRITE):
        print("Computing memorization ratio results.")
        if model is None or tokenizer is None:
            model, tokenizer = load_model_and_tokenizer(MODEL_ID, LOAD_IN_8BIT, device)

        tqdm.pandas()
        val_df_contexts_per_qe["mr_and_answers_and_outputs"] = val_df_contexts_per_qe.progress_apply(
            lambda row: compute_memorization_ratio(
                query=row["query_form"],
                entity=row["entity"],
                contexts=row["contexts"],
                model=model,
                tokenizer=tokenizer,
                context_template=dataset.context_templates[0],
                bs=BATCH_SZ,
                answer_entity=row["answer"],
            ),
            axis=1,
        )
        val_df_contexts_per_qe["sampled_mr"] = val_df_contexts_per_qe["mr_and_answers_and_outputs"].apply(
            lambda x: x[0]
        )
        val_df_contexts_per_qe["sampled_answergroups"] = val_df_contexts_per_qe["mr_and_answers_and_outputs"].apply(
            lambda x: x[1]
        )
        val_df_contexts_per_qe["sampled_outputs"] = val_df_contexts_per_qe["mr_and_answers_and_outputs"].apply(
            lambda x: x[2]
        )
        val_df_contexts_per_qe.drop(columns=["mr_and_answers_and_outputs"], inplace=True)
        val_df_contexts_per_qe.to_csv(val_results_path)
    else:
        print("Loading cached sus score and mr score results from disk.")
        val_df_contexts_per_qe = pd.read_csv(
            val_results_path,
            index_col=0,
            converters={
                "contexts": literal_eval,
                "entity": literal_eval,
                "persuasion_scores": literal_eval,
                "sampled_mr": literal_eval,
                "sampled_answergroups": literal_eval,
                "sampled_outputs": literal_eval,
            },
        )

    # After loading/preprocessing your dataset, log it as an artifact to W&B
    if LOG_DATASETS:
        print(f"Logging results to w&b run {wandb.run}.")
        artifact = wandb.Artifact(name=data_id, type="dataset")
        artifact.add_dir(local_path=data_dir)
        run.log_artifact(artifact)

    wandb.finish()


if __name__ == "__main__":
    main()
