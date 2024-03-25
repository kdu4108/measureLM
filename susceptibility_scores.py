import argparse
from ast import literal_eval
import gc
import json
import random
import os
import sys
from tqdm import tqdm
import yaml

import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import wandb

from utils import construct_paths_and_dataset_kwargs, construct_artifact_name
from measuring.estimate_probs import compute_memorization_ratio, estimate_cmi
from preprocessing.datasets import CountryCapital, FriendEnemy, WorldLeaders, YagoECQ, EntityContextQueryDataset
from preprocessing.utils import format_query

from dotenv import load_dotenv

load_dotenv()
hf_token = os.environ.get("HF_TOKEN")


def load_model_and_tokenizer(model_id, load_in_8bit, device, compile_torch=False):
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            load_in_8bit=load_in_8bit,
            torch_dtype=torch.float16 if not load_in_8bit else None,
            token=hf_token,
            device_map="auto",
        )
    except:  # noqa: E722
        print(f"Failed to load model {model_id} in 8-bit. Attempting to load normally.")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            token=hf_token,
            device_map="auto",
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        padding_side="left",
    )

    torch.cuda.empty_cache()
    gc.collect()

    if compile_torch:
        # TODO: debug why this appears not to work.
        model = torch.compile(model)

    return model, tokenizer


def get_args():
    parser = argparse.ArgumentParser(description="Arguments for computing susceptibility scores.")
    parser.add_argument("DATASET_NAME", type=str, help="Name of the dataset class")
    parser.add_argument(
        "-P", "--RAW_DATA_PATH", type=str, default="data/YagoECQ/yago_qec.json", help="Path to the raw data"
    )
    parser.add_argument("-S", "--SEED", type=int, default=0, help="Random seed")
    parser.add_argument(
        "-M",
        "--MODEL_ID",
        type=str,
        default="EleutherAI/pythia-6.9b-deduped",
        help="Name of the model to use from huggingface",
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
        "-ET",
        "--ENTITY_TYPES",
        type=json.loads,
        default=["entities", "gpt_fake_entities"],
        help="Entity types to use (e.g., a subset of ['entities', 'gpt_fake_entities'])",
    )
    parser.add_argument(
        "-QT",
        "--QUERY_TYPES",
        type=json.loads,
        default=["closed", "open"],
        help="Query types to use (e.g., a subset of ['closed', 'open'])",
    )
    parser.add_argument(
        "-CT",
        "--CONTEXT_TYPES",
        type=json.loads,
        default=["base"],
        help="Context types to use (e.g., a subset of ['assertive', 'base', 'negation'])",
    )
    parser.add_argument(
        "-AM", "--ANSWER_MAP", type=json.loads, default=dict(), help="answer map from int to list of ints"
    )
    parser.add_argument(
        "-MR",
        "--COMPUTE_MR",
        action="store_true",
        help="Whether to compute the memorization ratio",
    )
    parser.add_argument("-BS", "--BATCH_SIZE", type=int, default=32, help="Batch size for inference")
    parser.add_argument(
        "-O",
        "--OVERWRITE",
        action="store_true",
        help="Whether to overwrite existing results and recompute susceptibility scores",
    )
    return parser.parse_args()


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
    CONTEXT_TYPES = args.CONTEXT_TYPES
    ANSWER_MAP = {int(k): v for k, v in args.ANSWER_MAP.items()} if args.ANSWER_MAP else None
    COMPUTE_MR = args.COMPUTE_MR
    COMPUTE_P_SCORE_KL = True

    # Model parameters
    BATCH_SZ = args.BATCH_SIZE

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
        mr_results_path,
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
        CONTEXT_TYPES=CONTEXT_TYPES,
        ANSWER_MAP=ANSWER_MAP,
        verbose=True,
    )
    # GPU stuff
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # wandb stuff
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
    dataset_df = dataset.get_contexts_per_query_entity_df()

    # After loading/preprocessing your dataset, log it as an artifact to W&B
    print(f"Saving datasets and run config to {input_dir}.")
    os.makedirs(input_dir, exist_ok=True)
    dataset_df.to_csv(val_data_path)
    with open(os.path.join(input_dir, "config.yml"), "w") as yaml_file:
        yaml.dump(DATASET_KWARGS_IDENTIFIABLE, yaml_file, default_flow_style=False)

    # if LOG_DATASETS:
    #     print(f"Logging datasets to w&b run {wandb.run}.")
    #     artifact = wandb.Artifact(name=data_id, type="dataset")
    #     artifact.add_dir(local_path=data_dir)
    #     run.log_artifact(artifact)

    model, tokenizer = None, None

    # Try reading in val_df_contexts_per_qe from disk in case we need to separate out the mr into its own df from there
    try:
        print("Attempting to load cached sus score results from disk.")
        val_df_contexts_per_qe = pd.read_csv(
            val_results_path,
            index_col=0,
            converters={
                "contexts": literal_eval,  # required because this is a list of strings and shouldn't be read in as a string
                "entity": literal_eval,  # required because this is a tuple of strings and shouldn't be read in as a string
                "answer": str,  # required because pd will automatically try to convert this column to float when it can (which is the case for number-answers, e.g. luminosity)
            },
        )
        print("\tSuccessfully loaded cached sus score results from disk.")
    except FileNotFoundError:
        print("\tNo cached results found for susceptibility score results. Continuing.")
        val_df_contexts_per_qe = dataset_df.copy()

    try:
        print("Attempting to load cached MR results from disk.")
        mr_per_qe_df = pd.read_csv(
            mr_results_path,
            index_col=0,
            converters={
                "contexts": literal_eval,  # required because this is a list of strings and shouldn't be read in as a string
                "entity": literal_eval,  # required because this is a tuple of strings and shouldn't be read in as a string
                "answer": str,  # required because pd will automatically try to convert this column to float when it can (which is the case for number-answers, e.g. luminosity)
            },
        )
        print("\tSuccessfully loaded cached MR results from disk.")
    except FileNotFoundError:
        print(
            "\tNo cached results found for memorization ratio results at `mr_results_path`. Attempting to construct from val_df_contexts_per_qe."
        )
        if "sampled_mr" in val_df_contexts_per_qe.columns:
            print(
                f"\t\tConstructing memorization ratio results from val_df_contexts_per_qe and saving to disk at {mr_results_path}."
            )
            mr_per_qe_df = val_df_contexts_per_qe[
                [
                    "q_id",
                    "query_form",
                    "entity",
                    "answer",
                    "contexts",
                    "sampled_mr",
                    "sampled_answergroups",
                    "sampled_outputs",
                ]
            ]
            mr_per_qe_df.to_csv(mr_results_path)
            val_df_contexts_per_qe = val_df_contexts_per_qe.drop(
                columns=["sampled_mr", "sampled_answergroups", "sampled_outputs"]
            )
            val_df_contexts_per_qe.to_csv(val_results_path)
        else:
            print("\t\tFailed to construct memorization ratio results from val_df_contexts_per_qe.")
            mr_per_qe_df = dataset_df.copy()

    # Sanity check that dataset_df matches val_df_contexts_per_qe and mr_per_qe_df
    if mr_per_qe_df[dataset_df.columns].equals(dataset_df):
        print("mr_per_qe_df matches dataset_df.")
    else:
        if OVERWRITE:
            print("Overwriting mr_per_qe_df to be dataset_df.")
            mr_per_qe_df = dataset_df
        else:
            raise ValueError("mr_per_qe_df does not match dataset_df.")

    if val_df_contexts_per_qe[dataset_df.columns].equals(dataset_df):
        print("val_df_contexts_per_qe matches dataset_df.")
    else:
        print(val_df_contexts_per_qe[dataset_df.columns] == dataset_df)
        print(val_df_contexts_per_qe[dataset_df.columns])
        print(dataset_df)
        if OVERWRITE:
            print("Overwriting val_df_contexts_per_qe to be dataset_df.")
            val_df_contexts_per_qe = dataset_df
        else:
            raise ValueError("val_df_contexts_per_qe does not match dataset_df.")

    if (
        not os.path.exists(val_results_path)
        or (COMPUTE_P_SCORE_KL and "persuasion_scores_kl" not in val_df_contexts_per_qe.columns)
        or OVERWRITE
    ):
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
        val_df_contexts_per_qe["persuasion_scores_kl"] = val_df_contexts_per_qe[
            "sus_score_and_persuasion_scores"
        ].apply(lambda x: x[2])
        val_df_contexts_per_qe["full_query_example"] = val_df_contexts_per_qe.progress_apply(
            lambda row: format_query(
                query=row["query_form"], entity=row["entity"], context=row["contexts"][0], answer=row["answer"]
            ),
            axis=1,
        )
        val_df_contexts_per_qe.drop(columns=["sus_score_and_persuasion_scores"], inplace=True)
        val_df_contexts_per_qe.to_csv(val_results_path)
    else:
        print("All cached sus score results already on disk.")
        # val_df_contexts_per_qe = pd.read_csv(
        #     val_results_path,
        #     index_col=0,
        #     converters={"contexts": literal_eval, "entity": literal_eval, "persuasion_scores": literal_eval},
        # )

    if COMPUTE_MR and ("sampled_mr" not in mr_per_qe_df.columns or OVERWRITE):
        print("Computing memorization ratio results.")
        if model is None or tokenizer is None:
            model, tokenizer = load_model_and_tokenizer(MODEL_ID, LOAD_IN_8BIT, device)

        tqdm.pandas()
        mr_per_qe_df["mr_and_answers_and_outputs"] = mr_per_qe_df.progress_apply(
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
        mr_per_qe_df["sampled_mr"] = mr_per_qe_df["mr_and_answers_and_outputs"].apply(lambda x: x[0])
        mr_per_qe_df["sampled_answergroups"] = mr_per_qe_df["mr_and_answers_and_outputs"].apply(lambda x: x[1])
        mr_per_qe_df["sampled_outputs"] = mr_per_qe_df["mr_and_answers_and_outputs"].apply(lambda x: x[2])
        mr_per_qe_df.drop(columns=["mr_and_answers_and_outputs"], inplace=True)
        mr_per_qe_df.to_csv(mr_results_path)
    else:
        if not COMPUTE_MR:
            print("COMPUTE_MR is False, skipping computing MR.")
        else:
            print("MR already computed in `mr_results_path` cached on disk.")
        # val_df_contexts_per_qe = pd.read_csv(
        #     val_results_path,
        #     index_col=0,
        #     converters={
        #         "contexts": literal_eval,
        #         "entity": literal_eval,
        #         "persuasion_scores": literal_eval,
        #         "sampled_mr": literal_eval,
        #         "sampled_answergroups": literal_eval,
        #         "sampled_outputs": literal_eval,
        #     },
        # )

    # After loading/preprocessing your dataset, log it as an artifact to W&B
    if LOG_DATASETS:
        print(f"Logging results to w&b run {wandb.run}.")
        artifact_name = construct_artifact_name(data_id, SEED, model_id)
        artifact = wandb.Artifact(name=artifact_name, type="val_df_contexts_per_qe")
        artifact.add_dir(local_path=results_dir)
        run.log_artifact(artifact)

    wandb.finish()


if __name__ == "__main__":
    main()
