from ast import literal_eval
from typing import List, Dict
import os
import hashlib

import pandas as pd
import numpy as np
import random
import torch
import wandb

from susceptibility_scores import construct_paths_and_dataset_kwargs
from utils import load_artifact_from_wandb


def construct_df_given_query_id(
    yago_qec: dict,
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
    ENTITY_TYPES: List[str],
    QUERY_TYPES: List[str],
    ANSWER_MAP: Dict[int, List[str]],
    convert_cols=["contexts", "entity", "persuasion_scores", "relevant_context_inds"],
    verbose=False,
    overwrite_df=False,
) -> pd.DataFrame:
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
        UNIFORM_CONTEXTS=UNIFORM_CONTEXTS,
        DEDUPLICATE_ENTITIES=DEDUPLICATE_ENTITIES,
        ENTITY_SELECTION_FUNC_NAME=ENTITY_SELECTION_FUNC_NAME,
        ABLATE_OUT_RELEVANT_CONTEXTS=ABLATE_OUT_RELEVANT_CONTEXTS,
        OVERWRITE=False,
        ENTITY_TYPES=ENTITY_TYPES,
        QUERY_TYPES=QUERY_TYPES,
        ANSWER_MAP=ANSWER_MAP,
    )
    # Analysis dir
    analysis_dir = os.path.join(data_dir, "analysis")
    save_path = os.path.join(analysis_dir, "val_df_per_qe.csv")

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)

    if verbose:
        print(f"Analysis dir: {analysis_dir}")

    if wandb.run is not None and not os.path.exists(save_path):
        # TODO: this might result in bugs if we need to recompute.
        # Download val_df_per_qe from wandb (if not already cached there)
        artifact_name = f"{data_id}-{model_id}-val_df_per_qe".replace("/", ".")
        artifact_name = f"val_df_per_qe-{hashlib.sha256(artifact_name.encode()).hexdigest()[:8]}"
        artifact, files = load_artifact_from_wandb(artifact_name, save_dir=analysis_dir, verbose=verbose)

    if not overwrite_df and os.path.exists(save_path):
        if verbose:
            print(f"Loading val_df_per_qe for {QUERY_ID} from cache.")
        # print(f"Loading val_df_per_qe for {QUERY_ID} from cache at {save_path}.")
        val_df_per_qe = pd.read_csv(save_path, converters={col: literal_eval for col in convert_cols})
    else:
        print(f"Computing val_df_per_qe for {QUERY_ID}.")

        # Set random seeds
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)

        try:
            val_df_contexts_per_qe = pd.read_csv(
                val_results_path,
                index_col=0,
                converters={col: literal_eval for col in convert_cols},
            )
        except FileNotFoundError as e:
            print(f"Unable to find file at {val_results_path}, full error: {e}")
            return None
        if verbose:
            print("val_df_contexts_per_qe info:")
            print(val_df_contexts_per_qe.info())
            print(val_df_contexts_per_qe["entity"].value_counts())
            print(val_df_contexts_per_qe.iloc[0]["contexts"][:10])

        entities_df_tidy = pd.DataFrame(
            [
                (k, ent)
                for k, ents in yago_qec[QUERY_ID].items()
                for ent in ents
                if k in {"entities", "fake_entities", "gpt_fake_entities"}
            ],
            columns=["type", "entity"],
        )
        real_ents = set(entities_df_tidy[entities_df_tidy["type"] == "entities"]["entity"])
        fake_ents = set(entities_df_tidy[entities_df_tidy["type"] == "gpt_fake_entities"]["entity"])
        entities_df_tidy["entity"] = entities_df_tidy["entity"].apply(lambda x: (x,))
        entities_df_tidy = entities_df_tidy.drop_duplicates(
            ["entity"], keep="first"
        )  # hack: "entities" alphabetically precedes "gpt_fake_entities", so this keeps the duplicate entity as with the "real" category

        if verbose:
            print("Entities info:")
            print("# unique real ents:", len(real_ents))
            print("# unique fake ents:", len(fake_ents))
            print("# overlapping ents:", len(real_ents.intersection(fake_ents)))
            print("Overlapping ents:", real_ents.intersection(fake_ents))
            # print(
            #     entities_df["entities"].value_counts(),
            #     entities_df["fake_entities"].value_counts(),
            # )
            print(entities_df_tidy.head())
            print(entities_df_tidy.info())

        qids_to_ec = pd.DataFrame(
            [(k, v["entity_types"]) for k, v in yago_qec.items()],
            columns=["q_id", "entity_classes"],
        )
        cols = [
            "q_id",
            "query_form",
            "entity",
            "answer",
            "contexts",
            "persuasion_scores",
            "type",
            "susceptibility_score",
        ]
        if "sampled_mr" in val_df_contexts_per_qe:
            cols += ["sampled_mr", "sampled_answergroups", "sampled_outputs"]
            convert_cols += ["sampled_answergroups"]

        val_df_per_qe = val_df_contexts_per_qe.merge(
            entities_df_tidy,
            left_on="entity",
            right_on="entity",
            how="left",
        )[cols]
        val_df_per_qe = val_df_per_qe.merge(qids_to_ec)  # add entity types

        query_forms = val_df_per_qe["query_form"].unique()
        closed_qfs = yago_qec[QUERY_ID]["query_forms"]["closed"]
        open_qfs = yago_qec[QUERY_ID]["query_forms"]["open"]

        val_df_per_qe.loc[val_df_per_qe["query_form"].isin(closed_qfs), "query_type"] = "closed"
        val_df_per_qe.loc[val_df_per_qe["query_form"].isin(open_qfs), "query_type"] = "open"

        val_df_per_qe["relevant_context_inds"] = val_df_per_qe.apply(
            lambda row: [i for i, context in enumerate(row["contexts"]) if row["entity"][0] in context],
            axis=1,
        )
        if verbose:
            print("val_df_per_qe info:")
            print(val_df_per_qe)

            print(f"query forms: {query_forms}")

        val_df_per_qe.to_csv(save_path, index=False)

        if wandb.run is not None:
            artifact = wandb.Artifact(name=artifact_name, type="resultsdf")
            artifact.add_file(local_path=save_path)
            if verbose:
                print(f"Logging artifact: {artifact.name}.")
                # print(f"Logging artifact: {artifact.name} containing {save_path}")
            wandb.run.log_artifact(artifact)

    return val_df_per_qe


def permutation_test(persuasion_scores: List[float], relevant_context_inds: List[int], k=100, alpha=0.95):
    persuasion_scores = np.array(persuasion_scores)

    relevant_p_scores = persuasion_scores[relevant_context_inds]
    irrelevant_p_scores = np.delete(persuasion_scores, relevant_context_inds)

    irrelevant_samples = np.random.choice(irrelevant_p_scores, size=(k, len(relevant_context_inds)), replace=True)
    irrelevant_sample_means = np.mean(irrelevant_samples, axis=1)
    relevant_p_scores_mean = np.mean(relevant_p_scores)
    relevant_p_score_percentile = (relevant_p_scores_mean > irrelevant_sample_means).mean()

    return relevant_p_score_percentile, relevant_p_score_percentile > alpha


def percent_ents_passing_pscore_permutation_test(val_df_per_qe, query_type="closed"):
    if query_type is not None:
        val_df_per_qe = val_df_per_qe[val_df_per_qe["query_type"] == query_type]

    return (
        np.array(
            [
                permutation_test(
                    val_df_per_qe.iloc[i]["persuasion_scores"],
                    val_df_per_qe.iloc[i]["relevant_context_inds"],
                )[0]
                for i in range(len(val_df_per_qe))
            ]
        ),
        np.array(
            [
                permutation_test(
                    val_df_per_qe.iloc[i]["persuasion_scores"],
                    val_df_per_qe.iloc[i]["relevant_context_inds"],
                )[1]
                for i in range(len(val_df_per_qe))
            ]
        ).mean(),
    )
