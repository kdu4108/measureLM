import os
import hashlib
import json
from typing import Optional, List, Dict
from requests.exceptions import HTTPError

import wandb

from preprocessing.utils import extract_name_from_yago_uri


def construct_artifact_name(data_id, SEED, model_id, prefix=""):
    artifact_name = f"{data_id}-{SEED}-{model_id}".replace("/", ".")
    artifact_name = prefix + hashlib.sha256(artifact_name.encode()).hexdigest()[:8]
    return artifact_name


def load_artifact_from_wandb(artifact_name: str, save_dir: str, verbose=True) -> Optional[str]:
    # Load artifact if it exists.
    # This will try to download the entire directory of the data_id,
    #  including ALL logged subdirectories (e.g. the qa_triples, tokenized_qa_triples, and encoded_qa_triples)
    #  in addition to the dataframe pickles.
    artifact, artifact_files = None, None
    try:
        artifact = wandb.run.use_artifact(f"{artifact_name}:latest")
        artifact.download(root=save_dir)
        artifact_files = [file.path for file in artifact.manifest.entries.values()]
        if verbose:
            print("Downloading artifact files:", artifact_files)
    except (wandb.CommError, HTTPError) as e:
        # HTTPError occurs when the artifact has been deleted, this appears to be a bug with wandb. Catching this exception for now.
        print(type(e), e)

    return artifact, artifact_files


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
    CONTEXT_TYPES: List[str],
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
            **{
                "query_id": QUERY_ID,
                "subname": SUBNAME,
                "entity_types": ENTITY_TYPES,
                "query_types": QUERY_TYPES,
                "context_types": CONTEXT_TYPES,
            },
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
        "-CT_" + "_".join(sorted(DATASET_KWARGS_IDENTIFIABLE["context_types"]))
        if (
            "context_types" in DATASET_KWARGS_IDENTIFIABLE
            and DATASET_KWARGS_IDENTIFIABLE["context_types"]
            and DATASET_KWARGS_IDENTIFIABLE["context_types"] != ["base"]
        )
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
    mr_results_path = os.path.join(results_dir, "mr.csv")

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
        mr_results_path,
        data_id,
        model_id,
        DATASET_KWARGS_IDENTIFIABLE,
    )
