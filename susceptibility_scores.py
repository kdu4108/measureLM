import argparse
import gc
import os
import sys
import random
from tqdm import tqdm

from transformers import GPTNeoXForCausalLM, AutoTokenizer
import torch
import numpy as np
import plac
import wandb

from measuring.estimate_probs import estimate_cmi
from preprocessing.datasets import CountryCapital, FriendEnemy, WorldLeaders, YagoECQ
from preprocessing.utils import extract_name_from_yago_uri


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


def get_args():
    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument("DATASET_NAME", type=str, help="Name of the dataset class")
    parser.add_argument(
        "-P", "--RAWDATAPATH", type=str, default="data/YagoECQ/yago_qec.json", help="Path to the raw data"
    )
    parser.add_argument("-S", "--SEED", type=int, default=0, help="Random seed")
    parser.add_argument(
        "-M", "--MODELID", type=str, default="EleutherAI/pythia-6.9b-deduped", help="Name of the model to use"
    )
    parser.add_argument("-B", "--LOADIN8BIT", action="store_true", help="Whether to load in 8 bit")
    parser.add_argument("-Q", "--QUERYID", type=str, help="Name of the query id, if using YagoECQ dataset")
    parser.add_argument("-MC", "--MAXCONTEXTS", type=int, default=450, help="Max number of contexts in dataset")
    parser.add_argument("-ME", "--MAXENTITIES", type=int, default=90, help="Max number of entities in dataset")
    parser.add_argument("-T", "--CAPPERTYPE", action="store_true", help="Whether to cap per type")
    parser.add_argument(
        "-A", "--ABLATEOUTRELEVANTCONTEXTS", action="store_true", help="Whether to ablate out relevant contexts"
    )
    parser.add_argument(
        "-O",
        "--OVERWRITE",
        action="store_true",
        help="Whether to overwrite existing results and recompute susceptibility scores",
    )
    parser.add_argument("-ET", "--ENTITYTYPES", type=str, nargs="+", default=["entities"], help="Entity types to use")
    parser.add_argument("-QT", "--QUERYTYPES", type=str, nargs="+", default=["closed"], help="Query types to use")
    return parser.parse_args()


def main():
    args = get_args()
    DATASET_NAME = args.DATASET_NAME
    RAWDATAPATH = args.RAWDATAPATH
    SEED = args.SEED
    MODELID = args.MODELID
    LOADIN8BIT = args.LOADIN8BIT
    QUERYID = args.QUERYID
    MAXCONTEXTS = args.MAXCONTEXTS
    MAXENTITIES = args.MAXENTITIES
    CAPPERTYPE = args.CAPPERTYPE
    ABLATEOUTRELEVANTCONTEXTS = args.ABLATEOUTRELEVANTCONTEXTS
    # OVERWRITE = args.OVERWRITE
    ENTITYTYPES = args.ENTITYTYPES
    QUERYTYPES = args.QUERYTYPES

    # Set random seeds
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    SUBNAME = f"{extract_name_from_yago_uri(QUERYID)[0]}_{extract_name_from_yago_uri(QUERYID)[1]}"  # TODO: probably need to fix this
    DATASET_KWARGS_IDENTIFIABLE = dict(
        max_contexts=MAXCONTEXTS,
        max_entities=MAXENTITIES,
        cap_per_type=CAPPERTYPE,
        raw_data_path=RAWDATAPATH,
        ablate_out_relevant_contexts=ABLATEOUTRELEVANTCONTEXTS,
    )
    if DATASET_NAME == "YagoECQ":
        DATASET_KWARGS_IDENTIFIABLE = {
            **DATASET_KWARGS_IDENTIFIABLE,
            **{"query_id": QUERYID, "subname": SUBNAME, "entity_types": ENTITYTYPES, "query_types": QUERYTYPES},
        }

    LOG_DATASETS = True

    # Model parameters
    BATCH_SZ = 16

    # wandb stuff
    PROJECT_NAME = "context-vs-bias"
    GROUP_NAME = None
    TAGS = ["yago"]

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
        "-cappertype"
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
        "-ET_" + "_".join(DATASET_KWARGS_IDENTIFIABLE["entity_types"])
        if "entity_types" in DATASET_KWARGS_IDENTIFIABLE and DATASET_KWARGS_IDENTIFIABLE["entity_types"]
        else ""
    )

    data_id += (
        "-QT_" + "_".join(DATASET_KWARGS_IDENTIFIABLE["query_types"])
        if "query_types" in DATASET_KWARGS_IDENTIFIABLE and DATASET_KWARGS_IDENTIFIABLE["query_types"]
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
    val_data_path = os.path.join(input_dir, "val.csv")

    DATASET_KWARGS_IDENTIFIABLE = {
        **DATASET_KWARGS_IDENTIFIABLE,
        **dict(
            entities_path=entities_path,
            contexts_path=contexts_path,
            queries_path=queries_path,
        ),
    }

    # Construct model id
    model_id = f"{MODELID}"
    model_id += "-8bit" if LOADIN8BIT else ""
    model_dir = os.path.join(data_dir, "models", model_id)

    # Results path
    results_dir = os.path.join(model_dir, "results")
    val_results_path = os.path.join(results_dir, "val.csv")

    print(f"Data dir: {data_dir}")
    print(f"Model dir: {model_dir}")

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    dataset = getattr(sys.modules[__name__], DATASET_NAME)(**DATASET_KWARGS_IDENTIFIABLE)

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

    model, tokenizer = load_model_and_tokenizer(MODELID, LOADIN8BIT, device)

    tqdm.pandas()
    val_df_contexts_per_qe["susceptibility_score"] = val_df_contexts_per_qe.progress_apply(
        lambda row: estimate_cmi(
            query=row["query_form"],
            entity=row["entity"],
            contexts=row["contexts"],
            model=model,
            tokenizer=tokenizer,
            answer_map=None,
            bs=BATCH_SZ,
        ),
        axis=1,
    )
    val_df_contexts_per_qe.to_csv(val_results_path)

    # After loading/preprocessing your dataset, log it as an artifact to W&B
    if LOG_DATASETS:
        print(f"Logging results to w&b run {wandb.run}.")
        artifact = wandb.Artifact(name=data_id, type="dataset")
        artifact.add_dir(local_path=data_dir)
        run.log_artifact(artifact)

    wandb.finish()


if __name__ == "__main__":
    main()
