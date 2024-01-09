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


@plac.pos("DATASET_NAME", "Name of the dataset class", type=str)
@plac.opt("RAWDATAPATH", "Path to the raw data", type=str, abbrev="P")
@plac.opt("SEED", "random seed", type=int, abbrev="S")
@plac.opt("MODELID", "Name of the model to use", type=str, abbrev="M")
@plac.flg("LOADIN8BIT", "whether to load in 8 bit", abbrev="B")
@plac.opt("QUERYID", "Name of the query id, if using YagoECQ dataset", type=str, abbrev="Q")
@plac.opt("MAXCONTEXTS", "Max number of contexts in dataset", type=int, abbrev="MC")
@plac.opt("MAXENTITIES", "Max number of entities in dataset", type=int, abbrev="ME")
@plac.flg("CAPPERTYPE", "whether to cap per type", abbrev="T")
@plac.flg("ABLATEOUTRELEVANTCONTEXTS", "whether to ablate out relevant contexts", abbrev="A")
@plac.flg("OVERWRITE", "whether to overwrite existing results and recompute susceptibility scores", abbrev="O")
def main(
    DATASET_NAME,
    RAWDATAPATH="data/YagoECQ/yago_qec.json",
    SEED=0,
    MODELID="EleutherAI/pythia-6.9b-deduped",
    LOADIN8BIT=False,
    QUERYID=None,
    MAXCONTEXTS=450,
    MAXENTITIES=90,
    CAPPERTYPE=False,
    ABLATEOUTRELEVANTCONTEXTS=False,
    OVERWRITE=False,
):
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
        DATASET_KWARGS_IDENTIFIABLE = {**DATASET_KWARGS_IDENTIFIABLE, **{"query_id": QUERYID, "subname": SUBNAME}}

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
    plac.call(main)
