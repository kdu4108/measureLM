# RUN FROM ROOT DIRECTORY: `python -m analysis.wandb_download_data`
from tqdm import tqdm
import json
import yaml
import wandb
from pathlib import Path
from analysis.analysis_helpers import load_val_df_from_wandb
from utils import load_artifact_from_wandb

CONFIG_PATH = "analysis/config-s11.yml"
with open(CONFIG_PATH) as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# Data parameters
DATASET_NAME = config["DATASET_NAME"]
RAW_DATA_PATH = config["RAW_DATA_PATH"]
SEED = config["SEED"]
MODEL_ID = config["MODEL_ID"]
LOAD_IN_8BIT = config["LOAD_IN_8BIT"]
MAX_CONTEXTS = config["MAX_CONTEXTS"]
MAX_ENTITIES = config["MAX_ENTITIES"]
CAP_PER_TYPE = config["CAP_PER_TYPE"]
ABLATE_OUT_RELEVANT_CONTEXTS = config["ABLATE_OUT_RELEVANT_CONTEXTS"]
UNIFORM_CONTEXTS = config["UNIFORM_CONTEXTS"]
DEDUPLICATE_ENTITIES = config["DEDUPLICATE_ENTITIES"]
ENTITY_SELECTION_FUNC_NAME = config["ENTITY_SELECTION_FUNC_NAME"]
ENTITY_TYPES = config["ENTITY_TYPES"]
QUERY_TYPES = config["QUERY_TYPES"]
CONTEXT_TYPES = config["CONTEXT_TYPES"]
ANSWER_MAP = config["ANSWER_MAP"]

DATASET_KWARGS = dict(
    DATASET_NAME=DATASET_NAME,
    RAW_DATA_PATH=RAW_DATA_PATH,
    SEED=SEED,
    MODEL_ID=MODEL_ID,
    LOAD_IN_8BIT=LOAD_IN_8BIT,
    MAX_CONTEXTS=MAX_CONTEXTS,
    MAX_ENTITIES=MAX_ENTITIES,
    CAP_PER_TYPE=CAP_PER_TYPE,
    ABLATE_OUT_RELEVANT_CONTEXTS=ABLATE_OUT_RELEVANT_CONTEXTS,
    UNIFORM_CONTEXTS=UNIFORM_CONTEXTS,
    DEDUPLICATE_ENTITIES=DEDUPLICATE_ENTITIES,
    ENTITY_SELECTION_FUNC_NAME=ENTITY_SELECTION_FUNC_NAME,
    ENTITY_TYPES=ENTITY_TYPES,
    CONTEXT_TYPES=CONTEXT_TYPES,
    QUERY_TYPES=QUERY_TYPES,
    ANSWER_MAP=ANSWER_MAP,
)

# wandb stuff
PROJECT_NAME = "context-vs-bias"
GROUP_NAME = None
TAGS = ["download"]
LOG_DATASETS = True

params_to_log = {k: v for k, v in locals().items() if k.isupper()}

run = wandb.init(
    project=PROJECT_NAME,
    group=GROUP_NAME,
    entity="ethz-rycolab",
    config=params_to_log,
    tags=TAGS,
)
print(dict(wandb.config))

artifact, files = load_artifact_from_wandb(f"{DATASET_NAME}-yago_qec", save_dir=Path(RAW_DATA_PATH).parent)
print(artifact.name)
with open(RAW_DATA_PATH) as f:
    yago_qec = json.load(f)

query_ids = list(yago_qec.keys())
for qid, v in list(yago_qec.items())[:10]:
    print(qid, len(v["entities"]), len(set(v["entities"])))

# Download results for all model sizes (if they exist)
model_id_and_quantize_tuples = [
    ("EleutherAI/pythia-70m-deduped", False, 32),
    ("EleutherAI/pythia-410m-deduped", False, 32),
    ("EleutherAI/pythia-1.4b-deduped", False, 32),
    ("EleutherAI/pythia-2.8b-deduped", False, 16),
    ("EleutherAI/pythia-6.9b-deduped", False, 8),
    ("EleutherAI/pythia-12b-deduped", True, 8),
]
for model_id, load8bit, _ in tqdm(model_id_and_quantize_tuples):
    qid_to_results_paths = {
        query_id: load_val_df_from_wandb(
            **{**DATASET_KWARGS, **dict(MODEL_ID=model_id, LOAD_IN_8BIT=load8bit)},
            QUERY_ID=query_id,
            verbose=False,
            overwrite_df=True,
            yago_qec=yago_qec,
        )
        # query_id: construct_df_given_query_id(query_id, convert_cols=["entity"], verbose=False)
        for query_id in tqdm(query_ids)
    }
