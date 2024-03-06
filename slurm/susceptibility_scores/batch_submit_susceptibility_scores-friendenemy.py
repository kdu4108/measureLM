import subprocess
import json
from typing import List, Dict

from transformers import AutoTokenizer


RUN_LOCALLY = False
RAW_DATA_PATH = "data/FriendEnemy/raw-friend-enemy.csv"

# dataset_names_and_rdps = [("YagoECQ", YAGO_QEC_PATH)]
dataset_names_and_rdps = [("FriendEnemy", RAW_DATA_PATH)]
seeds = [1]

if RUN_LOCALLY:
    model_id_and_quantize_tuples = [("EleutherAI/pythia-70m-deduped", False)]
    max_contexts = [10]
    max_entities = [5]
else:
    model_id_and_quantize_tuples = [("EleutherAI/pythia-6.9b-deduped", True)]
    max_contexts = [657]
    max_entities = [73]
    query_ids = [None]

# ent_selection_fns = ["top_entity_uri_degree", "top_entity_namesake_degree"]
ent_selection_fns = ["random_sample"]

# entity_types = json.dumps(
#     ["entities", "fake_entities"], separators=(",", ":")
# )  # separators is important to remove spaces from the string. This is important downstream for bash to be able to read the whole list.
entity_types = json.dumps(
    ["entities", "gpt_fake_entities"], separators=(",", ":")
)  # separators is important to remove spaces from the string. This is important downstream for bash to be able to read the whole list.
# query_types = json.dumps(
#     ["closed", "open"], separators=(",", ":")
# )  # separators is important to remove spaces from the string. This is important downstream for bash to be able to read the whole list.
query_types = json.dumps(
    ["closed", "open"], separators=(",", ":")
)  # separators is important to remove spaces from the string. This is important downstream for bash to be able to read the whole list.

answer_map = dict()
# answer_map = {0: [" No", " no", " NO", "No", "no", "NO"], 1: [" Yes", " yes", " YES", "Yes", "yes", "YES"]}

cap_per_type = False
ablate = False
deduplicate_entities = False
uniform_contexts = False
overwrite = True

compute_mr = False
batch_sz = 32


def convert_answer_map_to_tokens(model_id: str, answer_map: Dict[int, List[str]]) -> str:
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        padding_side="left",
    )

    answer_map_token_ids = dict()
    for k, v in answer_map.items():
        list_of_token_ids: List[List[str]] = tokenizer(v)["input_ids"]
        valid_token_ids = []
        for token_id in list_of_token_ids:
            if len(token_id) == 1:
                valid_token_ids.append(token_id[0])
            else:
                print(
                    f"tokenizer tokenized an answer map token into multiple tokens ({token_id}), which is invalid input."
                )
        answer_map_token_ids[k] = valid_token_ids
    #     answer_map_token_ids = {
    #         k: [x[0] for x in tokenizer(v)["input_ids"] if len(x) == 1],
    #         for k, v in answer_map.items()
    #     }
    res = json.dumps(answer_map_token_ids, separators=(",", ":"))
    print(res)
    return res


for ds, rdp in dataset_names_and_rdps:
    for seed in seeds:
        for model_id, do_quantize in model_id_and_quantize_tuples:
            answer_map_in_tokens = convert_answer_map_to_tokens(model_id, answer_map)
            for qid in query_ids:
                for mc in max_contexts:
                    for me in max_entities:
                        for es in ent_selection_fns:
                            if RUN_LOCALLY:
                                subprocess.run(
                                    [
                                        "python",
                                        "susceptibility_scores.py",
                                        f"{ds}",
                                        "-P",
                                        rdp,
                                        "-S",
                                        f"{seed}",
                                        "-M",
                                        f"{model_id}",
                                        "-Q",
                                        f"{qid}",
                                        "-MC",
                                        f"{mc}",
                                        "-ME",
                                        f"{me}",
                                        "-ET",
                                        f"{entity_types}",
                                        "-QT",
                                        f"{query_types}",
                                        "-AM",
                                        f"{answer_map_in_tokens}",
                                        "-ES",
                                        f"{es}",
                                        "-BS",
                                        f"{batch_sz}",
                                    ]
                                    + (["-B"] if do_quantize else [])
                                    + (["-A"] if ablate else [])
                                    + (["-T"] if cap_per_type else [])
                                    + (["-D"] if deduplicate_entities else [])
                                    + (["-U"] if uniform_contexts else [])
                                    + (["-O"] if overwrite else [])
                                )
                            else:
                                cmd = (
                                    [
                                        "sbatch",
                                        "slurm/susceptibility_scores/submit_susceptibility_score.cluster",
                                        f"{ds}",
                                        f"{rdp}",
                                        f"{seed}",
                                        f"{model_id}",
                                        f"{qid}",
                                        f"{mc}",
                                        f"{me}",
                                        f"{entity_types}",
                                        f"{query_types}",
                                        f"{answer_map_in_tokens}",
                                        f"{es}",
                                        f"{batch_sz}",
                                    ]
                                    + (["-B"] if do_quantize else [])
                                    + (["-A"] if ablate else [])
                                    + (["-T"] if cap_per_type else [])
                                    + (["-D"] if deduplicate_entities else [])
                                    + (["-U"] if uniform_contexts else [])
                                    + (["-O"] if overwrite else [])
                                )
                                print(cmd)
                                subprocess.check_call(cmd)
