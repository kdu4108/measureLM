import subprocess
import json

RUN_LOCALLY = False
YAGO_QEC_PATH = "data/YagoECQ/yago_qec.json"  # assuming you are running from the root project directory

with open(YAGO_QEC_PATH) as f:
    yago_qec = json.load(f)

dataset_names_and_rdps = [("YagoECQ", YAGO_QEC_PATH)]
seeds = [0]  # val every 10

if RUN_LOCALLY:
    model_id_and_quantize_tuples = [("EleutherAI/pythia-70m-deduped", False)]
    max_contexts = [10]
    max_entities = [5]
    query_ids = list(yago_qec.keys())[:5]
else:
    model_id_and_quantize_tuples = [("EleutherAI/pythia-6.9b-deduped", True)]
    max_contexts = [500]
    max_entities = [100]
    query_ids = list(yago_qec.keys())
    # query_ids = ["http://schema.org/founder"]

entity_types = json.dumps(
    ["entities", "fake_entities"], separators=(",", ":")
)  # separators is important to remove spaces from the string. This is important downstream for bash to be able to read the whole list.
query_types = json.dumps(
    ["closed", "open"], separators=(",", ":")
)  # separators is important to remove spaces from the string. This is important downstream for bash to be able to read the whole list.
cap_per_type = False
ablate = False

for ds, rdp in dataset_names_and_rdps:
    for seed in seeds:
        for model_id, do_quantize in model_id_and_quantize_tuples:
            for qid in query_ids:
                for mc in max_contexts:
                    for me in max_entities:
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
                                ]
                                + (["-B"] if do_quantize else [])
                                + (["-A"] if ablate else [])
                                + (["-T"] if cap_per_type else [])
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
                                ]
                                + (["-B"] if do_quantize else [])
                                + (["-A"] if ablate else [])
                                + (["-T"] if cap_per_type else [])
                            )
                            print(cmd)
                            subprocess.check_call(cmd)
